# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Primary capsuls."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from monty.collections import AttrDict
import numpy as np

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from stacked_capsule_autoencoders.capsules import math_ops
from stacked_capsule_autoencoders.capsules import neural
from stacked_capsule_autoencoders.capsules import prob
from stacked_capsule_autoencoders.capsules.tensor_ops import make_brodcastable

tfd = tfp.distributions


class CapsuleImageEncoder(snt.AbstractModule):
  """Primary capsule for images."""
  OutputTuple = collections.namedtuple(  # pylint:disable=invalid-name
      'PrimaryCapsuleTuple',
      'pose feature presence presence_logit '
      'img_embedding')

  def __init__(self,
               encoder,
               n_caps,
               n_caps_dims,
               n_features=0,
               noise_scale=4.,
               similarity_transform=False,
               encoder_type='linear',
               **encoder_kwargs):

    super(CapsuleImageEncoder, self).__init__()
    self._encoder = encoder # cnn_encoder 那个4层的
    self._n_caps = n_caps # config.n_part_caps,
    self._n_caps_dims = n_caps_dims # config.n_part_caps_dims,
    self._n_features = n_features #  config.n_part_special_features,
    self._noise_scale = noise_scale # 4.0
    self._similarity_transform = similarity_transform # False
    self._encoder_type = encoder_type #'conv_att'
    self._encoder_kwargs = dict(
        n_layers=2, n_heads=4, n_dims=32, layer_norm=False)
    self._encoder_kwargs.update(encoder_kwargs)

  def _build(self, x):
    batch_size = x.shape[0] # x.shape=[128,40,40,1]
    img_embedding = self._encoder(x) # cnn_encoder对输入进行 embedding
              # img_embedding.shape=[128,5,5,128]
    splits = [self._n_caps_dims, self._n_features, 1]  # 1 for presence #[6,16,1]
    n_dims = sum(splits) # 由 caps_dim + feature_dim + presence 组成 =25

    if self._encoder_type == 'linear':
      ## 在 mnist 中 _n_caps 用的是 40
      n_outputs = self._n_caps * n_dims # 出来的维度

      h = snt.BatchFlatten()(img_embedding)  #debug 一下这个flatten 为啥要写成这样子?
      h = snt.Linear(n_outputs)(h) # h 通过一下这个linear 就是说的那个 MLP 吧

    else:
      h = snt.AddBias(bias_dims=[1, 2, 3])(img_embedding) #所有维度都加上 bias

      if self._encoder_type == 'conv':
        h = snt.Conv2D(n_dims * self._n_caps, 1, 1)(h)
        h = tf.reduce_mean(h, (1, 2))
        h = tf.reshape(h, [batch_size, self._n_caps, n_dims])

      elif self._encoder_type == 'conv_att': # 选择这个
        h = snt.Conv2D(n_dims * self._n_caps + self._n_caps, 1, 1)(h) # conv 最后多了个 _n_caps  最后多一个输出 #(128,5,5,960)
        h = snt.MergeDims(1, 2)(h) # 合并h 第一二维的数据 #(128,25,960)
        h, a = tf.split(h, [n_dims * self._n_caps, self._n_caps], -1) #从最后一维分开 对应上面多一个 #(128,25,920) (128,25,40)

        h = tf.reshape(h, [batch_size, -1, n_dims, self._n_caps]) #(128,25,23,40)
        a = tf.nn.softmax(a, 1)  # presence? ?啥意思呢? (128,25,40)  25好像是出来的 5*5的卷积, 合并起来 然后求softmax
        a = tf.reshape(a, [batch_size, -1, 1, self._n_caps]) #(128,25,1,40)
        h = tf.reduce_sum(h * a, 1) #(128,23,40) 真的是概率 乘一下 然后加起来啊

      else:
        raise ValueError('Invalid encoder type="{}".'.format(
            self._encoder_type))

    h = tf.reshape(h, [batch_size, self._n_caps, n_dims]) #(128,40,23) ??
    # 我有点不理解为啥前面要写那么复杂啊...
    pose, feature, pres_logit = tf.split(h, splits, -1) # important !!!!!!! 直接出来结果了 #(128,40,6) (128,40,16) (128,40,1)

    if self._n_features == 0:
      feature = None

    pres_logit = tf.squeeze(pres_logit, -1) # 把最后一维压缩掉 (128,40)
    if self._noise_scale > 0.:
      pres_logit += ((tf.random.uniform(pres_logit.shape) - .5)
                     * self._noise_scale) # 加上一个随机的噪声? 原来的数值有多大呢? 就要加 0.5 均值的噪声啊


    pres = tf.nn.sigmoid(pres_logit) # (128,40)
    pose = math_ops.geometric_transform(pose, self._similarity_transform) # (128,40,6) 不变,  看看里面干了点啥?
    return self.OutputTuple(pose, feature, pres, pres_logit, img_embedding) # 返回这样的 named Tuple 好像多了一个维度


def choose_nonlinearity(name):
  nonlin = getattr(math_ops, name, getattr(tf.nn, name, None))

  if not nonlin:
    raise ValueError('Invalid nonlinearity: "{}".'.format(name))

  return nonlin


class TemplateBasedImageDecoder(snt.AbstractModule):
  """Template-based primary capsule decoder for images."""

  _templates = None

  def __init__(self,
               output_size,
               template_size,
               n_channels=1,
               learn_output_scale=False,
               colorize_templates=False,
               output_pdf_type='mixture',
               template_nonlin='relu1',
               color_nonlin='relu1',
               use_alpha_channel=False):

    super(TemplateBasedImageDecoder, self).__init__()
    self._output_size = output_size
    self._template_size = template_size
    self._n_channels = n_channels
    self._learn_output_scale = learn_output_scale
    self._colorize_templates = colorize_templates


    self._output_pdf_type = output_pdf_type
    self._template_nonlin = choose_nonlinearity(template_nonlin)
    self._color_nonlin = choose_nonlinearity(color_nonlin)
    self._use_alpha_channel = use_alpha_channel

  @property
  def templates(self):
    self._ensure_is_connected()
    return tf.squeeze(self._templates, 0)

  @snt.reuse_variables
  def make_templates(self, n_templates=None, template_feature=None):

    if self._templates is not None:
      if n_templates is not None and self._templates.shape[1] != n_templates:
        raise ValueError

    else:
      with self._enter_variable_scope():
        # create templates
        n_dims = self._n_channels

        template_shape = ([1, n_templates] + list(self._template_size) +
                          [n_dims])
        n_elems = np.prod(template_shape[2:])

        # make each templates orthogonal to each other at init
        n = max(n_templates, n_elems)
        q = np.random.uniform(size=[n, n])
        q = np.linalg.qr(q)[0]
        q = q[:n_templates, :n_elems].reshape(template_shape).astype(np.float32)

        q = (q - q.min()) / (q.max() - q.min())

        template_logits = tf.get_variable('templates', initializer=q)
        # prevent negative ink
        self._template_logits = template_logits
        self._templates = self._template_nonlin(template_logits)

        if self._use_alpha_channel:
          self._templates_alpha = tf.get_variable(
              'templates_alpha',
              shape=self._templates[Ellipsis, :1].shape,
              initializer=tf.zeros_initializer())

        self._n_templates = n_templates

    templates = self._templates
    if template_feature is not None:


      if self._colorize_templates:
        mlp = snt.BatchApply(snt.nets.MLP([32, self._n_channels]))
        template_color = mlp(template_feature)[:, :, tf.newaxis, tf.newaxis]

        if self._color_nonlin == math_ops.relu1:
          template_color += .99

        template_color = self._color_nonlin(template_color)
        templates = tf.identity(templates) * template_color

    return templates

  def _build(self,
             pose,
             presence=None,
             template_feature=None,
             bg_image=None,
             img_embedding=None):
    """Builds the module.

    Args:
      pose: [B, n_templates, 6] tensor.
      presence: [B, n_templates] tensor.
      template_feature: [B, n_templates, n_features] tensor; these features are
        used to change templates based on the input, if present.
      bg_image: [B, *output_size] tensor representing the background.
      img_embedding: [B, d] tensor containing image embeddings.

    Returns:
      [B, n_templates, *output_size, n_channels] tensor.
    """
    batch_size, n_templates = pose.shape[:2].as_list()
    templates = self.make_templates(n_templates, template_feature)

    if templates.shape[0] == 1:
      templates = snt.TileByDim([0], [batch_size])(templates)

    # it's easier for me to think in inverse coordinates
    warper = snt.AffineGridWarper(self._output_size, self._template_size)
    warper = warper.inverse()

    grid_coords = snt.BatchApply(warper)(pose)
    resampler = snt.BatchApply(tf.contrib.resampler.resampler)
    transformed_templates = resampler(templates, grid_coords)

    if bg_image is not None:
      bg_image = tf.expand_dims(bg_image, axis=1)
    else:
      bg_image = tf.nn.sigmoid(tf.get_variable('bg_value', shape=[1]))
      bg_image = tf.zeros_like(transformed_templates[:, :1]) + bg_image

    transformed_templates = tf.concat([transformed_templates, bg_image], axis=1)

    if presence is not None:
      presence = tf.concat([presence, tf.ones([batch_size, 1])], axis=1)

    if True:  # pylint: disable=using-constant-test

      if self._use_alpha_channel:
        template_mixing_logits = snt.TileByDim([0], [batch_size])(
            self._templates_alpha)
        template_mixing_logits = resampler(template_mixing_logits, grid_coords)

        bg_mixing_logit = tf.nn.softplus(
            tf.get_variable('bg_mixing_logit', initializer=[0.]))

        bg_mixing_logit = (
            tf.zeros_like(template_mixing_logits[:, :1]) + bg_mixing_logit)

        template_mixing_logits = tf.concat(
            [template_mixing_logits, bg_mixing_logit], 1)

      else:
        temperature_logit = tf.get_variable('temperature_logit', shape=[1])
        temperature = tf.nn.softplus(temperature_logit + .5) + 1e-4
        template_mixing_logits = transformed_templates / temperature

    scale = 1.
    if self._learn_output_scale:
      scale = tf.get_variable('scale', shape=[1])
      scale = tf.nn.softplus(scale) + 1e-4

    if self._output_pdf_type == 'mixture':
      template_mixing_logits += make_brodcastable(
          math_ops.safe_log(presence), template_mixing_logits)

      rec_pdf = prob.MixtureDistribution(template_mixing_logits,
                                         [transformed_templates, scale],
                                         tfd.Normal)


    else:
      raise ValueError('Unknown pdf type: "{}".'.format(self._output_pdf_type))

    return AttrDict(
        raw_templates=tf.squeeze(self._templates, 0),
        transformed_templates=transformed_templates[:, :-1],
        mixing_logits=template_mixing_logits[:, :-1],
        pdf=rec_pdf)

