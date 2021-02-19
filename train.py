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

r"""Training loop."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import shutil
import sys

import traceback  # pylint:disable=g-import-not-at-top
# 这个absl 在 tf.app.run()中对flags进行了初始化
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

# data_config和 model_config 定义了一堆flags
from stacked_capsule_autoencoders.capsules.configs import data_config
from stacked_capsule_autoencoders.capsules.configs import model_config
from stacked_capsule_autoencoders.capsules.train import create_hooks
from stacked_capsule_autoencoders.capsules.train import tools


flags.DEFINE_string('dataset', 'mnist', 'Choose from: {mnist, constellation.}')
flags.DEFINE_string('model', 'scae', 'Choose from {scae, constellation}.')


# name是必须要提供的, 作为flags
flags.DEFINE_string('name', 'mnist', '')
flags.mark_flag_as_required('name')

# logdir 这里 用gdrive可以用colab进行训练, 不过其实也可以 , 我们直接在本地目录下创建一下gdrive就好了
flags.DEFINE_string('logdir', 'gdrive/MyDrive/stacked_capsule_autoencoders/checkpoints/{name}',
                    'Log and checkpoint directory for the experiment.')

flags.DEFINE_float('grad_value_clip', 0., '')
flags.DEFINE_float('grad_norm_clip', 0., '')

flags.DEFINE_float('ema', .9, 'Exponential moving average weight for smoothing '
                   'reported results.')

flags.DEFINE_integer('run_updates_every', 10, '')
flags.DEFINE_boolean('global_ema_update', True, '')

# train 300K 步 用的好像是global step
flags.DEFINE_integer('max_train_steps', int(3e5), '')

# snapshot 是每 3600面来一下
flags.DEFINE_integer('snapshot_secs', 3600, '')
flags.DEFINE_integer('snapshot_steps', 0, '')
flags.DEFINE_integer('snapshots_to_keep', 5, '')
# 这个应该是tensorboard的summary吧
flags.DEFINE_integer('summary_steps', 500, '')

# 这个可以改小一点, 不过得看看loss是不是算的 test loss 要不然有性能损失
flags.DEFINE_integer('report_loss_steps', 500, '')

# 哪里进行了 plot?
flags.DEFINE_boolean('plot', False, 'Produces intermediate results plots '
                     'if True.')
flags.DEFINE_integer('plot_steps', 1000, '')

# overwrite 一定要 False 决定是否吧checkpoint 进行重写
flags.DEFINE_boolean('overwrite', False, 'Overwrites any existing run of the '
                     'same name if True; otherwise it tries to restore the '
                     'model if a checkpoint exists.')

# ???
flags.DEFINE_boolean('check_numerics', False, 'Adds check numerics ops.')


def main(_=None):
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name,redefined-outer-name
  # 这里的config 没看懂在干嘛, 好像是便于传FLAGS ???
  config = FLAGS
  FLAGS.__dict__['config'] = config

  # 这个写 logdir 的方式挺好的, 用 %s 后面用format 字符串
  FLAGS.logdir = FLAGS.logdir.format(name=FLAGS.name)

  logdir = FLAGS.logdir
  # info 可以当做是print 来用么? 像是 c++ 的 print 而不是 python的把
  logging.info('logdir: %s', logdir)

  # 一定不要 overwrite 不然不能继续训练, 不过我还没找到继续训练的代码在哪里
  if os.path.exists(logdir) and FLAGS.overwrite:
    logging.info('"overwrite" is set to True. Deleting logdir at "%s".', logdir)
    shutil.rmtree(logdir)

  # 用默认图 好像是可以共享么? 有什么好处啊
  # Build the graph
  with tf.Graph().as_default():

    # !!!!! 构造 model 是这里最重要的部分
    model_dict = model_config.get(FLAGS)
    data_dict = data_config.get(FLAGS)

    lr = model_dict.lr
    opt = model_dict.opt
    model = model_dict.model
    trainset = data_dict.trainset
    validset = data_dict.validset

    lr = tf.convert_to_tensor(lr)
    tf.summary.scalar('learning_rate', lr)

    # Training setup
    global_step = tf.train.get_or_create_global_step()

    # Optimisation target
    validset = tools.maybe_convert_dataset(validset)
    trainset = tools.maybe_convert_dataset(trainset)

    target, gvs = model.make_target(trainset, opt)

    if gvs is None:
      gvs = opt.compute_gradients(target)

    suppress_inf_and_nans = (config.grad_value_clip > 0
                             or config.grad_norm_clip > 0)
    report = tools.gradient_summaries(gvs, suppress_inf_and_nans)
    report['target'] = target
    valid_report = dict()

    gvs = tools.clip_gradients(gvs, value_clip=config.grad_value_clip,
                               norm_clip=config.grad_norm_clip)

    try:
      report.update(model.make_report(trainset))
      valid_report.update(model.make_report(validset))
    except AttributeError:
      logging.warning('Model %s has no "make_report" method.', str(model))
      raise

    plot_dict, plot_params = None, None
    if config.plot:
      try:
        plot_dict, plot_params = model.make_plot(trainset, 'train')
        valid_plot, valid_params = model.make_plot(validset, 'valid')

        plot_dict.update(valid_plot)
        if plot_params is not None:
          plot_params.update(valid_params)

      except AttributeError:
        logging.warning('Model %s has no "make_plot" method.', str(model))

    report = tools.scalar_logs(report, config.ema, 'train',
                               global_update=config.global_ema_update)
    report['lr'] = lr
    valid_report = tools.scalar_logs(
        valid_report, config.ema, 'valid',
        global_update=config.global_ema_update)

    reports_keys = sorted(report.keys())

    def _format(k):
      if k in ('lr', 'learning_rate'):
        return '.2E'
      return '.3f'

    report_template = ', '.join(['{}: {}{}:{}{}'.format(
        k, '{', k, _format(k), '}') for k in reports_keys])

    logging.info('Trainable variables:')
    tools.log_variables_by_scope()

    # inspect gradients
    for g, v in gvs:
      if g is None:
        logging.warning('No gradient for variable: %s.', v.name)

    tools.log_num_params()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if FLAGS.check_numerics:
      update_ops += [tf.add_check_numerics_ops()]

    with tf.control_dependencies(update_ops):
      train_step = opt.apply_gradients(gvs, global_step=global_step)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.train.SingularMonitoredSession(
        hooks=create_hooks(FLAGS, plot_dict, plot_params),
        checkpoint_dir=logdir, config=sess_config) as sess:

      train_itr, _ = sess.run([global_step, update_ops])
      train_tensors = [global_step, train_step]
      report_tensors = [report, valid_report]
      all_tensors = report_tensors + train_tensors

      while train_itr < config.max_train_steps:

        if train_itr % config.report_loss_steps == 0:
          report_vals, valid_report_vals, train_itr, _ = sess.run(all_tensors)

          logging.info('')
          logging.info('train:')
          logging.info('#%s: %s', train_itr,
                       report_template.format(**report_vals))

          logging.info('valid:')
          valid_logs = dict(report_vals)
          valid_logs.update(valid_report_vals)
          logging.info('#%s: %s', train_itr,
                       report_template.format(**valid_logs))

          vals_to_check = list(report_vals.values())
          if (np.isnan(vals_to_check).any()
              or np.isnan(vals_to_check).any()):
            logging.fatal('NaN in reports: %s; breaking...',
                          report_template.format(**report_vals))

        else:
          train_itr, _ = sess.run(train_tensors)


if __name__ == '__main__':
  try:
    # INFO 的信息就能提醒出来了
    # 用的是 absl.logging 这个库挺好的 出来是红色字体
    logging.set_verbosity(logging.INFO)
    # 直接处理 flags, 调用main 函数
    tf.app.run()
  ## 每次把pdb调试调出来就是因为 google的这个模板
  except Exception as err:  # pylint: disable=broad-except
    FLAGS = flags.FLAGS

    last_traceback = sys.exc_info()[2]
    traceback.print_tb(last_traceback)
    print(err)
    pdb.post_mortem(last_traceback)
