#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import json
import math
import logging
import time
import multiprocessing
import numpy as np

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

from rocketqa.reader import reader_de_predict, reader_de_train
from rocketqa.model.ernie import ErnieConfig
from rocketqa.model.dual_encoder_predict import create_predict_model
from rocketqa.model.dual_encoder_train import create_train_model
from rocketqa.utils.args import print_arguments, check_cuda, prepare_logger
from rocketqa.utils.init import init_pretraining_params, init_checkpoint
from rocketqa.utils.finetune_args import parser
from rocketqa.utils.optimization import optimization


class DualEncoder(object):

    def __init__(self, conf_path, use_cuda=False, device_id=0, batch_size=1, **kwargs):
        if "model_path" in kwargs:
            args = self._parse_args(conf_path, model_path=kwargs["model_path"])
        else:
            args = self._parse_args(conf_path)
        if "model_name" in kwargs:
            args.model_name = kwargs["model_name"].replace('/', '-')
        else:
            args.model_name = "my_de"
        args.use_cuda = use_cuda
        self.ernie_config = ErnieConfig(args.ernie_config_path)
        args.batch_size = batch_size

        if args.use_cuda:
            dev_list = fluid.cuda_places()
            place = dev_list[device_id]
            dev_count = 1
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        self.exe = fluid.Executor(place)

        self.predict_reader = reader_de_predict.DEPredictorReader(
            vocab_path=args.vocab_path,
            label_map_config=args.label_map_config,
            q_max_seq_len=args.q_max_seq_len,
            p_max_seq_len=args.p_max_seq_len,
            do_lower_case=args.do_lower_case,
            in_tokens=args.in_tokens,
            random_seed=args.random_seed,
            tokenizer=args.tokenizer,
            for_cn=args.for_cn,
            task_id=args.task_id)

        self.startup_prog = fluid.Program()

        self.test_prog = fluid.Program()
        with fluid.program_guard(self.test_prog, self.startup_prog):
            with fluid.unique_name.guard():
                self.test_pyreader, self.graph_vars = create_predict_model(
                    args,
                    pyreader_name=args.model_name + '_test_reader',
                    ernie_config=self.ernie_config,
                    is_prediction=True,
                    share_parameter=args.share_parameter)

        self.test_prog = self.test_prog.clone(for_test=True)

        self.exe.run(self.startup_prog)

        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                            "only doing validation or testing!")
        init_pretraining_params(
            self.exe,
            args.init_checkpoint,
            main_program=self.startup_prog)
        self.args = args


    def _parse_args(self, conf_path, model_path=''):
        args, unknown = parser.parse_known_args()
        with open(conf_path, 'r', encoding='utf8') as json_file:
            config_dict = json.load(json_file)

        args.do_train = False
        args.do_val = False
        args.do_test = True
        args.use_fast_executor = True
        args.q_max_seq_len = config_dict['q_max_seq_len']
        args.p_max_seq_len = config_dict['p_max_seq_len']
        args.ernie_config_path = model_path + config_dict['model_conf_path']
        args.vocab_path = model_path + config_dict['model_vocab_path']
        args.init_checkpoint = model_path + config_dict['model_checkpoint_path']
        if "for_cn" in config_dict:
            args.for_cn = config_dict["for_cn"]
        if 'share_parameter' in config_dict:
            args.share_parameter = config_dict['share_parameter']
        else:
            args.share_parameter = 0

        return args

    def _parse_train_args(self, train_set, epoch, save_model_path, config_dict):

        self.args.train_set = train_set
        self.args.save_model_path = save_model_path
        self.args.epoch = epoch

        if "save_steps" in config_dict:
            self.args.save_steps = config_dict['save_steps']
        else:
            self.args.save_steps = 0

        if "batch_size" in config_dict:
            self.args.batch_size = config_dict['batch_size']
        if 'learning_rate' in config_dict:
            self.args.learning_rate = config_dict['learning_rate']
        else:
            self.args.learning_rate = 2e-5
        if 'log_folder' in config_dict:
            self.args.log_folder = config_dict['log_folder']

    def encode_query(self, query):

        data = []
        for q in query:
            data.append(q + '\t-\t-')

        self.test_pyreader.decorate_tensor_provider(
            self.predict_reader.data_generator(
                data,
                self.args.batch_size,
                shuffle=False))

        self.test_pyreader.start()
        fetch_list = [self.graph_vars["q_rep"]]

        while True:
            try:
                q_rep = self.exe.run(program=self.test_prog,
                                                fetch_list=fetch_list)
                for data_q_rep in q_rep[0]:
                    yield data_q_rep
            except fluid.core.EOFException:
                self.test_pyreader.reset()
                break

        return


    def encode_para(self, para, title=[]):

        data = []
        if len(title) != 0:
            assert (len(para) == len(title)), "The input para(List) and title(List) should be the same length"
            for t, p in zip(title, para):
                data.append('-\t' + t + '\t' + p)
        else:
            for p in para:
                data.append('-\t\t' + p)

        self.test_pyreader.decorate_tensor_provider(
            self.predict_reader.data_generator(
                data,
                self.args.batch_size,
                shuffle=False))

        self.test_pyreader.start()
        fetch_list = [self.graph_vars["p_rep"]]

        while True:
            try:
                p_rep = self.exe.run(program=self.test_prog,
                                                fetch_list=fetch_list)
                for data_p_rep in p_rep[0]:
                    yield data_p_rep
            except fluid.core.EOFException:
                self.test_pyreader.reset()
                break

        return


    def matching(self, query, para, title=[]):

        data = []
        assert (len(para) == len(query)), "The input query(List) and para(List) should be the same length"
        if len(title) != 0:
            assert (len(para) == len(title)), "The input query(List) and para(List) should be the same length"
            for q, t, p in zip(query, title, para):
                data.append(q + '\t' + t + '\t' + p)
        else:
            for q, p in zip(query, para):
                data.append(q + '\t\t' + p)

        self.test_pyreader.decorate_tensor_provider(
            self.predict_reader.data_generator(
                data,
                self.args.batch_size,
                shuffle=False))

        self.test_pyreader.start()
        fetch_list = [self.graph_vars["probs"]]
        inner_probs = []

        while True:
            try:
                probs = self.exe.run(program=self.test_prog,
                                                fetch_list=fetch_list)
                #inner_probs.extend(probs[0].tolist())
                for data_prob in probs[0].tolist():
                    yield data_prob
            except fluid.core.EOFException:
                self.test_pyreader.reset()
                break

        return

    def train(self, train_set, epoch, save_model_path, **kwargs):
        self._parse_train_args(train_set, epoch, save_model_path, kwargs)
        check_cuda(self.args.use_cuda)
        log = logging.getLogger()
        dev_count = 1
        if self.args.log_folder == '':
            self.args.log_folder = '.'
        if not os.path.exists(self.args.log_folder):
            os.makedirs(self.args.log_folder)
        prepare_logger(log, save_to_file=self.args.log_folder + '/log.train')
        print_arguments(self.args, log)

        reader = reader_de_train.DETrainReader(
            vocab_path=self.args.vocab_path,
            label_map_config=self.args.label_map_config,
            q_max_seq_len=self.args.q_max_seq_len,
            p_max_seq_len=self.args.p_max_seq_len,
            total_num=self.args.train_data_size,
            do_lower_case=self.args.do_lower_case,
            in_tokens=self.args.in_tokens,
            random_seed=self.args.random_seed,
            tokenizer=self.args.tokenizer,
            for_cn=self.args.for_cn,
            task_id=self.args.task_id)

        startup_prog = fluid.Program()
        if self.args.random_seed is not None:
            startup_prog.random_seed = self.args.random_seed

        train_data_generator = reader.data_generator(
            input_file=self.args.train_set,
            batch_size=self.args.batch_size,
            epoch=self.args.epoch,
            dev_count=dev_count,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(self.args.train_set)
        if self.args.save_steps == 0:
            self.args.save_steps = int(math.ceil(num_train_examples * self.args.epoch / self.args.batch_size / 2))

        max_train_steps = self.args.epoch * num_train_examples // self.args.batch_size // dev_count

        warmup_steps = int(max_train_steps * self.args.warmup_proportion)
        log.info("Device count: %d" % dev_count)
        log.info("Num train examples: %d" % num_train_examples)
        log.info("Max train steps: %d" % max_train_steps)
        log.info("Num warmup steps: %d" % warmup_steps)
        log.info("Learning rate: %f" % self.args.learning_rate)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_train_model(
                    self.args,
                    pyreader_name='train_reader',
                    ernie_config=self.ernie_config,
                    batch_size=self.args.batch_size)
                scheduled_lr = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=self.args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=self.args.weight_decay,
                    scheduler=self.args.lr_scheduler,
		            use_dynamic_loss_scaling=self.args.use_dynamic_loss_scaling,
		            incr_every_n_steps=self.args.incr_every_n_steps,
		            decr_every_n_nan_or_inf=self.args.decr_every_n_nan_or_inf,
		            incr_ratio=self.args.incr_ratio,
		            decr_ratio=self.args.decr_ratio)

        self.exe.run(startup_prog)

        init_pretraining_params(
            self.exe,
            self.args.init_checkpoint,
            main_program=startup_prog)
        train_pyreader.decorate_tensor_provider(train_data_generator)
        train_pyreader.start()
        steps = 0
        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr

        time_begin = time.time()
        last_epoch = 0
        current_epoch = 0
        total_loss = []
        while True:
            try:
                steps += 1
                if steps % self.args.skip_steps != 0:
                    self.exe.run(fetch_list=[], program=train_program)
                else:
                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, current_epoch = reader.get_train_progress()
                    train_fetch_list = [
                        graph_vars["loss"], graph_vars["accuracy"]
                    ]
                    outputs = self.exe.run(fetch_list=train_fetch_list, program=train_program)
                    tmp_loss = np.mean(outputs[0])
                    tmp_acc = np.mean(outputs[1])
                    total_loss.append(tmp_loss)

                    if self.args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            outputs["learning_rate"]
                            if warmup_steps > 0 else self.args.learning_rate)
                        log.info(verbose)

                    log.info(
                        "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                        "ave acc: %f, speed: %f steps/s" %
                        (current_epoch, current_example * dev_count, num_train_examples,
                         steps, np.mean(total_loss), tmp_acc,
                         self.args.skip_steps / used_time))

                    time_begin = time.time()

                if steps % self.args.save_steps == 0:
                    save_path = os.path.join(self.args.save_model_path,
                                            "step_" + str(steps))
                    fluid.io.save_persistables(self.exe, save_path, train_program)

                if last_epoch != current_epoch:
                    last_epoch = current_epoch

            except fluid.core.EOFException:
                save_path = os.path.join(self.args.save_model_path, "step_" + str(steps))
                fluid.io.save_persistables(self.exe, save_path, train_program)
                train_pyreader.reset()
                break

