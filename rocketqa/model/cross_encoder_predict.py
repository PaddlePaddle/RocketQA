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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
import logging
import numpy as np

from six.moves import xrange
import paddle.fluid as fluid

from rocketqa.model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_predict_model(args,
                 pyreader_name,
                 ernie_config,
                 is_prediction=False,
                 task_name="",
                 joint_training=0):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1], [-1, 1]],
        dtypes=[
            'int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64'
        ],
        lod_levels=[0, 0, 0, 0, 0, 0, 0],
        name=task_name + "_" + pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    def _model(is_noise=False):
        if joint_training == 1:
            ernie = ErnieModel(
                src_ids=src_ids,
                position_ids=pos_ids,
                sentence_ids=sent_ids,
                task_ids=task_ids,
                input_mask=input_mask,
                config=ernie_config,
                is_noise=is_noise,
                model_name='qtp_')
            cls_feats = ernie.get_pooled_output(joint_training=1)
        else:
            ernie = ErnieModel(
                src_ids=src_ids,
                position_ids=pos_ids,
                sentence_ids=sent_ids,
                task_ids=task_ids,
                input_mask=input_mask,
                config=ernie_config,
                is_noise=is_noise)
            cls_feats = ernie.get_pooled_output()

        if not is_noise:
            cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        if joint_training == 1:
            logits = fluid.layers.fc(
                input=cls_feats,
                size=1,
                param_attr=fluid.ParamAttr(
                    name="qtp__cls_out_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name="qtp__cls_out_b",
                    initializer=fluid.initializer.Constant(0.)))
            probs = logits

        else:
            logits = fluid.layers.fc(
                input=cls_feats,
                size=args.num_labels,
                param_attr=fluid.ParamAttr(
                    name=task_name + "_cls_out_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name=task_name + "_cls_out_b",
                    initializer=fluid.initializer.Constant(0.)))
            probs = fluid.layers.softmax(logits)

        graph_vars = {
            "probs": probs,
        }
        return graph_vars


    if not is_prediction:
        graph_vars = _model(is_noise=True)
        old_loss = graph_vars["loss"]
        token_emb = fluid.default_main_program().global_block().var("word_embedding")
        token_emb.stop_gradient = False
        token_gradient = fluid.gradients(old_loss, token_emb)[0]
        token_gradient.stop_gradient = False
        epsilon = 1e-8
        norm = (fluid.layers.sqrt(
            fluid.layers.reduce_sum(fluid.layers.square(token_gradient)) + epsilon))
        gp = (0.01 * token_gradient) / norm
        gp.stop_gradient = True
        fluid.layers.assign(token_emb + gp, token_emb)
        graph_vars = _model()
        fluid.layers.assign(token_emb - gp, token_emb)
    else:
        graph_vars = _model()

    return pyreader, graph_vars

