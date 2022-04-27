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
                 share_parameter=0):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.q_max_seq_len, 1], [-1, args.q_max_seq_len, 1],
            [-1, args.q_max_seq_len, 1], [-1, args.q_max_seq_len, 1],
            [-1, args.q_max_seq_len, 1],
            [-1, args.p_max_seq_len, 1], [-1, args.p_max_seq_len, 1],
            [-1, args.p_max_seq_len, 1], [-1, args.p_max_seq_len, 1],
            [-1, args.p_max_seq_len, 1],
            [-1, 1], [-1, 1]],
    dtypes=['int64', 'int64', 'int64', 'int64', 'float32',
            'int64', 'int64', 'int64', 'int64', 'float32',
            'int64', 'int64'],
    lod_levels=[0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0],
    name=pyreader_name,
    use_double_buffer=True)

    (src_ids_q, sent_ids_q, pos_ids_q, task_ids_q, input_mask_q,
     src_ids_p, sent_ids_p, pos_ids_p, task_ids_p, input_mask_p,
     labels, qids) = fluid.layers.read_file(pyreader)

    if share_parameter == 0:
        ernie_q = ErnieModel(
            src_ids=src_ids_q,
            position_ids=pos_ids_q,
            sentence_ids=sent_ids_q,
            task_ids=task_ids_q,
            input_mask=input_mask_q,
            config=ernie_config,
            model_name='query_')
    else:
        ernie_q = ErnieModel(
            src_ids=src_ids_q,
            position_ids=pos_ids_q,
            sentence_ids=sent_ids_q,
            task_ids=task_ids_q,
            input_mask=input_mask_q,
            config=ernie_config,
            model_name='titlepara_')

    ## pos para
    ernie_p = ErnieModel(
        src_ids=src_ids_p,
        position_ids=pos_ids_p,
        sentence_ids=sent_ids_p,
        task_ids=task_ids_p,
        input_mask=input_mask_p,
        config=ernie_config,
        model_name='titlepara_')

    q_cls_feats = ernie_q.get_cls_output()
    p_cls_feats = ernie_p.get_cls_output()

    #multiply
    multi = fluid.layers.elementwise_mul(q_cls_feats, p_cls_feats)
    probs = fluid.layers.reduce_sum(multi, dim=-1)

    graph_vars = {
        "probs": probs,
        "q_rep": q_cls_feats,
        "p_rep": p_cls_feats
    }

    return pyreader, graph_vars
