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
import numpy as np

from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid

from rocketqa.model.ernie import ErnieModel

def create_train_model(args,
                 pyreader_name,
                 ernie_config,
                 batch_size=16,
                 is_prediction=False,
                 task_name="",
                 fleet_handle=None):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1],
            [batch_size, 1], [batch_size, 1]],
        dtypes=['int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0],
        name=task_name + "_" + pyreader_name,
        use_double_buffer=True)

    (src_ids_q, sent_ids_q, pos_ids_q, task_ids_q, input_mask_q,
     src_ids_p_pos, sent_ids_p_pos, pos_ids_p_pos, task_ids_p_pos, input_mask_p_pos,
     src_ids_p_neg, sent_ids_p_neg, pos_ids_p_neg, task_ids_p_neg, input_mask_p_neg,
     labels, qids) = fluid.layers.read_file(pyreader)

    ernie_q = ErnieModel(
        src_ids=src_ids_q,
        position_ids=pos_ids_q,
        sentence_ids=sent_ids_q,
        task_ids=task_ids_q,
        input_mask=input_mask_q,
        config=ernie_config,
        model_name='query_')
    ## pos para
    ernie_pos = ErnieModel(
        src_ids=src_ids_p_pos,
        position_ids=pos_ids_p_pos,
        sentence_ids=sent_ids_p_pos,
        task_ids=task_ids_p_pos,
        input_mask=input_mask_p_pos,
        config=ernie_config,
        model_name='titlepara_')
    ## neg para
    ernie_neg = ErnieModel(
        src_ids=src_ids_p_neg,
        position_ids=pos_ids_p_neg,
        sentence_ids=sent_ids_p_neg,
        task_ids=task_ids_p_neg,
        input_mask=input_mask_p_neg,
        config=ernie_config,
        model_name='titlepara_')

    q_cls_feats = ernie_q.get_cls_output()
    pos_cls_feats = ernie_pos.get_cls_output()
    neg_cls_feats = ernie_neg.get_cls_output()
    #src_ids_p_pos = fluid.layers.Print(src_ids_p_pos, message='pos: ')
    #pos_cls_feats = fluid.layers.Print(pos_cls_feats, message='pos: ')

    p_cls_feats = fluid.layers.concat([pos_cls_feats, neg_cls_feats], axis=0)

    if is_prediction:
        p_cls_feats = fluid.layers.slice(p_cls_feats, axes=[0], starts=[0], ends=[batch_size])
        multi = fluid.layers.elementwise_mul(q_cls_feats, p_cls_feats)
        probs = fluid.layers.reduce_sum(multi, dim=-1)

        graph_vars = {
            "probs": probs,
            "q_rep": q_cls_feats,
            "p_rep": p_cls_feats
        }
        return pyreader, graph_vars

    if args.use_cross_batch and fleet_handle is not None:
        print("worker num is: {}".format(fleet_handle.worker_num()))
        all_p_cls_feats = fluid.layers.collective._c_allgather(
                p_cls_feats, fleet_handle.worker_num(), use_calc_stream=True)

        #multiply
        logits = fluid.layers.matmul(q_cls_feats, all_p_cls_feats, transpose_x=False, transpose_y=True)
        worker_id = fleet_handle.worker_index()

    else:
        logits = fluid.layers.matmul(q_cls_feats, p_cls_feats, transpose_x=False, transpose_y=True)
        worker_id = 0

    probs = logits

    all_labels = np.array(range(batch_size * worker_id * 2, batch_size * (worker_id * 2 + 1)), dtype='int64')
    matrix_labels = fluid.layers.assign(all_labels)
    matrix_labels = fluid.layers.unsqueeze(matrix_labels, axes=1)
    matrix_labels.stop_gradient=True

    ce_loss = fluid.layers.softmax_with_cross_entropy(
           logits=logits, label=matrix_labels)
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(
        input=probs, label=matrix_labels, total=num_seqs)

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "q_rep": q_cls_feats,
        "p_rep": p_cls_feats
    }

    return pyreader, graph_vars

