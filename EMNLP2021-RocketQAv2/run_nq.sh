#!/bin/bash
# This script contains the entire process of RocketQAv2. You can reproduce the results of the paper base on these processes.

WARM_JOINT='../checkpoint/nq_joint-encoder_warmup'
DATA_PATH='../corpus/nq'

output_file_ce='data_train/nq_joint.rand32+distill32'

## Joint-Training
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET=../$output_file_ce
MODEL_PATH=$WARM_JOINT
sh script/run_joint_model_train.sh $TRAIN_SET $MODEL_PATH 3 8 32

### Retrieval Inference
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/nq/test.query.txt'
MODEL_PATH='../checkpoint/nq_joint-encoder_trained'
TOP_K=100
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### Retrieval Evaluation
recall_topk_test='model/output/res.top100'
python metric/nq_eval.py $recall_topk_test


## Re-Ranking Inference
cd model
export CUDA_VISIBLE_DEVICES=0
TEST_SET='recall_top100_testset.tsv' # use $recall_topk_test to get top100 textual query-paras.
MODEL_PATH='../checkpoint/nq_joint_trained'
sh script/run_reranking.sh $TEST_SET $MODEL_PATH

### Re-Ranking Evaluation
recall_topk_test='model/output/res.top100'
cand_score='model/output/cross-encoder_score'
python metric/nq_eval_rerank.py $cand_score $recall_topk_test
