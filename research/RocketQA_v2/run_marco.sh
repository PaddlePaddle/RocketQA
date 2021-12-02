#!/bin/bash
# This script contains the entire process of RocketQAv2. You can reproduce the results of the paper base on these processes.

WARM_JOINT='../checkpoint/marco_joint-encoder_warmup'
DATA_PATH='../corpus/marco'

output_file_ce='data_train/marco_joint.rand128+aug128'

### Joint-Training (needs 32 cards training in paper)
cd model
TRAIN_SET=../$output_file_ce
ODEL_PATH=$WARM_JOINT
sh script/run_joint_model_train.sh $TRAIN_SET $MODEL_PATH 2 8 128

### Retrieval Inference
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/marco/dev.query.txt'
MODEL_PATH='../checkpoint/marco_joint-encoder_trained'
TOP_K=1000
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### Retrieval Evaluation
recall_topk_dev='model/output/res.top1000'
python metric/msmarco_eval.py corpus/marco/qrels.dev.tsv $recall_topk_dev


### Re-Ranking Inference
cd model
export CUDA_VISIBLE_DEVICES=0
TEST_SET='recall_top50_devset.tsv' # use $recall_topk_dev to get top50 textual query-paras.
MODEL_PATH='../checkpoint/marco_joint-encoder_trained'
sh script/run_reranking.sh $TEST_SET $MODEL_PATH

### Re-Ranking Evaluation
recall_topk_dev='model/output/res.top50' #use recall_topk_dev to get top50 results
cand_score='model/output/cross-encoder_score'
python metric/generate_candrank.py $cand_score $recall_topk_dev
python metric/msmarco_eval.py corpus/marco/qrels.dev.tsv metric/ranking_res
