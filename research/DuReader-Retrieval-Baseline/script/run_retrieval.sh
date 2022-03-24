#!/bin/bash
set -x

if [ $# != 4 ];then
    echo "USAGE: sh script/run_retrieval.sh \$QUERY_FILE \$MODEL_PATH \$DATA_PATH \$TOP_K"
    exit 1
fi

QUERY_FILE=$1
MODEL_PATH=$2
DATA_PATH=$3
TOP_K=$4

sh script/run_dual_encoder_inference.sh 0 q $MODEL_PATH $QUERY_FILE

for card in {0..3};do
    nohup sh script/run_dual_encoder_inference.sh ${card} ${card} $MODEL_PATH $DATA_PATH &
    pid[$card]=$!
    echo $card start: pid=$! >> output/test.log
done
wait

for part in {0..3};do
    nohup python src/index_search.py $part $TOP_K $QUERY_FILE >> output/test.log &
done
wait

para_part_cnt=`cat $DATA_PATH/part-00 | wc -l`
python src/merge.py $para_part_cnt $TOP_K 4 >> output/test.log
rm -rf output/res.top${TOP_K}-part*
