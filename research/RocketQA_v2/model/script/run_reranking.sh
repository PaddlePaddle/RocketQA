#!/bin/bash
set -x

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95

export GLOG_v=1

if [ $# != 2 ];then
    echo "USAGE: sh script/run_cross_encoder_train.sh \$TEST_SET \$MODEL_PATH"
    exit 1
fi

TEST_SET=$1
MODEL_PATH=$2
epoch=1
node=1

CHECKPOINT_PATH=output
if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

lr=1e-5
batch_size=160
train_exampls=`cat $TRAIN_SET | wc -l`
save_steps=$[$train_exampls/$batch_size/$node]
data_size=$[$save_steps*$batch_size*$node]
new_save_steps=$[$save_steps]

python -m paddle.distributed.launch \
    --cluster_node_ips $(hostname -i) \
    --node_ip $(hostname -i) \
    --log_dir log \
    ./src/train_je.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --use_mix_precision true \
                   --use_recompute true \
                   --p_cnt_per_q 8 \
                   --train_data_size ${data_size} \
                   --batch_size ${batch_size} \
                   --init_checkpoint ${MODEL_PATH} \
                   --train_set none \
                   --test_set $TEST_SET \
                   --test_save ./output/score \
                   --save_steps ${new_save_steps} \
                   --validation_steps ${new_save_steps} \
                   --checkpoints ${CHECKPOINT_PATH} \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --epoch $epoch \
                   --q_max_seq_len 32 \
                   --p_max_seq_len 128 \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/base/ernie_config.json \
                   --learning_rate ${lr} \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --for_cn false \
                   --random_seed 1 \
                   1>>log/train.log 2>&1

