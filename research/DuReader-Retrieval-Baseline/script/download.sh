#!/bin/bash

mkdir -p finetuned-models
mkdir -p pretrained-models

echo "Download DuReader_retrieval dataset"
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/dureader-retrieval-baseline-dataset.tar.gz
tar -zxvf dureader-retrieval-baseline-dataset.tar.gz
rm dureader-retrieval-baseline-dataset.tar.gz

echo "Download pre-trained model (ERNIE1.0)"
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/ernie_base_1.0_CN.tar.gz
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/ernie_base_1.0_twin_CN.tar.gz
tar -zxvf ernie_base_1.0_CN.tar.gz
tar -zxvf ernie_base_1.0_twin_CN.tar.gz
mv ernie_base_1.0_CN pretrained-models
mv ernie_base_1.0_twin_CN pretrained-models/
rm ernie_base_1.0_CN.tar.gz ernie_base_1.0_twin_CN.tar.gz

echo "Download fine-tuned models (dual- and cross-encoder)"
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/dual_finetuned_params.tar.gz
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/cross_finetuned_params.tar.gz
tar -zxvf dual_finetuned_params.tar.gz
tar -zxvf cross_finetuned_params.tar.gz
mv dual_params finetuned-models/
mv cross_params finetuned-models/
rm dual_finetuned_params.tar.gz cross_finetuned_params.tar.gz
