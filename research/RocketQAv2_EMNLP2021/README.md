# EMNLP2021-RocketQAv2

This is a repository of the paper: [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/abs/2110.07367), EMNLP 2021. 

## Introduction
PAIR is a novel joint training approach for dense passage retrieval and passage reranking. A major contribution is that we introduce the dynamic listwise distillation, where we design a unified listwise training approach for both the retriever and the re-ranker.
Extensive experiments show the effectiveness of our approach on both MSMARCO and Natural Questions datasets.

The pipeline of RocketQAv2 training approach is shown as follows:
![RocketQAv2-Pipeline](pipeline.png)


## Preparation
### Environment
* Python 3.7
* PaddlePaddle 1.8 (Please refer to the [Installation Guide](http://www.paddlepaddle.org/#quick-start)) 
* cuda >= 9.0  
* cudnn >= 7.0
* faiss
### Download data
To download the raw corpus of MSMARCO & Natural Questions, as well as the preprocessed training data, run
```
sh wget_data.sh
```
The downloaded data will be saved into <u>`corpus`</u> (including the training and development/test sets of MSMARCO & NQ and all the passages in MSMARCO and Wikipedia to be indexed), <u>`data_train`</u> (including the preprocessed training data of RocketQAv2).
```
├── corpus/
│   ├── marco                   # The original dataset of MSMARCO 
│   │   ├── train.query.txt
│   │   ├── train.query.txt.format
│   │   ├── qrels.train.tsv
│   │   ├── qrels.train.addition.tsv
│   │   ├── dev.query.txt
│   │   ├── dev.query.txt.format
│   │   ├── qrels.dev.tsv
│   │   ├── para.txt
│   │   ├── para.title.txt
│   │   ├── para_8part          # The paragraphs were divided into 8 parts to facilitate the inference
│   │   ├── 
│   ├── nq                      # The original dataset of NQ 
│   │   ├── ...                 # (has the same directory structure as MSMARCO) 
```

```
├── data_train/
│   ├── marco_joint.rand128+aug128                  # Hybrid Training examples of MSMARCO with 128 instance list
│   ├── nq_joint.rand32+aug32                       # Hybrid Training examples of Natural Questions with 32 instance list
│   ├── marco_joint.rand8                           # Undenoised Training examples of MSMARCO with 8 instance list
│   ├── nq_joint.rand8                              # Undenoised Training examples of Natural Questions with 8 instance list
```


### Download the trained models
To download our trained models and the initial pre-trained language model (ERNIE 2.0), run
```
sh wget_trained_model.sh
```
The downloaded model parameters will be saved into <u>`checkpoint`</u>, including
```
├── checkpoint/   
│   ├── marco_joint-encoder_warmup                  # Initial parameters for joint-encoder on MSMARCO
│   ├── nq_joint-encoder_warmup                     # Initial parameters for joint-encoder on NQ
│   ├── marco_joint-encoder_trained                 # Final joint-encoder model on MSMARCO
│   ├── nq_joint-encoder_trained                    # Final joint-encoder model on NQ
```


## Training


### The Training Procedure
To reproduce the results of the paper, you can follow the commands in **```run_marco.sh```** / **```run_nq.sh```**. These scripts contain the entire process of RocketQAv2.


#### Joint Model Training
To train a joint model, run
```
cd model
sh script/run_joint-model_train.sh $TRAIN_SET $MODEL_PATH $nodes $epochs $instance_num
```

#### Dual-encoder inference
To do the inference of dual-encoder and get top K retrieval results (retrieved by FAISS), run
```
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K
```
Here, we separate whole candidate passages into 8 parts, and predict their embeddings with 8 GPU cards simultaneously. After getting top K results on each part, we merge them to get the final file. (ie. <u>`$recall_topk_file`</u> in Data Processing)


Tips: remember to specify GPU cards before training by
```
export CUDA_VISIBLE_DEVICES=0,1,xxx
```
#### Cross-encoder inference
To do the inference of cross-encoder and get scores from cross-encoder, run
```
sh script/run_reranking.sh $TEST_SET $MODEL_PATH
```
<u>`$TEST_SET`</u> consists of textual queries and its top-k passages recalled by retriever, each line is in the form of query\tpassage.

## Evaluation
To evaluate the retriever on MSMARCO development set, run
```
python metric/msmarco_eval.py corpus/marco/qrels.dev.tsv $recall_topk_file
```
To evaluate the retriever on NQ test set, run
```
python metric/nq_eval.py $recall_topk_file
```
To evaluate the retriever on MSMARCO development set, run
```
python metric/generate_candrank.py $cand_score $recall_topk_test
python metric/msmarco_eval.py corpus/marco/qrels.dev.tsv metric/ranking_res
```
To evaluate the retriever on NQ test set, run
```
python metric/nq_eval_rerank.py $cand_score $recall_topk_test
```

The table below shows the results of our experiments on two datasets.  
<table>
<tr>
<th rowspan="2">Model</th><th colspan="3">MSMARCO Dev</th><th colspan="3">NQ Test</th>
</tr>
<tr>
<th>MRR@10</th><th>R@50</th><th>R@1000</th><th>R@5</th><th>R@20</th><th>R@100</th>
</tr>
<tr>
<td>RocketQAv2 (retriever)</td><td>38.8</td><td>86.2</td><td>98.1</td><td>75.1</td><td>83.7</td><td>89.0</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>MSMARCO Dev MRR@10</th>
</tr>

<tr>
<td>RocketQAv2 (re-ranker)</td><td>41.9</td>
</tr>
</table>


## Citation
If you find our paper and code useful, please cite the following paper:
```
@article{ren2021rocketqav2,
  title={RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking},
  author={Ren, Ruiyang and Qu, Yingqi and Liu, Jing and Zhao, Wayne Xin and She, Qiaoqiao and Wu, Hua and Wang, Haifeng and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2110.07367},
  year={2021}
}
```
