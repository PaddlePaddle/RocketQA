# RocketQA End-to-End QA-system Development Tool

This repository provides a simple and efficient toolkit for running RocketQA models and build a Question Answering (QA) system. 

## RocketQA
**RocketQA** is a series of dense retrieval models for Open-Domain QA. 

Open-Domain QA aims to find the answers of natural language questions from a large collection of documents. Common approaches often contain two stages, firstly a dense retriever selects a few relevant contexts, and then a neural reader extracts the answer.

RocketQA focuses on improving the dense contexts retrieval stage, and propose the following methods:
#### 1. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)

#### 2. [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)

#### 3. [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf)


## Features
* ***State-of-the-art***, RocketQA models achieve SOTA performance on MSMARCO passage ranking dataset and Natural Question dataset.
* ***First-Chinese-model***, RocketQA-zh is the first open source Chinese dense retrieval model.
* ***Easy-to-use***, both python installation package and DOCKER environment are provided.
* ***Solution-for-QA-system***, developers can build an End-to-End QA system with one line of code.
  
  

## Installation

### Install python package
First, install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html).
```bash
# GPU version:
$ pip install paddlepaddle-gpu

# CPU version:
$ pip install paddlepaddle
```

Second, install rocketqa package:
```bash
$ pip install rocketqa
```

NOTE: RocketQA package MUST be running on Python3.6+ with [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 2.0+ :

### Download Docker environment

```bash
docker pull rocketqa/rocketqa

docker run -it docker.io/rocketqa/rocketqa bash
```

  
## API
The RocketQA development tool supports two types of models, ERNIE-based dual encoder for answer retrieval and ERNIE-based cross encoder for answer re-ranking. And the development tool provides the following methods:

<<<<<<< HEAD
#### [`rocketqa.available_models()`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L17)

Returns the names of the available RocketQA models. 

#### [`rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L52)
=======
#### [`rocketqa.available_models()`](https://github.com/sfwydyc/rocketqa/blob/faef4c83a82a424fae79c12e173ae048f78e4f10/rocketqa/rocketqa.py#L17)

Returns the names of the available RocketQA models. 

#### `rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`
>>>>>>> 3a99cf2720486df8cc54acc0e9ce4cbcee993413

Returns the model specified by the input parameter. Both dual encoder and cross encoder can be initialized by this method. With input parameter, developers can load RocketQA models returned by "available_models()" or their own checkpoints.

---

Dual-encoder returned by "load_model()" supports the following methods:

<<<<<<< HEAD
#### [`model.encode_query(query: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L126)

Given a list of queries, returns their representation vectors encoded by model.

#### [`model.encode_para(para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L154)

Given a list of passages and their corresponding titles (optional), returns their representations vectors encoded by model.

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L187)
=======
#### `model.encode_query(query: List[str])`

Given a list of queries, returns their representation vectors encoded by model.

#### `model.encode_para(para: List[str], title: List[str])`

Given a list of passages and their corresponding titles (optional), returns their representations vectors encoded by model.

#### `model.matching(query: List[str], para: List[str], title: List[str])`
>>>>>>> 3a99cf2720486df8cc54acc0e9ce4cbcee993413

Given a list of queries and passages (and titles), returns their matching scores (dot product between two representation vectors). 

---

Cross-encoder returned by "load_model()" supports the following method:

<<<<<<< HEAD
#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/cross_encoder.py#L129)
=======
#### `model.matching(query: List[str], para: List[str], title: List[str])`
>>>>>>> 3a99cf2720486df8cc54acc0e9ce4cbcee993413

Given a list of queries and passages (and titles), returns their matching scores (probability that the paragraph is the query's right answer).
  
  

## Examples

With the examples below, developers can run RocketQA models or their own checkpoints. 

###  Run RocketQA Model
To run RocketQA models, developers should set the parameter `model` in 'load_model()' method with RocketQA model name return by 'available_models()' method.

```python
import rocketqa

query_list = ["trigeminal definition"]
para_list = [
    "Definition of TRIGEMINAL. : of or relating to the trigeminal nerve.ADVERTISEMENT. of or relating to the trigeminal nerve. ADVERTISEMENT."]

# init dual encoder
dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, batch_size=16)

# encode query & para
q_embs = dual_encoder.encode_question(query=query_list)
p_embs = dual_encoder.encode_passage(para=para_list)
# compute dot product of query representation and para representation
dot_products = dual_encoder.matching(query=query_list, para=para_list)
```

### Run Self-development Model
To run checkpoints, developers should write a config file, and set the parameter `model` in 'load_model()' method with the path of the config file.

```python
import rocketqa

query_list = ["交叉验证的作用"]
title_list = ["交叉验证的介绍"]
para_list = ["交叉验证(Cross-validation)主要用于建模应用中，例如PCR 、PLS回归建模中。在给定的建模样本中，拿出大部分样本进行建模型，留小部分样本用刚建立的模型进行预报，并求这小部分样本的预报误差，记录它们的平方加和。"]

# conf
ce_conf = {
    "model": "./own_model/config.json",     # path of config file
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

# init cross encoder
cross_encoder = rocketqa.load_model(**ce_conf)

# compute matching score of query and para
ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
```

The config file is a JSON format file.
```bash
{
    "model_type": "cross_encoder",
    "max_seq_len": 160,
    "model_conf_path": "en_large_config.json",  # path relative to config file
    "model_vocab_path": "en_vocab.txt",         # path relative to config file
    "model_checkpoint_path": "marco_cross_encoder_large", # path relative to config file
    "joint_training": 0
}
```
  


## Start your QA-System

With the examples below, developers can build own QA-System

### Running with JINA
```bash
cd examples/jina_example/
pip3 install -r requirements.txt

# Index
python3 app.py index

# Search
python3 app.py query
<<<<<<< HEAD

To know more, please visit [JINA example](https://github.com/PaddlePaddle/RocketQA/tree/main/examples/jina_example)
=======
>>>>>>> 3a99cf2720486df8cc54acc0e9ce4cbcee993413
```



### Running with Faiss

```bash
cd examples/faiss_example/
pip3 install -r requirements.txt

# Index
python3 index.py ${language} ${data_file} ${index_file}

# Start service
python3 rocketqa_service.py ${language} ${data_file} ${index_file}

# request
python3 query.py
```

