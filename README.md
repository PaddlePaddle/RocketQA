# RocketQA

In recent years, the dense retrievers based on pre-trained language models have achieved remarkable progress. To facilitate more developers using cutting edge technologies, this repository provides an easy-to-use toolkit for running and fine-tuning the state-of-the-art dense retrievers, namely **RocketQA**. This toolkit has the following advantages:


* ***State-of-the-art***: It provides well-trained RocketQA models, which achieve SOTA performance on many dense retrieval datasets. And it will continue to update the [latest models](https://github.com/PaddlePaddle/RocketQA#news).
* ***First-Chinese-model***: It provides the first open source Chinese dense retrieval model, which is trained on millions of manual annotation data from [DuReader](https://github.com/baidu/DuReader).
* ***Easy-to-use***: By integrating this toolkit with [JINA](https://jina.ai/), developers can build an end-to-end question answering system with several lines of code.

## Installation

We provide two installation methods: ***Python Installation Package*** and ***Docker Environment***


### Install with Python Package
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

NOTE: this toolkit MUST be running on Python3.6+ with [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 2.0+.

### Install with Docker

```bash
docker pull rocketqa/rocketqa

docker run -it docker.io/rocketqa/rocketqa bash
```

## Getting Started

Refer to the examples below, you can build your own Search Engine with several lines of code.

### Running with JINA
[JINA](https://jina.ai/) is a cloud-native neural search framework to build SOTA and scalable deep learning search applications in minutes. Here is a simple example to build a Search Engine based on JINA and RocketQA.

```bash
cd examples/jina_example
pip3 install -r requirements.txt

# Index: Encodes and indexes text, then starts a searching service
python3 app.py index toy_data/test.tsv

# Query: Encodes query and searches for answer, returns candidates ranked by relevance score
python3 app.py query_cli
```
Please view [JINA example](https://github.com/PaddlePaddle/RocketQA/tree/main/examples/jina_example) to know more.

### Running with FAISS
We also provide a simple example built on [Faiss](https://github.com/facebookresearch/faiss).
```bash
cd examples/faiss_example/
pip3 install -r requirements.txt

# Index: Encodes and indexes text
python3 index.py en ../marco.tp.1k marco_index

# Start service
python3 rocketqa_service.py en ../marco.tp.1k marco_index

# Request: Encodes query and searches for answer, returns candidates ranked by relevance score
python3 query.py
```


## API
RocketQA provide two types of models, ERNIE-based dual encoder for answer retrieval and ERNIE-based cross encoder for answer re-ranking. For running RocketQA models and your own checkpoints, you can use the following functions.

### Load model

#### [`rocketqa.available_models()`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L17)

Returns the names of the available RocketQA models. To know more about the available models, please see the code comment.

#### [`rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L52)

Returns the model specified by the input parameter. It can initialize both dual encoder and cross encoder. By setting input parameter, you can load either RocketQA models returned by "available_models()" or your own checkpoints.

### Dual encoder
Dual-encoder returned by "load_model()" supports the following functions:

#### [`model.encode_query(query: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L126)

Given a list of queries, returns their representation vectors encoded by model.

#### [`model.encode_para(para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L154)

Given a list of paragraphs and their corresponding titles (optional), returns their representations vectors encoded by model.

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L187)

Given a list of queries and paragraphs (and titles), returns their matching scores (dot product between two representation vectors). 

### Croess encoder
Cross-encoder returned by "load_model()" supports the following function:

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/cross_encoder.py#L129)

Given a list of queries and paragraphs (and titles), returns their matching scores (probability that the paragraph is the query's right answer).
  
  

## Examples

Following the examples below, you can run RocketQA models and your own checkpoints. 

###  Run RocketQA Model
To run RocketQA models, you should set the parameter `model` in 'load_model()' with RocketQA model name return by 'available_models()'.

```python
import rocketqa

query_list = ["trigeminal definition"]
para_list = [
    "Definition of TRIGEMINAL. : of or relating to the trigeminal nerve.ADVERTISEMENT. of or relating to the trigeminal nerve. ADVERTISEMENT."]

# init dual encoder
dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, device_id=0, batch_size=16)

# encode query & para
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list)
# compute dot product of query representation and para representation
dot_products = dual_encoder.matching(query=query_list, para=para_list)
```

### Run Self-development Model
To run your own checkpoints, you should write a config file, and set the parameter `model` in 'load_model()' with the path of the config file.

```python
import rocketqa

query_list = ["交叉验证的作用"]
title_list = ["交叉验证的介绍"]
para_list = ["交叉验证(Cross-validation)主要用于建模应用中，例如PCR 、PLS回归建模中。在给定的建模样本中，拿出大部分样本进行建模型，留小部分样本用刚建立的模型进行预报，并求这小部分样本的预报误差，记录它们的平方加和。"]

# conf
ce_conf = {
    "model": ${YOUR_CONFIG},     # path of config file
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

# init cross encoder
cross_encoder = rocketqa.load_model(**ce_conf)

# compute matching score of query and para
ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
```

${YOUR_CONFIG} is a JSON format file.
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

## News
* August 26, 2021: [RocketQA v2](https://arxiv.org/pdf/2110.07367.pdf) was accepted by EMNLP 2021.
* May 5, 2021: [PAIR](https://aclanthology.org/2021.findings-acl.191.pdf) was accepted by ACL 2021
* March 11, 2021: [RocketQA v1](https://arxiv.org/pdf/2010.08191.pdf) was accepted by NAACL 2021.
  
## Citations

If you find RocketQA v1 models helpful, feel free to cite our publication [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)
```
@inproceedings{rocketqa_v1,
    title="RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering",
    author="Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu and Haifeng Wang",
    year="2021",
    booktitle = "In Proceedings of NAACL"
}
```

If you find PAIR models helpful, feel free to cite our publication [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)
```
@inproceedings{rocketqa_pair,
    title="PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval",
    author="Ruiyang Ren, Shangwen Lv, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang and Ji-Rong Wen",
    year="2021",
    booktitle = "In Proceedings of ACL Findings"
}
```

If you find RocketQA v2 models helpful, feel free to cite our publication [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf)

```
@inproceedings{rocketqa_v2,
    title="RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking",
    author="Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang and Ji-Rong Wen",
    year="2021",
    booktitle = "In Proceedings of EMNLP"
}
```


## License
This repository is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/RocketQA/blob/main/LICENSE).


## Contact Information
For help or issues using RocketQA, please submit a Github issue.


For other communication or cooperation, please contact Jing Liu (liujing46@baidu.com) or scan the following QR Code.

<img src="https://github.com/PaddlePaddle/RocketQA/blob/main/BaiduNLP-QRCode.png" width = "300" height = "300" alt="" align=center />

