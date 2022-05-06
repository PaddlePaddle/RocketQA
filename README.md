<p align=center> <img src="https://github.com/PaddlePaddle/RocketQA/blob/main/RocketQA_title.png" /> </p>

<div align=center>
  
![](https://img.shields.io/badge/license-Apache%202-blue) ![](https://img.shields.io/badge/version-v1.0-green) ![](https://img.shields.io/badge/JupyterNotebook-Try%20%F0%9F%9A%80RocketQA%20Now!-orange) ![](https://img.shields.io/badge/requirements-up%20to%20date-brightgreen) ![](https://img.shields.io/badge/size-1.68MB-blue)
  
 </div>

In recent years, the dense retrievers based on pre-trained language models have achieved remarkable progress. To facilitate more developers using cutting edge technologies, this repository provides an easy-to-use toolkit for running and fine-tuning the state-of-the-art dense retrievers, namely **🚀RocketQA**. This toolkit has the following advantages:


* ***State-of-the-art***: 🚀RocketQA provides our well-trained models, which achieve SOTA performance on many dense retrieval datasets. And it will continue to update the [latest models](https://github.com/PaddlePaddle/RocketQA#news).
* ***First-Chinese-model***: 🚀RocketQA provides the first open source Chinese dense retrieval model, which is trained on millions of manual annotation data from [DuReader](https://github.com/baidu/DuReader).
* ***Easy-to-use***: By integrating this toolkit with [JINA](https://jina.ai/), 🚀RocketQA can help developers build an end-to-end retrieval system and question answering system with several lines of code. <img src="https://github.com/PaddlePaddle/RocketQA/blob/main/RocketQA_flow.png" alt="" align=center />  

## News
* April 29, 2022: **Training function** is added to RocketQA toolkit. And the baseline models of **DuReader<sub>retrieval</sub>** (both cross encoder and dual encoder) are available in RocketQA models.
* March 30, 2022: The baseline of **DuReader<sub>retrieval</sub>** [leaderboard](https://aistudio.baidu.com/aistudio/competition/detail/157/0/introduction) was released. [[code/model]](https://github.com/PaddlePaddle/RocketQA/tree/main/research/DuReader-Retrieval-Baseline)
* March 30, 2022: We released **DuReader<sub>retrieval</sub>**, a large-scale Chinese benchmark for passage retrieval. The dataset contains over 90K questions and 8M passages from Baidu Search. [[paper]](https://arxiv.org/abs/2203.10232) [[data]](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval)
* December 3, 2021: The toolkit of dense retriever RocketQA was released, including the first chinese dense retrieval model trained on DuReader. 
* August 26, 2021: [RocketQA v2](https://arxiv.org/pdf/2110.07367.pdf) was accepted by EMNLP 2021. [[code/model]](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQAv2_EMNLP2021)
* May 5, 2021: [PAIR](https://aclanthology.org/2021.findings-acl.191.pdf) was accepted by ACL 2021. [[code/model]](https://github.com/PaddlePaddle/RocketQA/tree/main/research/PAIR_ACL2021)
* March 11, 2021: [RocketQA v1](https://arxiv.org/pdf/2010.08191.pdf) was accepted by NAACL 2021. [[code/model]](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021)


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

Second, install rocketqa package (latest version: 1.1.0):
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

Refer to the examples below, you can build and run your own Search Engine with several lines of code. We also provide a [Playground](https://aistudio.baidu.com/aistudio/projectdetail/3225255?contributionType=1) with JupyterNotebook. Try 🚀RocketQA straight away in your browser!

### Running with JINA
[JINA](https://jina.ai/) is a cloud-native neural search framework to build SOTA and scalable deep learning search applications in minutes. Here is a simple example to build a Search Engine based on JINA and RocketQA.

```bash
cd examples/jina_example
pip3 install -r requirements.txt

# Generate vector representations and build a libray for your Documents
# JINA will automaticlly start a web service for you
python3 app.py index toy_data/test.tsv

# Try some questions related to the indexed Documents
python3 app.py query_cli
```
Please view [JINA example](https://github.com/PaddlePaddle/RocketQA/tree/main/examples/jina_example) to know more.

### Running with FAISS
We also provide a simple example built on [Faiss](https://github.com/facebookresearch/faiss).
```bash
cd examples/faiss_example/
pip3 install -r requirements.txt

# Generate vector representations and build a libray for your Documents
python3 index.py zh ../data/dureader.para test_index

# Start a web service on http://localhost:8888/rocketqa
python3 rocketqa_service.py zh ../data/dureader.para test_index

# Try some questions related to the indexed Documents
python3 query.py
```


## API
You can also easily integrate 🚀RocketQA into your own task. We provide two types of models, ERNIE-based dual encoder for answer retrieval and ERNIE-based cross encoder for answer re-ranking. For running our models, you can use the following functions.

### Load model

#### [`rocketqa.available_models()`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L17)

Returns the names of the available RocketQA models. To know more about the available models, please see the code comment.

#### [`rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L52)

Returns the model specified by the input parameter. It can initialize both dual encoder and cross encoder. By setting input parameter, you can load either RocketQA models returned by "available_models()" or your own checkpoints.

### Dual encoder
Dual-encoder returned by "load_model()" supports the following functions:

#### [`model.encode_query(query: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/1746b938d659c7f8d0b9f960e3199dcbd945adac/rocketqa/encoder/dual_encoder.py#L151)

Given a list of queries, returns their representation vectors encoded by model.

#### [`model.encode_para(para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/1746b938d659c7f8d0b9f960e3199dcbd945adac/rocketqa/encoder/dual_encoder.py#L179)

Given a list of paragraphs and their corresponding titles (optional), returns their representations vectors encoded by model.

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/1746b938d659c7f8d0b9f960e3199dcbd945adac/rocketqa/encoder/dual_encoder.py#L212)

Given a list of queries and paragraphs (and titles), returns their matching scores (dot product between two representation vectors). 

#### [`model.train(train_set: str, epoch: int, save_model_path: str, args)`](https://github.com/PaddlePaddle/RocketQA/blob/1746b938d659c7f8d0b9f960e3199dcbd945adac/rocketqa/encoder/dual_encoder.py#L247)

Given the hyperparameters `train_set`, `epoch` and `save_model_path`, you can train your own dual encoder model or finetune our models. Other settings like `save_steps` and `learning_rate` can also be set in `args`. Please refer to examples/example.py for detail.

### Cross encoder
Cross-encoder returned by "load_model()" supports the following function:

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/1746b938d659c7f8d0b9f960e3199dcbd945adac/rocketqa/encoder/cross_encoder.py#L156)

Given a list of queries and paragraphs (and titles), returns their matching scores (probability that the paragraph is the query's right answer).
  
#### [`model.train(train_set: str, epoch: int, save_model_path: str, args)`](https://github.com/PaddlePaddle/RocketQA/blob/1746b938d659c7f8d0b9f960e3199dcbd945adac/rocketqa/encoder/cross_encoder.py#L193)

Given the hyperparameters `train_set`, `epoch` and `save_model_path`, you can train your own cross encoder model or finetune our models. Other settings like `save_steps` and `learning_rate` can also be set in `args`. Please refer to examples/example.py for detail.


### Examples

Following the examples below, you can retrieve the vector representations of your documents and connect 🚀RocketQA to your own tasks.  

####  Run RocketQA Model
To run RocketQA models, you should set the parameter `model` in 'load_model()' with RocketQA model name returned by 'available_models()'.

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

####  Train Your Own Model
To train your own models, you can use `train()` function with your dataset and parameters. Training data contains 4 columns: query, title, para, label (0 or 1), separated by "\t". For detail about parameters and dataset, please refer to './examples/example.py'

```python
import rocketqa

# init cross encoder, and set device and batch_size
cross_encoder = rocketqa.load_model(model="zh_dureader_ce", use_cuda=True, device_id=0, batch_size=32)

# finetune cross encoder based on "zh_dureader_ce_v2"
cross_encoder.train('./examples/data/cross.train.tsv', 2, 'ce_models', save_steps=1000, learning_rate=1e-5, log_folder='log_ce')

```
  
#### Run Your Own Model
To run your own models, you should set parameter `model` in 'load_model()' with a JSON config file.

```python
import rocketqa

# init cross encoder
cross_encoder = rocketqa.load_model(model="./examples/ce_models/config.json", use_cuda=True, device_id=0, batch_size=16)

# compute relevance of query and para
relevance = cross_encoder.matching(query=query_list, para=para_list)
```

config is a JSON file like this
```
{
    "model_type": "cross_encoder",
    "max_seq_len": 384,
    "model_conf_path": "zh_config.json",
    "model_vocab_path": "zh_vocab.txt",
    "model_checkpoint_path": ${YOUR_MODEL},
    "for_cn": true,
    "share_parameter": 0
}
```
Folder `examples` provides more details.


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

If you find DuReader<sub>retrieval</sub> dataset helpful, feel free to cite our publication [DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine](https://arxiv.org/pdf/2203.10232.pdf)

```
@inproceedings{DuReader_retrieval,
    title="DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine",
    author="Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, Qiaoqiao She, Jing Liu, Hua Wu and Haifeng Wang",
    year="2022"
}
```

## License
This repository is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/RocketQA/blob/main/LICENSE).


## Contact Information
For help or issues using RocketQA, please submit a Github issue.


For other communication or cooperation, please contact Jing Liu (liujing46@baidu.com) or scan the following QR Code.

<img src="https://github.com/PaddlePaddle/RocketQA/blob/main/BaiduNLP-QRCode.png" width = "300" height = "300" alt="" align=center />

