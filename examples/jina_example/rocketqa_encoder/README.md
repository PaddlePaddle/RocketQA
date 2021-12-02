# RocketQADualEncoder

`RocketQA` is an optimized training approach to improving dense passage retrieval. The three major technical contributions include cross-batch negatives, denoised hard negative sampling and data augmentation. The experiment results show that RocketQA significantly outperforms previous state-of-the-art models on both MSMARCO and Natural Questions (NQ), and the performance of end-to-end QA can be improved based on RocketQA retriever.

For dense passage retrieval, [RocketQA](https://github.com/PaddlePaddle/RocketQA) applies a two-stage method including a dual-encoder and a cross-encoder.

This executor provides the dual-encoder part.

## Usages

By default, the request to `/index` endpoint calls `encode_passage()` function. The `encode_passage()` function concatenates the `.tags['title']` and `.tags['para']` as the passage inputs to calculate the passage embedding. 

The request to `/search` endpoint call `encode_question()` function. The `encode_question()` function encode the `.text` attribute as the question input to get the vector representation.

`RocketQADualEncoder` stores the embedding at the `.embedding` attribute.

We have offered following pretrained models

- `v1_marco_de`: TBD
- `v1_marco_ce`: TBD


## Cross-encoder-based Reranking

To rerank the retrieved passages, please check out the [RocketQAReranker]().

## Citation

```text
@inproceedings{qu2021rocketqa,
               title={RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering},
author={Qu, Yingqi and Ding, Yuchen and Liu, Jing and Liu, Kai and Ren, Ruiyang and Zhao, Wayne Xin and Dong, Daxiang and Wu, Hua and Wang, Haifeng},
booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
pages={5835--5847},
      year={2021}
}
```