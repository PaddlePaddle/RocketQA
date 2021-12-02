# RocketQAReranker

`RocketQA` is an optimized training approach to improving dense passage retrieval. The three major technical contributions include cross-batch negatives, denoised hard negative sampling and data augmentation. The experiment results show that RocketQA significantly outperforms previous state-of-the-art models on both MSMARCO and Natural Questions (NQ), and the performance of end-to-end QA can be improved based on RocketQA retriever.

For dense passage retrieval, [RocketQA](https://github.com/PaddlePaddle/RocketQA) applies a two-stage method including a dual-encoder and a cross-encoder.

This executor provides the cross-encoder part.

## Usages

`RocketQAReranker` is used only for the querying purpose. By default, the request to the `/index` endpoint is handled 
by the `rank()` function. The `rank()` function takes the `.text` attribute as the question and `.matches[i]` as 
the passage candidates. The `.matches[i]` are expected to have both `.tags['title']` and `.tags['para']` to store the title and the passage information as text.

An [ERNIE](https://github.com/PaddlePaddle/ERNIE)-based ranking model is used to rerank the passage candidates based on the question.

The reranked results are stored in the `.matches[i]` and each match has its relevance score at `.score['relevance'].value`.

We have offered following pretrained models

- `v1_marco_de`: TBD
- `v1_marco_ce`: TBD