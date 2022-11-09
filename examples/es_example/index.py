# -*- coding: utf-8 -*-

import argparse
import os
import sys

import faiss
import numpy as np
import rocketqa
from elasticsearch import Elasticsearch, helpers


class Indexer:
    def __init__(self, es_client, index_name, model):
        self.es_client = es_client
        self.index_name = index_name
        self.dual_encoder = rocketqa.load_model(
            model=model,
            use_cuda=False, # GPU: True
            device_id=0,
            batch_size=32,
        )

    def index(self, tps):
        titles, paras = zip(*tps)
        embs = self.dual_encoder.encode_para(para=paras, title=titles)
    
        def gen_actions():
            for i, emb in enumerate(embs):
                # Normalize the NumPy array to a unit vector to use `dot_product` similarity,
                # see https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params.
                emb = emb / np.linalg.norm(emb)
                yield dict(
                    _index=self.index_name,
                    _id=i+1,
                    _source=dict(
                        title=titles[i],
                        paragraph=paras[i],
                        vector=emb,
                    ),
                )
        return helpers.bulk(self.es_client, gen_actions())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', choices=['zh', 'en'], help='The language')
    parser.add_argument('data_file', help='The data file')
    parser.add_argument('index_name', help='The index name')
    args = parser.parse_args()

    if args.lang == 'zh':
        model = 'zh_dureader_de_v2'
    elif args.lang == 'en':
        model = 'v1_marco_de'

    with open(args.data_file) as f:
      tps = [line.strip().split('\t') for line in f]

    es_client = Elasticsearch(
        "https://localhost:9200",
        http_auth=("elastic", "123456"),
        verify_certs=False,
    )

    indexer = Indexer(es_client, args.index_name, model)
    result = indexer.index(tps)
    print(result)


if __name__ == '__main__':
    main()
