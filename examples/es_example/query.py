# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
import rocketqa
from elasticsearch import Elasticsearch


class Querier:
    def __init__(self, es_client, index_name, de_model, ce_model):
        self.es_client = es_client
        self.index_name = index_name
        self.dual_encoder = rocketqa.load_model(
            model=de_model,
            use_cuda=False, # GPU: True
            device_id=0,
            batch_size=32,
        )
        self.cross_encoder = rocketqa.load_model(
            model=ce_model,
            use_cuda=False, # GPU: True
            device_id=0,
            batch_size=32,
        )

    def encode(self, query):
        embs = self.dual_encoder.encode_query(query=[query])
        vector = list(embs)[0]
        # Normalize the NumPy array to a unit vector to use `dot_product` similarity,
        # see https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params.
        vector = vector / np.linalg.norm(vector)
        return vector

    def search(self, query, topk=10):
        vector = self.encode(query)
        knn = dict(
            field="vector",
            query_vector=vector,
            k=topk,
            num_candidates=100,
        )
        result = self.es_client.knn_search(index=self.index_name, knn=knn)

        candidates = [
            dict(
                title=doc['_source']['title'],
                para=doc['_source']['paragraph'],
            )
            for doc in result['hits']['hits']
        ]
        return candidates

    def sort(self, query, candidates):
        queries = [query] * len(candidates)
        titles = [c['title'] for c in candidates]
        paras = [c['para'] for c in candidates]
        ranking_score = self.cross_encoder.matching(query=queries, para=paras, title=titles)
    
        answers = [
            dict(
                title=titles[i],
                para=paras[i],
                score=score,
            )
            for i, score in enumerate(ranking_score)
        ]
        return sorted(answers, key=lambda a: a['score'], reverse=True)


def main():
    es_client = Elasticsearch(
        "https://localhost:9200",
        http_auth=("elastic", "123456"),
        verify_certs=False,
    )
    querier = Querier(es_client, "test-index", 'zh_dureader_de_v2', 'zh_dureader_ce_v2')

    while True:
        query = input('Query: ')

        candidates = querier.search(query)
        print('Candidates:')
        for c in candidates:
            print(c['title'], '\t', c['para'])

        answers = querier.sort(query, candidates)
        print('Answers:')
        for a in answers:
            print(a['title'], '\t', a['para'], '\t', a['score'])


if __name__ == '__main__':
    main()
