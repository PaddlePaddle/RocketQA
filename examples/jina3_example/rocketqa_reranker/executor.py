import numpy as np

import rocketqa

from jina import Executor, requests
from docarray import Document, DocumentArray
from docarray.score import NamedScore

class RocketQAReranker(Executor):
    """
    Re-rank the `matches` of a Document based on the relevance to the question stored in the `text` field with RocketQA matching model.
    """
    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        """
        :param model: A model name return by `rocketqa.available_models()` or the path of an user-specified checkpoint config
        :param use_cuda: Set to `True` (default: `False`) to use GPU
        :param device_id: The GPU device id to load the model. Set to integers starting from 0 to `N`, where `N` is the number of GPUs minus 1.
        :param batch_size: the batch size during inference.
        """
        super().__init__(*args, **kwargs)
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size

    @requests(on='/search')
    def rank(self, docs, **kwargs):
        for doc in docs:
            question = doc.text
            doc_arr = DocumentArray([doc])
            match_batches_generator = (doc_arr['@m']
                                       .batch(batch_size=self.b_s))

            reranked_matches = DocumentArray()
            reranked_scores = []
            unsorted_matches = DocumentArray()
            for matches in match_batches_generator:
                titles, paras = matches[:,('tags__title', 'tags__para')]
                score_list = self.encoder.matching(query=[question] * len(paras), para=paras, title=titles)
                reranked_scores.extend(score_list)
                unsorted_matches += list(matches)
            sorted_args = np.argsort(reranked_scores).tolist()
            sorted_args.reverse()
            for idx in sorted_args:
                score = reranked_scores[idx]
                m = Document(
                    id=unsorted_matches[idx].id,
                    tags={
                        'title': unsorted_matches[idx].tags['title'],
                        'para': unsorted_matches[idx].tags['para']
                    }
                )
                m.scores['relevance'] = NamedScore(value=score)
                reranked_matches.append(m)
            doc.matches = reranked_matches
