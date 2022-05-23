from jina import Executor, requests
from docarray import Document,DocumentArray
import numpy as np
import rocketqa


class RocketQADualEncoder(Executor):
    """
    Calculate the `embedding` of the passages and questions with RocketQA Dual-Encoder models.
    """

    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        """
        :param model: A model name return by `rocketqa.available_models()` or the path of an user-specified
        checkpoint config :param use_cuda: Set to `True` (default: `False`) to use GPU :param device_id: The GPU
        device id to load the model. Set to integers starting from 0 to `N`, where `N` is the number of GPUs minus 1.
        :param batch_size: the batch size during inference.
        """
        super().__init__(*args, **kwargs)
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size

    @requests(on='/index')
    def encode_passage(self, docs, **kwargs):
        batch_generator =docs['@r'].batch(batch_size=self.b_s)
        for batch in batch_generator:
            titles, paras = batch[:,('tags__title','tags__para')]
            para_embs = self.encoder.encode_para(para=paras, title=titles)
            for doc, emb in zip(batch, para_embs):
                doc.embedding = emb.squeeze()

    @requests(on='/search')
    def encode_question(self, docs, **kwargs):
        for doc in docs:
            query_emb = self.encoder.encode_query(query=[doc.text])
            query_emb = np.array(list(query_emb))
            doc.embedding = query_emb.squeeze()
