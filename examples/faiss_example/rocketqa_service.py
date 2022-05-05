import os
import sys
import json
import faiss
import numpy as np
from tornado import web
from tornado import ioloop
import rocketqa

class FaissTool():
    """
    Faiss index tools
    """
    def __init__(self, text_filename, index_filename):
        self.engine = faiss.read_index(index_filename)
        self.id2text = []
        for line in open(text_filename):
            self.id2text.append(line.strip())

    def search(self, q_embs, topk=5):
        res_dist, res_pid = self.engine.search(q_embs, topk)
        result_list = []
        for i in range(topk):
            result_list.append(self.id2text[res_pid[0][i]])
        return result_list


class RocketQAServer(web.RequestHandler):

    def __init__(self, application, request, **kwargs):
        web.RequestHandler.__init__(self, application, request)
        self._faiss_tool = kwargs["faiss_tool"]
        self._dual_encoder = kwargs["dual_encoder"]
        self._cross_encoder = kwargs["cross_encoder"]

    def get(self):
        """
        Get request
        """

    def post(self):
        input_request = self.request.body
        output = {}
        output['error_code'] = 0
        output['error_message'] = ''
        output['answer'] = []
        if input_request is None:
            output['error_code'] = 1
            output['error_message'] = "Input is empty"
            self.write(json.dumps(output))
            return

        try:
            input_data = json.loads(input_request)
        except:
            output['error_code'] = 2
            output['error_message'] = "Load input request error"
            self.write(json.dumps(output))
            return

        if "query" not in input_data:
            output['error_code'] = 3
            output['error_message'] = "[Query] is missing"
            self.write(json.dumps(output))
            return

        query = input_data['query']
        topk = 5
        if "topk" in input_data:
            topk = input_data['topk']

        # encode query
        q_embs = self._dual_encoder.encode_query(query=[query])
        q_embs = np.array(list(q_embs))

        # search with faiss
        search_result = self._faiss_tool.search(q_embs, topk)

        titles = []
        paras = []
        queries = []
        for t_p in search_result:
            queries.append(query)
            t, p = t_p.split('\t')
            titles.append(t)
            paras.append(p)
        ranking_score = self._cross_encoder.matching(query=queries, para=paras, title=titles)
        ranking_score = list(ranking_score)

        final_result = {}
        for i in range(len(paras)):
            final_result[query + '\t' + titles[i] + '\t' + paras[i]] = ranking_score[i]
        sort_res = sorted(final_result.items(), key=lambda a:a[1], reverse=True)

        for qtp, score in sort_res:
            one_answer = {}
            one_answer['probability'] = score
            q, t, p = qtp.split('\t')
            one_answer['title'] = t
            one_answer['para'] = p
            output['answer'].append(one_answer)

        result_str = json.dumps(output, ensure_ascii=False)
        self.write(result_str)


def create_rocketqa_app(sub_address, rocketqa_server, language, data_file, index_file):
    """
    Create RocketQA server application
    """
    if language == 'zh':
        de_model = 'zh_dureader_de_v2'
        ce_model = 'zh_dureader_ce_v2'
    else:
        de_model = 'v1_marco_de'
        ce_model = 'v1_marco_ce'

    de_conf = {
        "model": de_model,
        "use_cuda": True,
        "device_id": 0,
        "batch_size": 32
    }
    ce_conf = {
        "model": ce_model,
        "use_cuda": True,
        "device_id": 0,
        "batch_size": 32
    }
    dual_encoder = rocketqa.load_model(**de_conf)
    cross_encoder = rocketqa.load_model(**ce_conf)

    faiss_tool = FaissTool(data_file, index_file)
    print ('Load index done')

    return web.Application([(sub_address, rocketqa_server, \
                        dict(faiss_tool=faiss_tool, \
                              dual_encoder=dual_encoder, \
                              cross_encoder=cross_encoder))])


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print ("USAGE: ")
        print ("      python3 rocketqa_service.py ${language} ${data_file} ${index_file}")
        print ("--For Example:")
        print ("      python3 rocketqa_service.py zh ../data/dureader.para test.index")
        exit()

    language = sys.argv[1]
    if language != 'en' and language != 'zh':
        print ("illegal language, only [zh] and [en] is supported", file=sys.stderr)
        exit()

    data_file = sys.argv[2]
    index_file = sys.argv[3]
    sub_address = r'/rocketqa'
    port = '8888'
    app = create_rocketqa_app(sub_address, RocketQAServer, language, data_file, index_file)
    app.listen(port)
    ioloop.IOLoop.current().start()
