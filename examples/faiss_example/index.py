import os
import sys
import numpy as np
import faiss
import rocketqa


def build_index(encoder_conf, index_file_name, title_list, para_list):

    dual_encoder = rocketqa.load_model(**encoder_conf)
    para_embs = dual_encoder.encode_para(para=para_list, title=title_list)
    para_embs = np.array(list(para_embs))

    print("Building index with Faiss...")
    indexer = faiss.IndexFlatIP(768)
    indexer.add(para_embs.astype('float32'))
    faiss.write_index(indexer, index_file_name)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("USAGE: ")
        print ("      python3 index.py ${language} ${data_file} ${index_file}")
        print ("--For Example:")
        print ("      python3 index.py zh ../data/dureader.para test.index")
        exit()

    language = sys.argv[1]
    data_file = sys.argv[2]
    index_file = sys.argv[3]
    if language == 'zh':
        model = 'zh_dureader_de_v2'
    elif language == 'en':
        model = 'v1_marco_de'
    else:
        print ("illegal language, only [zh] and [en] is supported", file=sys.stderr)
        exit()

    para_list = []
    title_list = []
    for line in open(data_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    de_conf = {
            "model": model,
            "use_cuda": True,
            "device_id": 0,
            "batch_size": 32
    }
    build_index(de_conf, index_file, title_list, para_list)
