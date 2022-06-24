import os
import sys
import rocketqa


def train_dual_encoder(base_model, train_set):
    dual_encoder = rocketqa.load_model(model=base_model, use_cuda=True, device_id=5, batch_size=16)
    dual_encoder.train(train_set, 2, 'de_en_models', save_steps=10, learning_rate=1e-5, log_folder='de_en_log')


def train_cross_encoder(base_model, train_set):
    cross_encoder = rocketqa.load_model(model=base_model, use_cuda=True, device_id=5, batch_size=16)
    cross_encoder.train(train_set, 2, 'ce_en_models', save_steps=10, learning_rate=1e-5, log_folder='ce_en_log')


def test_dual_encoder(model, q_file, tp_file):
    query_list = []
    para_list = []
    title_list = []
    for line in open(q_file):
        query_list.append(line.strip())

    for line in open(tp_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    dual_encoder = rocketqa.load_model(model=model, use_cuda=True, device_id=1, batch_size=32)

    q_embs = dual_encoder.encode_query(query=query_list)
    for q in q_embs:
        print (' '.join(str(ii) for ii in q))
    p_embs = dual_encoder.encode_para(para=para_list, title=title_list)
    for p in p_embs:
        print (' '.join(str(ii) for ii in p))
    ips = dual_encoder.matching(query=query_list, \
                                para=para_list[:len(query_list)], \
                                title=title_list[:len(query_list)])
    for ip in ips:
        print (ip)

def test_cross_encoder(model, q_file, tp_file):

    query_list = []
    para_list = []
    title_list = []
    for line in open(q_file):
        query_list.append(line.strip())

    for line in open(tp_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    cross_encoder = rocketqa.load_model(model=model, use_cuda=True, device_id=0, batch_size=32)
    ranking_score = cross_encoder.matching(query=query_list, \
                                       para=para_list[:len(query_list)], \
                                       title=title_list[:len(query_list)])
    for rs in ranking_score:
        print (rs)


if __name__ == "__main__":
    # finetune model
    train_dual_encoder('zh_dureader_de', './examples/data/dual.train.tsv')
    # train_cross_encoder('zh_dureader_ce', './examples/data/cross.train.tsv')

    # test rocketqa model
    #test_dual_encoder('zh_dureader_de_v2', './data/dureader.q', './data/marco.tp.1k')
    #test_cross_encoder('zh_dureader_de_v2', './data/dureader.q', './data/marco.tp.1k')

    # test your own model
    # test_dual_encoder('./de_models/config.json', './data/dureader.q', './data/marco.tp.1k')
    #test_cross_encoder('./ce_models/config.json', './data/dureader.q', './data/marco.tp.1k')
