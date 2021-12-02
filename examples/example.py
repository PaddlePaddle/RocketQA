import os
import sys
import rocketqa

query_list = []
para_list = []
title_list = []
marco_q_file = 'marco.q'
for line in open(marco_q_file):
    query_list.append(line.strip())

marco_tp_file = 'marco.tp.1k'
for line in open(marco_tp_file):
    t, p = line.strip().split('\t')
    para_list.append(p)
    title_list.append(t)

dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, device_id=0, batch_size=32)

q_embs = dual_encoder.encode_question(query=query_list)
for q in q_embs:
    print (' '.join(str(ii) for ii in q))
p_embs = dual_encoder.encode_passage(para=para_list, title=title_list)
for p in p_embs:
    print (' '.join(str(ii) for ii in p))
ips = dual_encoder.matching(query=query_list, \
                            para=para_list[:len(query_list)], \
                            title=title_list[:len(query_list)])
for ip in ips:
    print (ip)

cross_encoder = rocketqa.load_model(model="v1_marco_ce", use_cuda=True, device_id=0, batch_size=32)
ranking_score = cross_encoder.matching(query=query_list, \
                                       para=para_list[:len(query_list)], \
                                       title=title_list[:len(query_list)])
for rs in ranking_score:
    print (rs)

