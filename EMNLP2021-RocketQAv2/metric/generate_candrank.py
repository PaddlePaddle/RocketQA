import csv
import sys
from collections import defaultdict

score_f = sys.argv[1]
id_f = sys.argv[2]
#id_f = 'marco_joint_qtp/qtp.test.id'
#id_f = 'dev.es_1000.id'
outputf = 'metric/ranking_res'

scores = []
q_ids = []
p_ids = []
q_dic = defaultdict(list)

with open(score_f, 'r') as f:
    for line in f:
        scores.append(float(line.strip()))

with open(id_f, 'r') as f:
    for line in f:
        v = line.strip().split('\t')
        q_ids.append(int(v[0]))
        p_ids.append(int(v[1]))

for q, p, s in zip(q_ids, p_ids, scores):
    q_dic[q].append((s, p))

output = []
for q in q_dic:
    rank = 0
    cands = q_dic[q]
    cands.sort(reverse=True)
    for cand in cands:
        rank += 1
        output.append([q, cand[1], rank])
        #print str(q) + '\t' + str(cand[1]) + '\t' + str(rank)
        if rank > 49:
            break

with open(outputf, 'w') as f:
    writer = csv.writer(f, delimiter= '\t')
    writer.writerows(output)

