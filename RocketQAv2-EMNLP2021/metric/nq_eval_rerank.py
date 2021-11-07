import numpy as np
from tqdm import tqdm
import random
import sys
from tokenizers import SimpleTokenizer
from tokenizers import has_answer

f_answer = open('corpus/nq/test.answers.txt', 'r')
#f_answer = open('nq.58812.ans', 'r')


print('reading text')
p_text = []
for i in range(8):
    f_p_text = open('corpus/nq/para_8part/part-0%d' % i, 'r')
    for line in f_p_text:
        line = line.strip('\n').split('\t')
        p_text.append(line[2].strip())


print('reading query-para-score')
para = {}
scores = []
score = {}

score_name = sys.argv[1]
recall_name = sys.argv[2]
f_s = open(score_name, 'r')
for line in f_s:
    scores.append(float(line.strip()))
f_s.close()

#f_qp = open('nq.58812.qtp.id', 'r')
f_qp = open(recall_name, 'r')
for i, line in enumerate(f_qp):
    line = line.strip('\n').split('\t')
    q = line[0]
    p = line[1]
    if q not in para and q not in score:
        para[q] = []
        score[q] = []
    para[q].append(p)
    score[q].append(scores[i])
f_qp.close()

print('calculating acc')
right_num_r20 = 0.0
right_num_r5 = 0.0
query_num = 0.0
MRR = 0.0
for qid, line in enumerate(f_answer):
    query_num += 1
    line = line.strip('\n').split('\t')
    answer = line[1:]
    #q = str(int(line[0])+1)
    q = str(qid+1)
    data = list(zip(score[q], para[q]))
    data.sort()
    data.reverse()
    data = data[:50]
    # random.shuffle(data)
    for i in range(20):
        if has_answer(p_text[int(data[i][1])], answer):
            right_num_r20 += 1
            break
    for i in range(5):
        if has_answer(p_text[int(data[i][1])], answer):
            right_num_r5 += 1
            break
    flag = 0
    for i in range(10):
        if has_answer(p_text[int(data[i][1])], answer):
            MRR += 1.0 / (i+1)
            break
query_num = qid + 1
r20 = right_num_r20 / query_num
r5 = right_num_r5 / query_num
MRR = MRR / query_num

print('recall@20: ' + str(r20))
print('recall@5: ' + str(r5))
print('MRR@10: ' + str(MRR))



