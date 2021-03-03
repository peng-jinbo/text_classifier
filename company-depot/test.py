import heapq
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
def load_score(file):
    scores = []
    score = []
    i = 0
    with open(file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.split()
            if tokens[0] == str(i):
                score.append(tokens[2])
            else:
                i += 1
                scores.append(score)
                while tokens[0] != str(i):
                    scores.append([])
                    i += 1
                score = []
                score.append(tokens[2])
        scores.append(score)
    return scores
    
    
def load_features(file):
    labels = []
    label = []
    i = 0
    with open(file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.split()
            if tokens[1] == "qid:" + str(i):
                if tokens[0] == "5":
                    label.append(1)
                else:
                    label.append(0)
            else:
                i += 1
                labels.append(label)
                while tokens[1] != "qid:" + str(i):
                    labels.append([])
                    i += 1
                label = []
                if tokens[0] == "5":
                    label.append(1)
                else:
                    label.append(0)
        labels.append(label)
    return labels

scores = load_score("/home/pengjb6/project/iiot/benchmark/company_v1/result/pro_testScoreFile.txt")
labels = load_features("/home/pengjb6/project/iiot/benchmark/company_v1/result/pro_test.txt")

def get_diff(h2m, m2l, real_result,predict_result):
    real_h= []
    real_m= []
    real_l= []
    predict_h = []
    predict_m = []
    predict_l = []
    for i in range(len(real_result)):
        if nor2len[real_result[i]] < m2l:
            real_l.append(label[i])
            predict_l.append(predict_result[i])
        elif nor2len[real_result[i]] <= h2m:
            real_m.append(label[i])
            predict_m.append(predict_result[i])
        else:
            real_h.append(label[i])
            predict_h.append(predict_result[i])
    return real_h,real_m,real_l,predict_h,predict_m,predict_l

def load_txt(file):
    test_data = []
    labels = []
    with open(file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            words = line.strip().split('\t\t')
            if len(words) == 3:
                test_data.append([words[0],words[1]])
                labels.append(int(words[2]))
    return test_data,labels


def main():
    file = "./test_file"
    score_file = ".result/pro_testScoreFile.txt"
    label_file = "./result/pro_test.txt"
    val_data, val_label= load_txt(file)
    ori = []
    nor = []
    for data in val_data:
        ori.append(data[0])
        nor.append(data[1])
    scores = load_score(score_file)
    labels = load_features(label_file)
    label_h,label_m,label_l,predict_h,predict_m,predict_l = get_diff(100,10,nor,labels)
    """
    test
    """

if __name__ == '__main__':
    main()