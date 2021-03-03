import fasttext
from tqdm import tqdm as tqdm
import pickle
import numpy as np
from scipy.stats import entropy
from multiprocessing.pool import Pool
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def get_result(text,model,num2label):
    result = model.predict(text)
    return [num2label[int(result[0][0][9:])],result[1][0]]
    
    
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
    
def get_probdis(text):
    result = model.predict(text,k = 30000)
    labels = result[0]
    prob = result[1]/sum(result[1])
    label2prob = {}
    for i in range(len(labels)):
        label2prob[int(labels[i][9:])] = prob[i]
    count = 0
    ret = []
    for i in range(len(labels)):
        ret.append(label2prob[count])
        count += 1
    return np.array(ret)
    
def js_divergence(p, q):
    m = (p + q) / 2
    js = entropy(p, m, axis=-1) / 2 + entropy(q, m, axis=-1) / 2
    return js
    
def get_js_score(texts):
    p = get_probdis(texts[0])
    q = get_probdis(texts[1])
    return js_divergence(p,q)
    
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
    
def get_pr(threshold):
    predict_label = []
    for i in predict_l:
        if i[1] >= threshold:
            predict_label.append(0)
        else:
            predict_label.append(1)  
    return [precision_score(label_l, predict_label),recall_score(label_l, predict_label)]
    
    
def main():
    model = fasttext.load_model("/home/pengjb6/project/iiot/benchmark/fasttext/fastText-0.9.2/model/fasttext_model.bin")
    test_data = []
    file = "/home/datamerge/SIGIR/Data/Test/new_mag_test.txt"
    with open(file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            words = line.strip().split('\t\t')
            if len(words) == 3:
                test_data.append([words[0],words[1],words[2]])
                ori = []
    nor = []
    label = []
    for data in test_data:
        ori.append(data[0])
        nor.append(data[1])
        label.append(int(data[2]))
    num2label = pickle.load(open("/home/pengjb6/project/iiot/benchmark/fasttext/num2label.pkl",'rb'))
    val_data, val_label= load_txt("/home/jxqi/SIGIR2021/Data/task3/task3_overall_test.txt")
    ori = []
    nor = []
    for data in val_data:
        ori.append(data[0])
        nor.append(data[1])
    points = np.arange(0,max_len+0.001,0.001)
    label_h,label_m,label_l,predict_h,predict_m,predict_l = get_diff(100,10,nor,predict)
    with Pool() as pool:
        pr_result = list(tqdm(pool.imap(get_pr,points),total = len(points)))
    pr_result = np.array(pr_result)
    np.save("pr_result.npy",pr_result)
    
    
if __name__ == '__main__':
    main()