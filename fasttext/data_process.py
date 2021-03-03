from tqdm import tqdm as tqdm
import pickle


def write_data(train_data, label2num, file):
    with open(file,'w') as f:
        for data in tqdm(train_data):
            f.write("__label__")
            f.write(str(label2num[data[1]]))
            f.write(' ')
            f.write(data[0])
            f.write('\n')

            
def load_data(file):
    data = []
    labels = set()
    with open(file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            words = line.strip().split('\t')
            if len(words) == 3:
                data.append([words[0],words[2]])
                labels.add(words[2])    
    return data, labels
    
def main():
    train_data = []
    labels = set()
    train_file = "./new_preprocess_train_mag.txt"
    test_file = "./new_preprocess_test_mag.txt"
    train_data, labels =  load_data(train_file)
                
                
    labels = list(labels)
    label2num = {}
    num2label = {}
    for i in range(len(labels)):
        label2num[labels[i]] = i
        num2label[i] = labels[i]
        
    pickle.dump(label2num,open("label2num.pkl",'wb'))
    pickle.dump(num2label,open("num2label.pkl",'wb'))

    write_data(train_data, label2num, "./fastText-0.9.2/data/train.txt")
    test_data, labels =  load_data(test_file)
    write_test_data(test_data,"./fastText-0.9.2/data/test.txt")            
    
    
    
if __name__ == '__main__':
    main()