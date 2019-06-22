from keras.models import load_model
import pickle
import numpy as np
from keras_contrib.layers import CRF
with open('data/pku_train_data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2word = pickle.load(inp)
    id2tag = pickle.load(inp)
    word2id = pickle.load(inp)

#S B M E
label_count=5
crf_layer = CRF(label_count)
model = load_model('model/pku_model.h5', custom_objects={'CRF':
CRF,'crf_loss':crf_layer.loss_function,'crf_viterbi_accuracy':crf_layer.accuracy})

def get_id_list(sentence):
    word_list=[]
    id_list=[]
    for i in sentence:
            word_list.append(i)
    for j in range(80):
        if(j<len(word_list)):
            try:
                id_list.append(int(word2id[word_list[j]]))
            except KeyError:
                word2id[word_list[j]]=len(word2id)+1
                id2word[len(id2word)+1]=word_list[j]
                id_list.append(int(word2id[word_list[j]]))
        else:
            id_list.append(0)
    return id_list
def simple_cut(sentence):
    id_list=get_id_list(sentence)#1维
    id_list2=np.array(id_list)
    id_list2=id_list2.reshape(1,80)
    pre = model.predict(id_list2)
    pre = pre.reshape(80, 5)
    length=0
    for i in range(80):
        if(id_list[i]==0):
            break
        length+=1
    wordcut=[]
    for i in range(length):
        if(pre[i][1]==1 ):
            wordcut.append(id2word[id_list[i]])
            wordcut.append('/')
        elif (pre[i][4] == 1):
            wordcut.append(id2word[id_list[i]])
            wordcut.append('/')
        else:
            wordcut.append(id2word[id_list[i]])
    wordcut=wordcut[:len(wordcut)-1]
    return wordcut
def cut_txt():
    with open('D://大三下//自然语言处理//语料库//icwb2-data//testing//pku_test.utf8','r',encoding='utf-8')as fr:
        with open('D://大三下//自然语言处理//pku_test.utf8','a',encoding='utf-8')as fw:
            for line in fr:
                punctuation = ['，', '。', '！', '？', '、', '‘', '’', '“', '”', '；', '’', '……', '：', '●', '。', '？','（','）','·','《','〈','〉','》','／','…']
                str=''
                for i in range(len(line)):
                    if line[i] in punctuation:
                        for word in simple_cut(str):
                            if(word=='/'):
                                fw.write(' ')
                            else:
                                fw.write(word)
                        fw.write(' ')
                        fw.write(line[i])
                        fw.write(' ')
                        str=''
                    elif (i==len(line)-1):
                        str = str + line[i]
                        for word in simple_cut(str):
                            if (word == '/'):
                                fw.write(' ')
                            else:
                                fw.write(word)
                    else :
                        str=str+line[i]
sentence='学校将进行不定期检查'
print("".join(simple_cut(sentence)))

# cut_txt('')

