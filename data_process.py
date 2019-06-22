import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from itertools import chain
import pickle
import os

# 以字符串的形式读入所有数据
with open('D://大三下//自然语言处理//语料库//icwb2-data//training//cityu_training.utf8', 'rb') as frb:
    texts = frb.read().decode('utf8')
sentences = texts.split('\r\n')  # 根据换行切分
print(sentences[0])
def f(x):
    return x+x
texts = u''.join(map(f,sentences))  # 把所有的词拼接起来
#print(texts)
print('Length of texts is %d' % len(texts))
print('Example of texts: \n', texts[:300])

# 重新以标点来划分
#sentences = re.split(u'[，。！？、‘’“”（）；’《》—……：●][" "]', texts)
sentences = re.split(u'[，。！？、‘’“”；’……：●。？（）·《〈〉》／…][" "]', texts)
print('Sentences number:', len(sentences))
print ('Sentence Example:\n', sentences[0])
print(len(sentences[0]))


def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words=[]
    tags=[]
    list=sentence.split(" ")
    for i in list:
        if(len(i)==1):
            words.append(i[0])
            tags.append("s")
        elif(len(i)==2):
            words.append(i[0])
            words.append(i[1])
            tags.append("b")
            tags.append("e")
        else:
            for j in range(len(i)):
                if(j==0):
                    words.append(i[j])
                    tags.append("b")
                elif(j==len(i)-1):
                    words.append(i[j])
                    tags.append("e")
                else:
                    words.append(i[j])
                    tags.append("m")
    return words, tags

datas = list()
labels = list()
print ('Start creating words and tags data ...')
for sentence in tqdm(iter(sentences)):
    words,tags = get_Xy(sentence)
    if(len(words)>0):
        datas.append(words)
        labels.append(tags)

print ('Length of datas is %d' % len(datas) )
print ('Example of datas: ', datas[1])
print ('Example of labels:', labels[1])

df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
print(df_data)
print(len(df_data["words"]))
print(len(df_data["tags"]))
#　句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))

df_data['sentence_len'].hist(bins=100)

# 1.用 chain(*lists) 函数把多个list拼接起来
all_words = list(chain(*df_data['words'].values))
# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1) # 注意从1开始，因为我们准备把0作为填充值
tags = [ 'x', 's', 'b', 'm', 'e']
tag_ids = range(len(tags))

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)
#pku 4689
vocab_size = len(set_words)
print ('vocab_size={}'.format(vocab_size))

max_len = 80
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))

print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
print ('Example of words: ', df_data['words'].values[0])
print ('Example of X: ', X[0])
print ('Example of tags: ', df_data['tags'].values[0])
print('Example of y: ', y[0])

#保存数据
if not os.path.exists('data/'):
    os.makedirs('data/')
with open('data/cityu_train_data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2word, outp)


    pickle.dump(id2tag, outp)
    pickle.dump(word2id,outp)
print( '** Finished saving the data.')



