import  os
import pickle
from keras.utils import np_utils
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout
from keras.models import Sequential
from keras_contrib.layers import CRF

with open('data/pku_train_data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

#pku_ 317333 msr 532310 cityu 275967
X_train=X[:275967]
y_train=y[:275967]

y_train = np_utils.to_categorical(y_train, num_classes=5)
print(X_train.shape)
print(y_train.shape)

embedding_size = 100#词向量维度
maxlen = 80#句子最大的长度
vocab_size =4917+1+300#词汇的类别总数#pku 4686 msr 5167 cityu 4917 300未出现的字
Hidden_unit_number=128#隐藏层单元数量
label_count=5#标签数量
dropout_rate=0.2#dropout ratem33
batch_size = 128#batch size
#
#
#
model = Sequential()
model.add(Embedding(vocab_size, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(Hidden_unit_number, return_sequences=True)))
model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(Hidden_unit_number, return_sequences=True)))
model.add(Dropout(dropout_rate))
model.add(TimeDistributed(Dense(label_count)))
crf_layer = CRF(label_count)
model.add(crf_layer)
model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
history = model.fit(X_train, y_train,batch_size=batch_size,epochs=10)

if not os.path.exists('model/'):
    os.makedirs('model/')
model.save('model/pku_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
