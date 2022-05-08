import sys, os, re, csv, codecs, numpy as np, pandas as pd

#=================Keras==============
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, backend

from sklearn.preprocessing import LabelEncoder    
from sklearn.preprocessing import OneHotEncoder 
#=================nltk===============
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import json


EMBEDDING_FILE='embedding/glove.twitter.27B.100d.txt'
TRAIN_DATA_LABEL='train.label.txt'
TRAIN_DATA_FILE='train_tweets/train_tweets.json'
DEV_DATA_LABEL='dev.label.txt'
DEV_DATA_FILE='dev_tweets/dev_tweets.json'
TEST_DATA_LABEL='test.label.txt'
TEST_DATA_FILE='test_tweets/test_tweets.json'
x_train = []
y_train = []
def read_file(url):
    f = open(url, "r", encoding='utf-8')
    return f.readlines()
import re
def text_to_wordlist(input, remove_stopwords=False, stem_words=True):

    # Different regex parts for smiley faces
    #return input
    input = input.lower()
    eyes = "[8:=;]"
    nose = "['`\-]?"
    input = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', "<URL>", input)
    input = re.sub("/", " / ", input)
    input = re.sub(r"@\w+", "<USER>", input)
    input = re.sub(r"#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}", "<SMILE>", input)
    input = re.sub(r"#{eyes}#{nose}p+", "<LOLFACE>", input)
    input = re.sub(r"#{eyes}#{nose}\(+|\)+#{nose}#{eyes}", "<SADFACE>", input)
    input = re.sub(r"#{eyes}#{nose}[\/|l*]", "<NEUTRALFACE>", input)
    input = re.sub(r"<3", "<HEART>", input)
    input = re.sub(r"[, . ( )]", "", input).strip()
    if input.isdigit():
        input = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", input)
    return input
def read_data(x_url, y_url, isTest = False):
    if isTest:
        x_final = []
        with open(x_url, "r") as json_file:
            json_dict = json.load(json_file)
            for i in range(len(json_dict)):
                str = ""
                for j in range(len(json_dict[i])):
                    str += json_dict[i][j]["text"]
                    str += " "
                tmp_ans = []
                tmp = str.strip("\n").split(" ")
                for item in tmp:
                    tmp_ans.append(text_to_wordlist(item))
                x_final.append(tmp_ans)
        return x_final
    x_final = []
    y_final = []
    file = read_file(y_url)
    tmp_y = []
    for line in file:
        line = line.strip("\n").split(",")
        tmp_y.append(line)
    with open(x_url, "r") as json_file:
        json_dict = json.load(json_file)
        for i in range(len(json_dict)):
            str = ""
            for j in range(len(json_dict[i])):
                str += json_dict[i][j]["text"]
                str += " "
            if str != "":
                tmp_ans = []
                tmp = str.strip("\n").split()
                for item in tmp:
                    tmp_ans.append(text_to_wordlist(item))
                x_final.append(tmp_ans)
                y_final.append(tmp_y[i])
    print(y_final[:10])
    y_final = np.array(y_final)
    lf=LabelEncoder().fit(y_final)
    data_label =lf.transform(y_final)
    of=OneHotEncoder(sparse=False).fit(data_label.reshape(-1,1))  
    y_final=of.transform(data_label.reshape(-1,1))
    print(y_final[:10])
    return x_final, y_final
x_train, y_train = read_data(TRAIN_DATA_FILE, TRAIN_DATA_LABEL)
x_dev, y_dev = read_data(DEV_DATA_FILE, DEV_DATA_LABEL)
x_test = read_data(TEST_DATA_FILE, TEST_DATA_LABEL, True)
assert(len(x_train) == len(y_train))
assert(len(x_dev) == len(y_dev))



embedding_file = read_file(EMBEDDING_FILE)
embeddings_index = {}
for line in embedding_file:
    embeddings_index[line.split()[0]] = np.asarray(line.split()[1:], dtype='float32')


embed_size = 100 # how big is each word vector
max_features = 35000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 500 # max number of words in a comment to use
number_filters = 100 # the number of CNN filters

tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
# tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train + x_dev + x_test))
comments_sequence = tokenizer.texts_to_sequences(x_train)
dev_comments_sequence = tokenizer.texts_to_sequences(x_dev)
test_comments_sequence = tokenizer.texts_to_sequences(x_test)
    
x_train_seq = pad_sequences(comments_sequence , maxlen=maxlen)
x_dev_seq = pad_sequences(dev_comments_sequence , maxlen=maxlen)
x_test_seq = pad_sequences(test_comments_sequence, maxlen=maxlen)



# embedding_file = read_file(EMBEDDING_FILE)
# print(embedding[:10])


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
key_del_list = []
for key in embeddings_index:
    if len(embeddings_index[key]) !=100:
        print()
        key_del_list.append(key)
for value in key_del_list:
    del embeddings_index[value]
embedding_matrix = np.zeros((nb_words, embed_size))


for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

filter_sizes = [5,6,10,11,20]
num_filters = 32
            
def get_model():    
    imaxlen,np = Input(shape=( ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[4], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
    maxpool_4 = MaxPool2D(pool_size=(maxlen - filter_sizes[4] + 1, 1))(conv_4)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3,maxpool_4])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(2, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()


batch_size = 128
epochs = 35


hist = model.fit(x_train_seq, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_dev_seq, y_dev),
                  verbose=2)



y_pred = model.predict(x_test_seq, batch_size=1024)
import csv
headers = ['Id','Predicted']
rows = []
cnt = 0;
for line in y_pred:
    str = 0
    if line[0] > line[1]:
        str = 0
    else:
        str = 1
    rows.append([cnt, str]);
    cnt += 1

with open('test.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)     
print(y_pred)
