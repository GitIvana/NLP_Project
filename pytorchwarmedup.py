# -*- coding: utf-8 -*-
from transformers import AutoModel
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sys, os, re, csv, codecs, numpy as np
import json
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import preprocessing
import time
TRAIN_DATA_LABEL='train.label.txt'
TRAIN_DATA_FILE='train_tweets.json'
DEV_DATA_LABEL='dev.label.txt'
DEV_DATA_FILE='dev_tweets.json'
TEST_DATA_LABEL='test.label.txt'
TEST_DATA_FILE='test_tweets.json'

# read local file
def read_file(url):
    f = open(url, "r", encoding='utf-8')
    file = f.readlines()
    f.close()
    return file
#Turn the original data into the data and label arrays required for training
def read_data(x_url, y_url, isTest = False):
    if isTest:
        x_final = []
        with open(x_url, "r") as json_file:
            json_dict = json.load(json_file)
            for i in range(len(json_dict)):
                str = []
                for j in range(len(json_dict[i])):
                    str.append(json_dict[i][j]["text"])
                x_final.append(str)
        json_file.close()
        return x_final
    x_final = []
    y_final = []
    file = read_file(y_url)
    tmp_y = []
    for line in file:
        line = line.strip("\n").split(",")
        tmp_y.append(line)
    print(len(tmp_y))
    with open(x_url, "r") as json_file:
        json_dict = json.load(json_file)
        for i in range(len(json_dict)):
            str = []
            for j in range(len(json_dict[i])):
                str.append(json_dict[i][j]["text"])
            if len(json_dict[i]) > 0 and json_dict[i][0]["text"] != "":
                x_final.append(str)
                y_final.append(tmp_y[i])
        json_file.close()
    label_encoder = preprocessing.LabelEncoder()
    y_final = label_encoder.fit_transform(y_final)
    return x_final, y_final

#the class of read dataset
class SSTDataset(Dataset):

    def __init__(self, dataUrl, labelUrl, maxlen):
        # read data
        self.sentences, self.labels = read_data(dataUrl, labelUrl)
        # make tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
        # longest input length
        self.maxlen = maxlen
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Preprocessing the text to be suitable for BERT
        sentences = self.sentences[index]
        label = self.labels[index]
        cnt = 0
        source_tweet = ""
        comment_tweets =  ""
        for sentence in sentences:
            if cnt == 0:
                cnt = 1
                source_tweet = sentence
            else:
                comment_tweets += (sentence + " ")
        tokens = self.tokenizer(source_tweet, 
                                comment_tweets, 
                                padding='max_length', 
                                truncation=True,
                                max_length=self.maxlen) #Tokenize the sentence
        tokens_ids_tensor = torch.tensor(tokens['input_ids']) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = torch.tensor(tokens['attention_mask'])

        segments_ids = torch.tensor(tokens['token_type_ids'])

        return tokens_ids_tensor, attn_mask, segments_ids ,label

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import AutoModel
import torch.optim as optim
# torch.backends.cudnn.enable = False
# torch.backends.cudnn.benchmark = False
class SentimentClassifier(nn.Module):

    def __init__(self):
        super(SentimentClassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
        hidden_size = self.bert_layer.config.hidden_size
        print(hidden_size)
         #Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)
    def forward(self, seq, attn_masks, segments_ids):

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask = attn_masks,  token_type_ids = segments_ids)
        hidden_size = self.bert_layer.config.hidden_size
        cont_reps = outputs.last_hidden_state
        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]
        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)
        return logits


import time
import csv

def train(net, criterion, opti, train_loader, dev_loader, test_set, max_eps, device):

    best_acc = 0
    st = time.time()
    y_pred = []
    for ep in range(max_eps):
        
        net.train()
        for it, (seq, attn_masks, segments_ids, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, segments_ids, labels = seq.to(device), attn_masks.to(device),segments_ids.to(device), labels.to(device)
            
            #Obtaining the logits from the model
            logits = net(seq, attn_masks, segments_ids)
            #logits = net(seq, attn_masks)
            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Backpropagating the gradients
            loss.backward()
            #print(it)

            #Optimization step
            opti.step()
              
            if it % 100 == 0:
                dev_acc, dev_loss = evaluate(net, criterion, dev_loader, device)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(it, ep, dev_loss, dev_acc, (time.time()-st)))
                
        dev_acc, dev_loss = evaluate(net, criterion, dev_loader, device)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}".format(ep, dev_acc, dev_loss))
        if dev_acc >= best_acc:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            y_pred = predict_rumour(net, test_set, device)
            torch.save(net.state_dict(), 'sstcls_{}.dat'.format(ep))
        st = time.time()
        
    #get test label
    headers = ['Id','Predicted']
    rows = []
    cnt = 0;
    for line in y_pred:
        rows.append([cnt, line]);
        cnt += 1

    with open('test.csv','w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
   

        
def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def evaluate(net, criterion, dataloader, device):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    with torch.no_grad():
        for seq, attn_masks, segments_ids, labels in dataloader:
            seq, attn_masks, segments_ids, labels = seq.to(device), attn_masks.to(device),segments_ids.to(device), labels.to(device)
            
            #Obtaining the logits from the model
            logits = net(seq, attn_masks, segments_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count

######################################
#read test dataset
class SSTDatasetTEST(Dataset):

    def __init__(self, dataUrl, maxlen):
        # read data
        self.sentences = read_data(dataUrl, "", isTest = True)
        # make tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
        # longset input length
        self.maxlen = maxlen
        

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        #Selecting the sentence and label at the specified index in the data frame
        sentences = self.sentences[index]
        #Preprocessing the text to be suitable for BERT
        cnt = 0
        source_tweet = ""
        comment_tweets =  ""
        for sentence in sentences:
            if cnt == 0:
                cnt = 1
                source_tweet = sentence
            else:
                comment_tweets += (sentence + " ")
        tokens = self.tokenizer(source_tweet, 
                                comment_tweets, 
                                padding='max_length', 
                                truncation=True,
                                max_length=self.maxlen) #Tokenize the sentence
        tokens_ids_tensor = torch.tensor(tokens['input_ids']) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = torch.tensor(tokens['attention_mask'])

        segments_ids = torch.tensor(tokens['token_type_ids'])

        return tokens_ids_tensor, attn_mask, segments_ids

def predict_rumour(net, test_set, device):

    net = net.eval()
    test_loader = DataLoader(test_set)
  
    predictions_lst = []
    with torch.no_grad():
        for seq, attn_masks, segments_ids in test_loader:
            seq, attn_masks, segments_ids = seq.to(device), attn_masks.to(device),segments_ids.to(device)
            logits = net(seq, attn_masks, segments_ids)
            probs = torch.sigmoid(logits)
            soft_prob = (probs > 0.5).long()
            if soft_prob.squeeze().item() == 0:
                predictions_lst.append(0)
            else:
                predictions_lst.append(1)
    return predictions_lst

if __name__ == '__main__':
    max_len = 10
    batch_size = 8
    num_epoch = 5
    #read local dataset
    test_set = SSTDatasetTEST(dataUrl = TEST_DATA_FILE,  maxlen = max_len)
    train_set = SSTDataset(dataUrl = TRAIN_DATA_FILE, labelUrl = TRAIN_DATA_LABEL, maxlen = max_len)
    dev_set = SSTDataset(dataUrl = DEV_DATA_FILE, labelUrl = DEV_DATA_LABEL,  maxlen = max_len)
    #Creating intsances of training and development dataloaders
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers = 2)
    dev_loader = DataLoader(dev_set, batch_size = batch_size, num_workers = 2)
    print("Done preprocessing training and development data.")

    print("Creating the sentiment classifier, initialised with pretrained BERT-BASE parameters...")
    # init network
    net = SentimentClassifier()
    # load decive
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    print("Done creating the sentiment classifier.")

    criterion = nn.BCEWithLogitsLoss()
    # Set training parameters
    opti = optim.Adam(net.parameters(), lr = 1e-5,betas=(0.9, 0.999), weight_decay=0.01)
    #warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    train(net, criterion, opti, train_loader, dev_loader, test_set, num_epoch, device)




