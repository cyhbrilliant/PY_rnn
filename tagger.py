import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]

word_to_ix = {}
tag_to_ix = {}
for sentence, tagger in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tagger:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
print(word_to_ix)
print(tag_to_ix)

def prepare_sequence(X_to_ix, sentence):
    return autograd.Variable(torch.LongTensor([X_to_ix[i] for i in sentence]))

class MakeTag(torch.nn.Module):
    def __init__(self, wordlen, taglen, embadlen, hiddensize):
        super(MakeTag, self).__init__()
        self.wordlen = wordlen
        self.taglen = taglen
        self.embadlen = embadlen
        self.hiddensize = hiddensize
        self.embadding = torch.nn.Embedding(wordlen, embadlen)
        self.gru = torch.nn.GRU(input_size=embadlen, hidden_size=hiddensize, num_layers=1)
        self.lo = torch.nn.Linear(hiddensize, taglen)
        self.h_state = self.initHidden()

    def initHidden(self):
        return autograd.Variable(torch.randn([1, 1, self.hiddensize]))

    def forward(self, sentence):
        x = self.embadding(sentence).view(-1, 1, self.embadlen)
        x, self.h_state = self.gru(x, self.initHidden())
        print(x.size())
        print(x.view(len(sentence), -1).size())
        x = self.lo(x.view(len(sentence),-1))
        x = F.log_softmax(x, dim=1)
        return x


maketag = MakeTag(len(word_to_ix), len(tag_to_ix), 6, 6)
loss_fn = torch.nn.NLLLoss()
Optim = torch.optim.Adam(params=maketag.parameters(), lr=0.001)

for iter in range(600):
    for word, tag in training_data:
        print(word,tag)
        predict = maketag(prepare_sequence(word_to_ix, word))
        loss = loss_fn(predict, prepare_sequence(tag_to_ix, tag))
        print(iter, loss.data.numpy())
        maketag.zero_grad()
        loss.backward()
        Optim.step()

print(maketag(prepare_sequence(word_to_ix, 'The dog read that book'.split())))


