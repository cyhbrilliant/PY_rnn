import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import io
import unicodedata
import string
import re
import random
import numpy as np
import os

isTrain = False
withCuda = True
if withCuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'


filepath = '../dataset/Translate/eng-fra.txt'
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# print(unicodeToAscii('Ślusàrski'))

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readPair(filepath):
    lines = io.open(filepath, encoding='utf-8').read().strip().split('\n')
    all_pair = [[normalizeString(s).split() for s in l.split('\t')] for l in lines]
    # print(all_pair[13000])
    return all_pair
all_pair = readPair(filepath)


eng_to_ix = {'SOS':0, 'EOS':1}
fra_to_ix = {'SOS':0, 'EOS':1}
eng_set = []
fra_set = []
eng_set.append('SOS')
eng_set.append('EOS')
fra_set.append('SOS')
fra_set.append('EOS')
for line in all_pair:
    for word in line[0]:
        if word not in eng_to_ix:
            eng_to_ix[word] = len(eng_to_ix)
            eng_set.append(word)
    for word in line[1]:
        if word not in fra_to_ix:
            fra_to_ix[word] = len(fra_to_ix)
            fra_set.append(word)
eng_len = len(eng_to_ix)
fra_len = len(fra_to_ix)
print('English Count:', len(eng_to_ix), '   ', 'Franch Count:', len(fra_to_ix))

def pair2tensor(pair):
    encoder_input = []
    decoder_input = []
    decoder_output = []

    for word in pair[0]:
        encoder_input.append(eng_to_ix[word])
    encoder_input.append(eng_to_ix['EOS'])

    decoder_input.append(fra_to_ix['SOS'])
    for word in pair[1]:
        index = fra_to_ix[word]
        decoder_input.append(index)
        decoder_output.append(index)
    decoder_output.append(fra_to_ix['EOS'])

    Vencoder_input = autograd.Variable(torch.LongTensor(encoder_input))
    Vdecoder_input = autograd.Variable(torch.LongTensor(decoder_input))
    Vdecoder_output = autograd.Variable(torch.LongTensor(decoder_output))
    return Vencoder_input, Vdecoder_input, Vdecoder_output

def randomSample(Cuda):
    index = random.randint(0, len(all_pair)-1)
    Vencoder_input, Vdecoder_input, Vdecoder_output = pair2tensor(all_pair[index])
    if Cuda:
        Vencoder_input = Vencoder_input.cuda()
        Vdecoder_input = Vdecoder_input.cuda()
        Vdecoder_output = Vdecoder_output.cuda()

    return Vencoder_input, Vdecoder_input, Vdecoder_output

# Vencoder_input, Vdecoder_input, Vdecoder_output = randomSample()
# print(Vencoder_input.size(), Vdecoder_input.size(), Vdecoder_output.size())

class Encoder(torch.nn.Module):
    def __init__(self, wordsize, embaddingsize, hiddensize, numlayers):
        super(Encoder, self).__init__()
        self.wordsize = wordsize
        self.embaddingsize = embaddingsize
        self.hiddensize = hiddensize
        self.numlayers = numlayers

        self.embadding = torch.nn.Embedding(wordsize, embaddingsize)
        self.gru = torch.nn.GRU(input_size= embaddingsize, hidden_size=hiddensize, num_layers=numlayers)

    def forward(self, x, h_state):
        embad = self.embadding(x).view(-1, 1, self.embaddingsize)
        outs, h_state = self.gru(embad, h_state)
        return outs, h_state


class Decoder(torch.nn.Module):
    def __init__(self, wordsize, embaddingsize, hiddensize, numlayers):
        super(Decoder, self).__init__()
        self.wordsize = wordsize
        self.embaddingsize = embaddingsize
        self.hiddensize = hiddensize
        self.numlayers = numlayers

        self.embadding = torch.nn.Embedding(wordsize, embaddingsize)
        self.gru = torch.nn.GRU(input_size=embaddingsize, hidden_size=hiddensize, num_layers=numlayers)
        self.out = torch.nn.Linear(hiddensize, wordsize)

    def forward(self, x, h_state):
        # print(x)
        embad = self.embadding(x).view(-1, 1, self.embaddingsize)
        # print(embad.size())
        outs, h_state = self.gru(embad, h_state)
        # print(outs.size())
        outs = F.log_softmax(self.out(outs.view(-1, self.hiddensize)), dim=1)
        return outs, h_state


class Translate(torch.nn.Module):
    def __init__(self, encoder_wordsize, decoder_wordsize, embaddingsize, hiddensize, numlayers, Cuda, isTrain):
        super(Translate, self).__init__()
        self.isTrain = isTrain
        self.Cuda = Cuda
        self.numlayers = numlayers
        self.hiddensize = hiddensize
        self.encoder = Encoder(encoder_wordsize, embaddingsize, hiddensize, numlayers)
        self.decoder = Decoder(decoder_wordsize, embaddingsize, hiddensize, numlayers)


    def forward(self, encoder_input, decoder_input = None):
        h_state = self.initHidden(self.Cuda)
        outs, h_state = self.encoder(encoder_input, h_state)
        if self.isTrain:
            outs, h_state = self.decoder(decoder_input, h_state)
        else:
            outs = []
            sos = autograd.Variable(torch.LongTensor([0])).cuda() \
                if self.Cuda else autograd.Variable(torch.LongTensor([0]))
            out, h_state = self.decoder(sos, h_state)
            topv, topi = out.data.topk(1)
            nextWord = autograd.Variable(torch.LongTensor([topi[0][0]])).cuda() \
                if self.Cuda else autograd.Variable(torch.LongTensor([topi[0][0]]))
            for step in range(50):
                out, h_state = self.decoder(nextWord, h_state)

                topv, topi = out.data.topk(1)
                if topi[0][0] == 1:
                    break
                else:
                    nextWord = autograd.Variable(torch.LongTensor([topi[0][0]])).cuda() \
                        if self.Cuda else autograd.Variable(torch.LongTensor([topi[0][0]]))
                    outs.append(topi[0][0])

        return outs

    def initHidden(self, Cuda):
        if Cuda:
            return autograd.Variable(torch.zeros(self.numlayers, 1, self.hiddensize)).cuda()
        else:
            return autograd.Variable(torch.zeros(self.numlayers, 1, self.hiddensize))


if isTrain:
    translate = Translate(eng_len, fra_len, 100, 100, 2, True, True) \
        if withCuda else Translate(eng_len, fra_len, 100, 100, 2, False, True)
    translate.train()
    if withCuda:
        translate.cuda()
    loss_fn = torch.nn.NLLLoss()
    Optim = torch.optim.Adam(params=translate.parameters(), lr=0.0001)

    for iter in range(1000000):
        Vencoder_input, Vdecoder_input, Vdecoder_output = randomSample(withCuda)
        outs = translate(Vencoder_input, Vdecoder_input)
        # print(outs.size())
        loss = loss_fn(outs, Vdecoder_output)
        print(iter, loss.cpu().data.numpy() if withCuda else loss.data.numpy())
        translate.zero_grad()
        loss.backward()
        Optim.step()

        if (iter + 1) % 1000 == 0:
            torch.save(translate.state_dict(), './model/translate.pkl')
            print('Save Model')

else:
    translate = Translate(eng_len, fra_len, 100, 100, 2, True, False) \
        if withCuda else Translate(eng_len, fra_len, 100, 100, 2, False, False)
    translate.load_state_dict(torch.load('./model/translate.pkl'))
    translate.eval()
    if withCuda:
        translate.cuda()
    sentence = 'run .'.split()
    Vsentence = []
    for word in sentence:
        Vsentence.append(eng_to_ix[word])
    Vsentence.append(eng_to_ix['EOS'])
    Vsentence = autograd.Variable(torch.LongTensor(Vsentence)).cuda() \
        if withCuda else autograd.Variable(torch.LongTensor(Vsentence))
    # print(Vsentence)
    outs = translate(Vsentence)
    sentence_trans = ''
    for ix in outs:
        sentence_trans += fra_set[ix] + ' '
    print(sentence_trans)