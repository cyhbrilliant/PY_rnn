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

isLoad = False
isTrain = True
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

eng_maxlength = 0
fra_maxlength = 0
for line in all_pair:
    eng_maxlength = len(line[0]) if len(line[0]) > eng_maxlength else eng_maxlength
    fra_maxlength = len(line[1]) if len(line[1]) > fra_maxlength else fra_maxlength
eng_maxlength += 1
fra_maxlength += 1
print('English MaxLength:', eng_maxlength, '   ', 'Franch MaxLength:', fra_maxlength)

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
    def __init__(self, wordsize, embaddingsize, hiddensize):
        super(Encoder, self).__init__()
        self.wordsize = wordsize
        self.embaddingsize = embaddingsize
        self.hiddensize = hiddensize

        self.embadding = torch.nn.Embedding(wordsize, embaddingsize)
        self.gru = torch.nn.GRU(input_size= embaddingsize, hidden_size=hiddensize, num_layers=1)

    def forward(self, x, h_state):
        embad = self.embadding(x).view(-1, 1, self.embaddingsize)
        outs, h_state = self.gru(embad, h_state)
        return outs, h_state

    def initHidden(self, Cuda):
        return autograd.Variable(torch.zeros(1, 1, self.hiddensize)).cuda() \
            if Cuda else autograd.Variable(torch.zeros(1, 1, self.hiddensize))

#只能串行操作
class Attention_Decoder(torch.nn.Module):
    def __init__(self, wordsize, embaddingsize, hiddensize, encoder_outs_maxlength):
        super(Attention_Decoder, self).__init__()
        self.wordsize = wordsize
        self.embaddingsize = embaddingsize
        self.hiddensize = hiddensize
        self.encoderoutput_maxlength = encoder_outs_maxlength

        self.embading = torch.nn.Embedding(wordsize, embaddingsize)
        self.dropout = torch.nn.Dropout(p = 0.1)
        self.attention = torch.nn.Linear(embaddingsize+hiddensize, encoder_outs_maxlength)
        self.attention_combine = torch.nn.Linear(embaddingsize+hiddensize, hiddensize)
        self.gru = torch.nn.GRU(input_size=hiddensize, hidden_size=hiddensize, num_layers=1)
        self.lo = torch.nn.Linear(hiddensize, wordsize)

    def forward(self, x, h_state, encoder_outs):
        embadded = self.dropout(self.embading(x).view(1, 1, -1))
        attention_weight = F.softmax(self.attention(torch.cat((h_state[0], embadded[0]), dim=1)), dim=1)
        attention_app = torch.bmm(attention_weight.unsqueeze(0), encoder_outs.unsqueeze(0))
        attention_com = F.relu(self.attention_combine(torch.cat((embadded[0], attention_app[0]), dim=1))).unsqueeze(0)
        out, h_state = self.gru(attention_com, h_state)

        out = F.log_softmax(self.lo(out.view(1, -1)), dim=1)
        # print(torch.max(out))
        return out, h_state




if isTrain:
    encoder = Encoder(eng_len, 300, 300)
    decoder = Attention_Decoder(fra_len, 300, 300, eng_maxlength)
    if isLoad:
        encoder.load_state_dict(torch.load('./model/translate_attention_encoder.pkl'))
        decoder.load_state_dict(torch.load('./model/translate_attention_decoder.pkl'))
    encoder.train()
    decoder.train()
    if withCuda:
        encoder.cuda()
        decoder.cuda()
    loss_fn = torch.nn.NLLLoss()
    Optim_encoder = torch.optim.Adam(params=encoder.parameters(), lr=0.0001)
    Optim_decoder = torch.optim.Adam(params=decoder.parameters(), lr=0.0001)

    loss_avg = 0
    for iter in range(1000000):
        # print('iter', iter)
        Optim_encoder.zero_grad()
        Optim_decoder.zero_grad()

        Vencoder_input, Vdecoder_input, Vdecoder_output = randomSample(withCuda)
        encoder_outs, h_state = encoder(Vencoder_input, encoder.initHidden(withCuda))
        sos = autograd.Variable(torch.LongTensor([0])).cuda() \
            if withCuda else autograd.Variable(torch.LongTensor([0]))
        Vdecoder_nextinput = sos

        Vencoder_outs = autograd.Variable(torch.zeros(eng_maxlength, 300))
        Vencoder_outs = Vencoder_outs.cuda() if withCuda else Vencoder_outs
        Vencoder_outs[0:encoder_outs.size(0)] = encoder_outs[:, 0]

        loss = 0
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for step in range(Vdecoder_output.size()[0]):
                out, h_state = decoder(Vdecoder_input[step], h_state, Vencoder_outs)
                loss += loss_fn(out, Vdecoder_output[step])
        else:
            for step in range(Vdecoder_output.size()[0]):
                out, h_state = decoder(Vdecoder_nextinput, h_state, Vencoder_outs)
                loss += loss_fn(out, Vdecoder_output[step])
                topv, topi = out.data.topk(1)
                Vdecoder_nextinput = autograd.Variable(torch.LongTensor([topi[0][0]])).cuda() \
                    if withCuda else autograd.Variable(torch.LongTensor([topi[0][0]]))

        loss_avg += loss.cpu().data.numpy()[0]/Vdecoder_output.size(0)
        loss.backward()
        Optim_encoder.step()
        Optim_decoder.step()

        if (iter + 1) % 1000 == 0:
            print(iter+1, loss_avg/1000.0)
            loss_avg = 0
            torch.save(encoder.state_dict(), './model/translate_attention_encoder.pkl')
            torch.save(decoder.state_dict(), './model/translate_attention_decoder.pkl')
            print('Save Model')
else:
    encoder = Encoder(eng_len, 300, 300)
    decoder = Attention_Decoder(fra_len, 300, 300, eng_maxlength)
    encoder.load_state_dict(torch.load('./model/translate_attention_encoder.pkl'))
    decoder.load_state_dict(torch.load('./model/translate_attention_decoder.pkl'))
    encoder.eval()
    decoder.eval()
    if withCuda:
        encoder.cuda()
        decoder.cuda()

    sentence = 'ok .'.split()
    # sentence = 'she bought him a sweater .'.split()
    Vsentence = []
    for word in sentence:
        Vsentence.append(eng_to_ix[word])
    Vsentence.append(eng_to_ix['EOS'])
    Vsentence = autograd.Variable(torch.LongTensor(Vsentence)).cuda() \
        if withCuda else autograd.Variable(torch.LongTensor(Vsentence))

    sentence_trans = ''
    encoder_outs, h_state = encoder(Vsentence, encoder.initHidden(withCuda))
    sos = autograd.Variable(torch.LongTensor([0])).cuda() \
        if withCuda else autograd.Variable(torch.LongTensor([0]))
    Vdecoder_nextinput = sos
    Vencoder_outs = autograd.Variable(torch.zeros(eng_maxlength, 300))
    Vencoder_outs = Vencoder_outs.cuda() if withCuda else Vencoder_outs
    Vencoder_outs[0:encoder_outs.size(0), :] = encoder_outs[:, 0, :]

    for step in range(fra_maxlength):
        out, h_state = decoder(Vdecoder_nextinput, h_state, Vencoder_outs)
        topv, topi = out.data.topk(1)
        # print(topi[0][0])
        if topi[0][0] == 1:
            break
        else:
            Vdecoder_nextinput = autograd.Variable(torch.LongTensor([topi[0][0]])).cuda() \
                if withCuda else autograd.Variable(torch.LongTensor([topi[0][0]]))
            sentence_trans += fra_set[topi[0][0]] + ' '
    print(sentence_trans)



