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

isLoad = True
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
    def __init__(self, wordsize, embaddingsize, hiddensize, numlayers, isbidirectional):
        super(Encoder, self).__init__()
        self.wordsize = wordsize
        self.embaddingsize = embaddingsize
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.isbidirectional = isbidirectional

        self.embadding = torch.nn.Embedding(wordsize, embaddingsize)
        self.gru = torch.nn.GRU(input_size= embaddingsize, hidden_size=hiddensize,
                                num_layers=numlayers, bidirectional=isbidirectional)

    def forward(self, x, h_state):
        embad = self.embadding(x).view(-1, 1, self.embaddingsize)
        outs, h_state = self.gru(embad, h_state)
        return outs, h_state

    def initHidden(self, Cuda):
        h_state = autograd.Variable(torch.zeros(self.numlayers*2, 1, self.hiddensize)) \
            if self.isbidirectional else autograd.Variable(torch.zeros(self.numlayers, 1, self.hiddensize))
        return h_state.cuda() if Cuda else h_state

#只能串行操作
class Attention_Decoder(torch.nn.Module):
    def __init__(self, wordsize, embaddingsize, hiddensize, numlayers,
                 encoder_outs_maxlength, isbidirectional):
        super(Attention_Decoder, self).__init__()
        self.wordsize = wordsize
        self.embaddingsize = embaddingsize
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.encoderoutput_maxlength = encoder_outs_maxlength
        self.isbidirectional = isbidirectional

        self.embading = torch.nn.Embedding(wordsize, embaddingsize)
        self.dropout = torch.nn.Dropout(p = 0.1)
        if isbidirectional:
            self.bn1 = torch.nn.BatchNorm1d(32)
            self.bn2 = torch.nn.BatchNorm1d(1+numlayers*2)
            self.bn3 = torch.nn.BatchNorm1d(1)
            self.bn4 = torch.nn.BatchNorm1d(32)
            self.bn5 = torch.nn.BatchNorm1d(3)
            self.bn6 = torch.nn.BatchNorm1d(1)
            self.conv1 = torch.nn.Conv1d(1+numlayers*2, 32, 3, 1, padding=1)
            self.conv2 = torch.nn.Conv1d(32, 1+numlayers*2, 3, 1, padding=1)
            self.conv3 = torch.nn.Conv1d(1+numlayers*2, 1, 3, 1, padding=1)
            self.attention = torch.nn.Linear(embaddingsize, encoder_outs_maxlength)
            self.conv4 = torch.nn.Conv1d(3, 32, 3, 1, padding=1)
            self.conv5 = torch.nn.Conv1d(32, 3, 3, 1, padding=1)
            self.conv6 = torch.nn.Conv1d(3, 1, 3, 1, padding=1)
            # self.attention_combine = torch.nn.Linear(embaddingsize + hiddensize * 2, embaddingsize)
            self.gru = torch.nn.GRU(input_size=embaddingsize, hidden_size=hiddensize, num_layers=numlayers * 2)
        else:
            self.bn1 = torch.nn.BatchNorm1d(32)
            self.bn2 = torch.nn.BatchNorm1d(1 + numlayers)
            self.bn3 = torch.nn.BatchNorm1d(1)
            self.bn4 = torch.nn.BatchNorm1d(32)
            self.bn5 = torch.nn.BatchNorm1d(2)
            self.bn6 = torch.nn.BatchNorm1d(1)
            self.conv1 = torch.nn.Conv1d(1 + numlayers, 32, 3, 1, padding=1)
            self.conv2 = torch.nn.Conv1d(32, 1 + numlayers, 3, 1, padding=1)
            self.conv3 = torch.nn.Conv1d(1 + numlayers, 1, 1, padding=1)
            self.attention = torch.nn.Linear(embaddingsize, encoder_outs_maxlength)
            self.conv4 = torch.nn.Conv1d(2, 32, 3, 1, padding=1)
            self.conv5 = torch.nn.Conv1d(32, 2, 3, 1, padding=1)
            self.conv6 = torch.nn.Conv1d(2, 1, 3, 1, padding=1)
            # self.attention_combine = torch.nn.Linear(embaddingsize + hiddensize, embaddingsize)
            self.gru = torch.nn.GRU(input_size=embaddingsize, hidden_size=hiddensize, num_layers=numlayers)
        self.lo = torch.nn.Linear(hiddensize, wordsize)

    def forward(self, x, h_state, encoder_outs):
        embadded = self.dropout(self.embading(x).view(1, 1, -1))
        pre_concat = torch.cat((h_state.view(1, -1, self.hiddensize), embadded), dim=1)
        pre_concatc = F.relu(self.bn1(self.conv1(pre_concat)))
        pre_concat = F.relu(self.bn2(self.conv2(pre_concatc) + pre_concat))
        pre_concat = F.relu(self.bn3(self.conv3(pre_concat)))
        attention_weight = F.softmax(self.attention(pre_concat.view(1, -1)), dim=1)
        attention_app = torch.bmm(attention_weight.unsqueeze(0), encoder_outs.unsqueeze(0))
        attention_app = attention_app.view(1, -1, self.hiddensize)
        attention_com = torch.cat((attention_app, embadded), dim=1)
        attention_comc = F.relu(self.bn4(self.conv4(attention_com)))
        attention_com = F.relu(self.bn5(self.conv5(attention_comc) + attention_com))
        attention_com = F.relu(self.bn6(self.conv6(attention_com)))
        out, h_state = self.gru(attention_com, h_state)
        out = F.log_softmax(self.lo(out.view(1, -1)), dim=1)
        return out, h_state

#hyper-param
hiddensize = 512
embadsize = 512
numlayers = 2
isbidirectional = True
learningrate = 0.01
teacher_forcing_ratio = 0.5
printevery = 1000.0
if isTrain:
    encoder = Encoder(eng_len, embadsize, hiddensize, numlayers, isbidirectional)
    decoder = Attention_Decoder(fra_len, embadsize, hiddensize, numlayers, eng_maxlength, isbidirectional)
    if isLoad:
        encoder.load_state_dict(torch.load('./model/translate_attention_encoder5.pkl'))
        decoder.load_state_dict(torch.load('./model/translate_attention_decoder5.pkl'))
    encoder.train()
    decoder.train()
    if withCuda:
        encoder.cuda()
        decoder.cuda()
    loss_fn = torch.nn.NLLLoss()
    Optim_encoder = torch.optim.SGD(params=encoder.parameters(), lr=learningrate)
    Optim_decoder = torch.optim.SGD(params=decoder.parameters(), lr=learningrate)

    loss_avg = 0
    for iter in range(1000000):
        # print('iter', iter)
        Optim_encoder.zero_grad()
        Optim_decoder.zero_grad()

        Vencoder_input, Vdecoder_input, Vdecoder_output = randomSample(withCuda)
        encoder_outs, h_state = encoder(Vencoder_input, encoder.initHidden(withCuda))
        # print(encoder_outs.size(), h_state.size())
        sos = autograd.Variable(torch.LongTensor([0])).cuda() \
            if withCuda else autograd.Variable(torch.LongTensor([0]))
        Vdecoder_nextinput = sos

        Vencoder_outs = autograd.Variable(torch.zeros(eng_maxlength, encoder.hiddensize * 2)) \
            if encoder.isbidirectional else autograd.Variable(torch.zeros(eng_maxlength, encoder.hiddensize))
        Vencoder_outs = Vencoder_outs.cuda() if withCuda else Vencoder_outs
        Vencoder_outs[0:encoder_outs.size(0)] = encoder_outs[:, 0]

        loss = 0
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for step in range(Vdecoder_output.size()[0]):
                out, h_state = decoder(Vdecoder_input[step], h_state, Vencoder_outs)
                # print(out.size(), h_state.size())
                losst = loss_fn(out, Vdecoder_output[step])
                loss += losst
                # print(losst)
        else:
            for step in range(Vdecoder_output.size()[0]):
                out, h_state = decoder(Vdecoder_nextinput, h_state, Vencoder_outs)
                # print(out.size(), h_state.size())
                losst = loss_fn(out, Vdecoder_output[step])
                loss += losst
                # print(losst)
                topv, topi = out.data.topk(1)
                if topi[0][0] == 1:
                    break
                Vdecoder_nextinput = autograd.Variable(torch.LongTensor([topi[0][0]])).cuda() \
                    if withCuda else autograd.Variable(torch.LongTensor([topi[0][0]]))

        loss_avg += loss.cpu().data.numpy()[0]/Vdecoder_output.size(0)
        loss.backward()
        Optim_encoder.step()
        Optim_decoder.step()

        if (iter + 1) %printevery == 0:
            print(iter+1, loss_avg/printevery)
            loss_avg = 0
            torch.save(encoder.state_dict(), './model/translate_attention_encoder5.pkl')
            torch.save(decoder.state_dict(), './model/translate_attention_decoder5.pkl')
            print('Save Model')
# else:
#     encoder = Encoder(eng_len, hiddensize, embadsize)
#     decoder = Attention_Decoder(fra_len, hiddensize, embadsize, eng_maxlength)
#     encoder.load_state_dict(torch.load('./model/translate_attention_encoder5.pkl'))
#     decoder.load_state_dict(torch.load('./model/translate_attention_decoder5.pkl'))
#     encoder.eval()
#     decoder.eval()
#     if withCuda:
#         encoder.cuda()
#         decoder.cuda()
#
#     use_self_sentence = False
#     Vencoder_input = None
#     Vdecoder_output = None
#     if use_self_sentence:
#         sentence = 'we hope to see you again .'.split()
#         # sentence = 'slip on your shoes .'.split()
#         Vencoder_input = []
#         for word in sentence:
#             Vencoder_input.append(eng_to_ix[word])
#             Vencoder_input.append(eng_to_ix['EOS'])
#         Vencoder_input = autograd.Variable(torch.LongTensor(Vencoder_input)).cuda() \
#             if withCuda else autograd.Variable(torch.LongTensor(Vencoder_input))
#     else:
#         Vencoder_input, Vdecoder_input, Vdecoder_output = randomSample(withCuda)
#
#     sentence_trans = ''
#     encoder_outs, h_state = encoder(Vencoder_input, encoder.initHidden(withCuda))
#     sos = autograd.Variable(torch.LongTensor([0])).cuda() \
#         if withCuda else autograd.Variable(torch.LongTensor([0]))
#     Vdecoder_nextinput = sos
#     Vencoder_outs = autograd.Variable(torch.zeros(eng_maxlength, encoder.hiddensize))
#     Vencoder_outs = Vencoder_outs.cuda() if withCuda else Vencoder_outs
#     Vencoder_outs[0:encoder_outs.size(0), :] = encoder_outs[:, 0, :]
#
#     for step in range(fra_maxlength):
#         out, h_state = decoder(Vdecoder_nextinput, h_state, Vencoder_outs)
#         topv, topi = out.data.topk(1)
#         # print(topi[0][0])
#         if topi[0][0] == 1:
#             break
#         else:
#             Vdecoder_nextinput = autograd.Variable(torch.LongTensor([topi[0][0]])).cuda() \
#                 if withCuda else autograd.Variable(torch.LongTensor([topi[0][0]]))
#             sentence_trans += fra_set[topi[0][0]] + ' '
#     sentence_input = ''
#     for ix in Vencoder_input.cpu().data:
#         sentence_input += eng_set[ix] + ' '
#     sentence_target = ''
#     for ix in Vdecoder_output.cpu().data:
#         sentence_target += fra_set[ix] + ' '
#     print('input = ', sentence_input)
#     print('predict = ', sentence_trans)
#     print('target = ', sentence_target)
#
#
