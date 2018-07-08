import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import io
import unicodedata
import string
import random
import numpy as np

istrain = False

datapath = '../dataset/Name/names/'
filenames = glob.glob(datapath+'*.txt')

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
all_letters_map = {}
for letter in all_letters:
    all_letters_map[letter] = len(all_letters_map)
# print(n_letters)
# print(string.ascii_letters)
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicodeToAscii('Ślusàrski'))

def readlines(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []
for filename in filenames:
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    category_lines[category] = readlines(filename)

# print(category_lines[all_categories[0]][0])

def names2tensor(category, name):
    Tcategory = torch.zeros(1, len(all_categories))
    Tcategory[0][category] = 1
    Tname = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        Tname[i][0][all_letters_map[letter]] = 1
        # print(letter, Tname[i][0])
    return autograd.Variable(Tcategory), autograd.Variable(Tname)

def names2target(name):
    Ttarget = torch.LongTensor(np.zeros(len(name)))
    for i, letter in enumerate(name):
        if i == 0:
            continue
        Ttarget[i-1] = all_letters_map[letter]
    Ttarget[-1] = n_letters-1
    return autograd.Variable(Ttarget)

def randomSample():
    index_category = random.randint(0, len(all_categories)-1)
    index_name = random.randint(0, len(category_lines[all_categories[index_category]])-1)
    name = category_lines[all_categories[index_category]][index_name]
    Vcategory, Vname = names2tensor(index_category, name)
    Vtarget = names2target(name)
    # print(all_categories[index_category], name)
    return Vname, Vcategory, Vtarget

print(randomSample()[2][-1])



class Generater(torch.nn.Module):
    def __init__(self, categorysize, inputsize, hiddensize, outsize):
        super(Generater, self).__init__()
        self.categorysize = categorysize
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.outsize = outsize

        self.i2o = torch.nn.Linear(categorysize+inputsize+hiddensize, outsize)
        self.i2h = torch.nn.Linear(categorysize+inputsize+hiddensize, hiddensize)
        self.o2o = torch.nn.Linear(outsize+hiddensize, outsize)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, category, x, h_state):
        # print(category.size(), x.size(), h_state.size())
        x_combined = torch.cat((category, x, h_state), dim=1)
        out = self.i2o(x_combined)
        h_state = self.i2h(x_combined)
        out_combined = torch.cat((out, h_state), dim=1)
        out = F.log_softmax(self.dropout(self.o2o(out_combined)), dim=1)

        return out, h_state

    def initHidden(self):
        return autograd.Variable(torch.zeros(1, self.hiddensize))

if istrain:
    generator = Generater(len(all_categories), n_letters, 128, n_letters)
    generator.train ()
    loss_fn = torch.nn.NLLLoss()
    Optim = torch.optim.Adam(params=generator.parameters(), lr=0.0005)

    for iter in range(100000):
        Vname, Vcategory, Vtarget = randomSample()
        hidden = generator.initHidden()
        loss = 0
        generator.zero_grad()
        for step in range(Vname.size(0)):
            predict, hidden = generator(Vcategory, Vname[step], hidden)
            loss += loss_fn(predict, Vtarget[step])
        print(iter, loss.data.numpy())
        loss.backward()
        Optim.step()
        # for p in generator.parameters():
        #     p.data.add_(-0.0005, p.grad.data)

        if (iter+1)%10000 == 0:
            torch.save(generator.state_dict(), './model/generator.pkl')
            print('Save Model')

else:
    generator = Generater(len(all_categories), n_letters, 128, n_letters)
    generator.load_state_dict(torch.load('./model/generator.pkl'))
    generator.eval()

    category = 11
    startletter = 'W'

    Vcategory = torch.zeros(1, len(all_categories))
    Vcategory[0][category] = 1
    Vcategory = autograd.Variable(Vcategory)
    Vname = torch.zeros(1, n_letters)
    Vname[0][all_letters_map[startletter]] = 1
    Vname = autograd.Variable(Vname)
    hidden = generator.initHidden()
    namestring = startletter
    for step in range(20):
        out, hidden = generator(Vcategory, Vname, hidden)
        topv, topi = out.data.topk(1)
        print(topi[0])
        if topi[0][0] == n_letters-1:
            break
        else:
            namestring += all_letters[topi[0][0]]
        Vname = torch.zeros(1, n_letters)
        Vname[0][topi[0][0]] = 1
        Vname = autograd.Variable(Vname)

    print(namestring)
