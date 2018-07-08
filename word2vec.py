import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
#         ("Give it to me".split(), "ENGLISH"),
#         ("No creo que sea una buena idea".split(), "SPANISH"),
#         ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]
#
# test_data = [("Yo creo que si".split(), "SPANISH"),
#              ("it is lost on me".split(), "ENGLISH")]
#
# # word_to_ix maps each word in the vocab to a unique integer, which will be its
# # index into the Bag of words vector
# word_to_ix = {}
# for sent, _ in data + test_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
# #
#
#
# torch.manual_seed(1)
# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(10000, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.LongTensor([5000,1000])
# hello_embed = embeds(autograd.Variable(lookup_tensor))
# print(hello_embed)

test_sentence = '''When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.'''.split()

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(len(word_to_ix))


class Pre(torch.nn.Module):
    def __init__(self, wordlen, embadlen, contextlen):
        super(Pre, self).__init__()
        self.embadding = torch.nn.Embedding(wordlen, embadlen)
        self.linear1 = torch.nn.Linear(contextlen*embadlen, 128)
        self.linear2 = torch.nn.Linear(128, wordlen)

    def forward(self, x):
        x = self.embadding(x).view(1,-1)
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x))
        return x

pre = Pre(len(word_to_ix), 10, 2)
loss_fn = torch.nn.NLLLoss()
Optim = torch.optim.Adam(params=pre.parameters(), lr=0.01)
Data = [([word_to_ix[test_sentence[i]], word_to_ix[test_sentence[i+1]]],word_to_ix[test_sentence[i+2]]) for i in range(len(test_sentence)-2)]

for iter in range(10):
    for input, target in Data:
        print(type(input))
        predict = pre(autograd.Variable(torch.LongTensor(input)))
        print(predict.size())
        print(autograd.Variable(torch.LongTensor([target])))

        loss = loss_fn(predict, autograd.Variable(torch.LongTensor([target])))
        print(iter,loss.data.numpy())
        pre.zero_grad()
        loss.backward()
        Optim.step()






