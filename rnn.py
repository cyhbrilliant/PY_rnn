import cv2
import matplotlib.pyplot as mp
import numpy as np
import torch
from matplotlib import animation
from torch.autograd import Variable
import torch.nn.functional as F
import h5py
import os

IsTraining=True
IsLoad_state_dict=True
IsGpuEval=True
if IsTraining:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    if IsGpuEval:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class Rnnstep(torch.nn.Module):
    def __init__(self):
        super(Rnnstep, self).__init__()
        self.L1 = torch.nn.Linear(1,60)
        self.L2 = torch.nn.Linear(60,60)
        self.Lo = torch.nn.Linear(60,1)

    def forward(self, x, Hidden_state):
        x = self.L1(x)
        Hidden_state = F.tanh(x+self.L2(Hidden_state))
        return self.Lo(Hidden_state), Hidden_state



rnn = Rnnstep()
Loss_fn = torch.nn.MSELoss()
Optimizer = torch.optim.Adam(rnn.parameters(),lr=0.001)
Hidden = Variable(torch.zeros(1, 60))
for step in range(10000):
    start, end = step * np.pi, (step + 3) * np.pi  # time steps
    # sin 预测 cos
    steps = np.linspace(start, end, 600, dtype=np.float32)
    x_np = np.sin(steps)  # float32 for converting torch FloatTensor
    y_np = np.cos(steps)
    O = []
    # print(x_np, y_np)
    for iter in range(600):
        out, Hidden = rnn(Variable(torch.from_numpy(x_np[np.newaxis,iter])), Hidden)
        O.append(out)
        Hidden = Variable(Hidden.data)
        # print(iter, loss.data.numpy())

    out = torch.cat(O, dim=1)
    # print(O)
    loss = Loss_fn(out, Variable(torch.from_numpy(y_np[np.newaxis,:])))
    print(step, loss.data.numpy())
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()

    if (step + 1) % 100 == 0:
        mp.plot(steps, out.data.numpy()[0,:], steps, y_np)
        mp.show()







