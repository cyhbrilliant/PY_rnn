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
        self.Rnn = torch.nn.RNN(input_size=1, hidden_size=60, num_layers=2)
        self.Lo = torch.nn.Linear(60,1)

    def forward(self, x, Hidden_state):
        x, Hidden_state = self.Rnn(x, Hidden_state)
        print(x.size())
        outs = []
        for i in range(x.size(0)):
            outs.append(self.Lo(x[i, :, :]))
        return torch.stack(outs, dim=0), Hidden_state


# Hidden = Variable(torch.zeros(1, 1, 60))
rnn = Rnnstep()
Loss_fn = torch.nn.MSELoss()
Optimizer = torch.optim.Adam(rnn.parameters(),lr=0.001)
for step in range(1000):
    start, end = step * np.pi, (step + 4) * np.pi  # time steps
    # sin 预测 cos
    steps = np.linspace(start, end, 600, dtype=np.float32)
    x_np = np.sin(steps)  # float32 for converting torch FloatTensor
    y_np = np.cos(steps)
    Hidden = Variable(torch.zeros(2, 1, 600))
    # print(x_np, y_np)
    out, Hidden = rnn(Variable(torch.from_numpy(x_np[:, np.newaxis, np.newaxis])), Hidden)
    print(out.data.numpy().shape)

    # Hidden = Variable(Hidden.data)
    loss = Loss_fn(out, Variable(torch.from_numpy(y_np[:, np.newaxis, np.newaxis])))
    print(step, loss.data.numpy())
    rnn.zero_grad()
    loss.backward()
    Optimizer.step()
    if (step+1)%100 == 0:
        mp.plot(steps, out.data.numpy()[:, 0, 0], steps, y_np)
        mp.show()







