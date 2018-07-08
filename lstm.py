import cv2
import numpy as np
import torch
from matplotlib import animation
from torch.autograd import Variable
import torch.nn.functional as F
import h5py
import os
import matplotlib.pyplot as mp


class Lstm(torch.nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.linear_f = torch.nn.Linear(61, 60)
        self.linear_i = torch.nn.Linear(61, 60)
        self.linear_c = torch.nn.Linear(61, 60)
        self.linear_t = torch.nn.Linear(61, 60)
        self.linear_o = torch.nn.Linear(60, 1)

    def forward(self, x, h_state, c_state):
        hx = torch.cat((h_state,x), dim=1)
        f = F.sigmoid(self.linear_f(hx))
        i = F.sigmoid(self.linear_i(hx))
        c = F.tanh(self.linear_c(hx))
        t = F.sigmoid(self.linear_t(hx))
        c_state = f*c_state + i*c
        h_state = t*F.tanh(c_state)
        out = self.linear_o(h_state)

        return out, h_state, c_state



lstm = Lstm()
Loss_fn = torch.nn.MSELoss()
Optimizer = torch.optim.Adam(params=lstm.parameters(), lr=0.001)


for iter in range(1000):
    start, end = iter * np.pi, (iter + 4) * np.pi  # time steps
    # sin 预测 cos
    steps = np.linspace(start, end, 600, dtype=np.float32)
    x_np = np.sin(steps)  # float32 for converting torch FloatTensor
    y_np = np.cos(steps)
    Vy_np = Variable(torch.from_numpy(y_np[np.newaxis,:]))
    O = []
    hidden = Variable(torch.zeros([1, 60]))
    cell = Variable(torch.zeros([1, 60]))
    for step in range(600):
        Vx_np = Variable(torch.from_numpy(x_np[np.newaxis, step]))
        out, hidden, cell = lstm(Vx_np, hidden, cell)
        O.append(out)
        hidden = Variable(hidden.data)
        cell = Variable(cell.data)
    outs = torch.cat(O, dim=1)
    loss = Loss_fn(outs, Vy_np)
    print(iter, loss.data.numpy())
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()

    if (iter+1)%100 == 0:
        mp.plot(steps, outs.data.numpy()[0, :], steps, y_np)
        mp.show()





