'''
nn module is the high level abstarction for neural networks like keras, tf slim, tfLearn
'''

import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

# in/out init
x=torch.randn(N, D_in)
y=torch.randn(N, D_out)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
)

loss_fn = nn.MSELoss(size_average=False)

learning_rate = 1e-4

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Reset gradients before backprop
    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad