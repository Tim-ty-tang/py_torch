import torch
import torch.nn as nn

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

# in/out init
x=torch.randn(N, D_in)
y=torch.randn(N, D_out)

learning_rate = 1e-4

model = TwoLayerNet(D_in, H, D_out)

# loss function
criterion = nn.MSELoss(size_average=False)
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(400):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    # backprop
    loss.backward()
    # step weight updates
    optimizer.step()

