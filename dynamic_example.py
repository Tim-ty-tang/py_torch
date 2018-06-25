import random
import torch
import torch.nn as nn

class DyanmicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DyanmicNet, self).__init__()
        self.input = nn.Linear(D_in, H)
        self.middle = nn.Linear(H,H)
        self.output = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input(x).clamp(min=0)

        for _ in range(random.randint(0,3)):
            h_relu = self.middle(h_relu).clamp(min=0)
        
        y_pred = self.output(h_relu)
        return y_pred
    
N, D_in, H, D_out = 64, 1000, 100, 10

# in/out init
x=torch.randn(N, D_in)
y=torch.randn(N, D_out)

learning_rate = 1e-4

model = DyanmicNet(D_in, H, D_out)

# loss function
criterion = nn.MSELoss(size_average=False)
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(400):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()