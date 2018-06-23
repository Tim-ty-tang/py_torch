import torch

dtype = torch.float
device=torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10

# in/out init
x=torch.randn(N, D_in, device=device, dtype=dtype)
y=torch.randn(N, D_out, device=device, dtype=dtype)

# weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6

for t in range(500):
    