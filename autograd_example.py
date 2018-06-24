import torch as torch

dtype = torch.float
device= torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10

# in/out init
x=torch.randn(N, D_in, device=device, dtype=dtype)
y=torch.randn(N, D_out, device=device, dtype=dtype)

# weights
# requires_grad=True indicates we want the gradient when backprop.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    # entire forward pass of x to y_pred
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # entire loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # autograd to compute backprop
    # this calculates w1.grad and w2.grad autoamtically
    loss.backward()

    # manual update weights using grad descent
    # alternative methods include operating on weight.data and weight.grad.data
    # also can use torch.optim.SGD
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

    