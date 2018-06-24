import torch

class myReLu(torch.autograd.Function):
    """
    implementing a custom autograd by subclassing autograd.function
    Define forward and backward graphs
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: get tensor containing input and return a tensor of output.torch
        ctx is context object that can be used to stash info for backprop
        we can cache arbitrary objects for backprop using ctx.save_for_backward method
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        in backprop we get tensor containing tensor od the loss wrt output and we need to compute the grad of the loss wrt input
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

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
    # apply function
    relu = myReLu.apply

    # forward
    y_pred = relu(x.mm(w1)).mm(w2)

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

