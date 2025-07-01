import torch

def calculations():
    # 2.1 Простые вычисления с градиентами
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x**2 + y**2 + z**2 + 2*x*y*z

    f.backward()
    print("2.1 Простые вычисления с градиентами (8 баллов):")
    print("f = ", f.item())
    print(f"df/dx = {x.grad.item()} (аналитически = {2*x.item() + 2*y.item()*z.item()})")
    print(f"df/dy = {y.grad.item()} (аналитически = {2*y.item() + 2*x.item()*z.item()})")
    print(f"df/dz = {z.grad.item()} (аналитически = {2*z.item() + 2*x.item()*y.item()})")
    print()

def mse():
    # 2.2 Градиент функции потерь
    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 4.0, 6.0])

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)

    y_pred = w * x + b
    mse = torch.mean((y_pred - y_true)**2)

    mse.backward()
    print("2.2 Градиент функции потерь (9 баллов):")
    print(f"MSE = {mse.item()}")
    print(f"dMSE/dw = {w.grad.item()} (аналитически: {(2*(w*x+b - y_true)*x).mean()})")
    print(f"dMSE/db = {b.grad.item()} (аналитически: {(2*(w*x+b - y_true)).mean()})")
    print()

def chain_rule():
    # 2.3 Цепное правило
    x = torch.tensor(0.9, requires_grad=True)
    f = torch.sin(x**2 + 1)

    autograd = torch.autograd.grad(f, x)[0]
    no_autograd = torch.cos(x**2 + 1) * 2 * x

    print("2.3 Цепное правило (8 баллов):")
    print(f"f = sin(x^2+1) = {f.item()}")
    print(f"Градиент НЕ через autograd = {no_autograd.item()}")
    print(f"Градиент ЧЕРЕЗ autograd = {autograd.item()}")

calculations()
mse()
chain_rule()