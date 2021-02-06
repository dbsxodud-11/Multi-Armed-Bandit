import torch

y = []
x = torch.normal(0, 1, size=(8, ))
# y.append(x)
# print(torch.cat(y))
# print(torch.chunk(x, 2))
a, b = torch.chunk(x, 2)
# print(a)
# print(torch.cat([a,b], dim=1))


# x = torch.normal(0, 1, size=(1, 16))
# y = torch.normal(0, 1, size=(1, 16))
# x = torch.chunk(x, 1)
# print(x)
# for x_, y_ in zip(x, y) :
#     print(x_)

x = []
for _ in range(5) :
    x.append(torch.normal(0, 1, size=(5,)))
print(x)
print(x[::-1])