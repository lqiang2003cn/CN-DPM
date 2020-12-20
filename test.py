import torch
x = torch.tensor([[1,10,5],[2,20,4],[30,1,1]])
y = torch.argmax(x,dim=1)
print(y)

destination = torch.tensor([0,0,0,0,0.9])
stm = destination == 0
w = torch.tensor([1,2,3,4,5])
z = w[stm]
print(z)

e = torch.tensor([ [11.,1.,2.],[22.,1.,2.]  ])
e[:, 0] = 0. # make first column zero
print(e)

a = torch.tensor([[4.],[5.],[6.]])
b = torch.tensor([[4.],[5.],[6.]])
ex = torch.exp(-a + b)
print(ex)

lc = torch.tensor([[-400.],[5.]])
us = torch.tensor([[0.],[1.]])
r = torch.logsumexp(torch.stack([lc,us.log()], dim=1), dim=1)
print(r)

f5 = torch.tensor(5.)
zzz = torch.exp(f5)+1
print(zzz.log())

nl = torch.tensor([ [1.,2.],[3.,4.],[5.,6.] ])
mi = nl.min(dim=1)[0].view(-1, 1)
#to_expert = torch.exp(-nl_joint + min_joint)
extr = torch.exp(-nl+mi)
print('c',extr)
fm1 = torch.tensor(-150.)
print(('exp of -500',torch.exp(fm1)))