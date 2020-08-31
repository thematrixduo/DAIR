import torch
from torch.distributions.normal import Normal
dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
a = dist.log_prob(torch.tensor([0.0,0.3,0.4]))
print(a)
