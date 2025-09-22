import torch, math
def rmse(a,b): return math.sqrt(torch.mean((a-b)**2).item())
