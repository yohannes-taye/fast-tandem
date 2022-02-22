#use robust_cvd module
import torch 
print(torch.__version__)
# version 1.10.1+cu102
path = "/home/tmc/tandem/tandem/exported/tandem/model.pt"

x = torch.jit.load(path)
print(x.eval())