import torch 
import torch.nn as nn
import debugpy

debugpy.listen(5678)
print("press play")
debugpy.wait_for_client()
# With square kernels and equal stride
m = nn.Conv3d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
m = nn.Conv3d(16, 1, kernel_size=1, stride=1, bias=True),

input = torch.randn(20, 16, 3)
output = m(input)


print(output)