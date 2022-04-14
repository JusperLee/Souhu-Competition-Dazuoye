###
# Author: Kai Li
# Date: 2022-04-14 12:19:08
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 12:23:50
###
import torch

a = torch.randn(2, 2, 2)
b = torch.tensor([1, 1])
print(a)
print(a[:, :, b])