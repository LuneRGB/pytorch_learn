import torch
print(dir(torch.cuda.is_available))
## 双下划线是一种规范，说明变量不能篡改

print(help(torch.cuda.is_available))
## 这个命令会打开vim