import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))