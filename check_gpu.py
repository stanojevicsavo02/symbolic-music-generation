import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.rand(3, device="cuda")   # stvarno pokreće nešto na GPU
print("tensor on gpu:", x)