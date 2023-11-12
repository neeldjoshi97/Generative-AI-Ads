# First, import PyTorch
import torch

# Check PyTorch version
print(torch.__version__)

# Instant validation
# True - Success!!!
# False - Something is wrong!!!
isAve = torch.cuda.is_available()
print(isAve)

# Get information about GPU
device_cnt = torch.cuda.device_count()
cur_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(cur_device)

print(f"Number of CUDA-enabled GPUs: {device_cnt}\nCurrent Device ID: {cur_device}\nCurrent Device Name: {device_name}")
