import torch
import os


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = 0, 0
    for id in cuda_device.split(','):
        gpu_max, gpu_used = devices_info[int(id)].split(',')
        total += int(gpu_max)
        used += int(gpu_used)

    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x