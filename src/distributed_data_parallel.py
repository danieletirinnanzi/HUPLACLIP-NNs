import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    """Sets up distributed process group for multi-GPU training, using a SINGLE NODE."""
    os.environ['MASTER_ADDR'] = 'localhost'  # or specify a machine if multi-node
    os.environ['MASTER_PORT'] = '12355'     # any open port
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"DDP setup completed. Rank = {rank}; world size = {world_size}")

def cleanup_ddp():
    """Cleans up distributed process group."""
    dist.destroy_process_group()