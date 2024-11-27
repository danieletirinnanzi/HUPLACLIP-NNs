import os
import torch

def setup_DDP(rank, world_size):
    # environment variables are SET AUTOMATICALLY when running exp with 'torchrun'
    print(os.environ['MASTER_ADDR'])
    print(os.environ['MASTER_PORT'])
    os.environ['NCCL_DEBUG'] = 'INFO'   # for debugging
    
    backends = ["nccl", "gloo", "mpi"]
    print("Checking backend availability:")
    for backend in backends:
        available = torch.distributed.is_backend_available(backend)
        print(f"  {backend}: {'Available' if available else 'Not Available'}")    
    
    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    # setting cuda device
    torch.cuda.set_device(rank)   
    
    
def cleanup_DDP():
    torch.distributed.destroy_process_group()