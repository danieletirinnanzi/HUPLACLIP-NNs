import torch
import os
from torch.utils.tensorboard import SummaryWriter

# Custom imports
from src.models import CNN, MLP
from src.utils import (
    load_model,
    load_config,
    load_resume_progress,
    load_temp_checkpoint,
)
from src.train_test import train_model

def main():
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        print("World size is: ", world_size, " (should be 2)")
    device_id = local_rank
    # Set up test config and directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join("docs", "mlp_exp_config.yml"))  # CHANGE IF NEEDED
    model_name = "MLP"  # CHANGE IF NEEDED

    # Create mock_writer for Tensorboard
    runs_dir = os.path.join(current_dir, "mock_runs_resume_training", model_name, f"rank{rank}")
    os.makedirs(runs_dir, exist_ok=True)
    mock_writer = SummaryWriter(log_dir=runs_dir)

    # Create mock results folder
    results_dir = os.path.join(current_dir, "mock_results_resume_training", model_name)
    torch.distributed.barrier()

    # Model config
    # Find the model config matching the model_name
    model_config = None
    for model in config["models"]:
        if model_name in model["model_name"]:
            model_config = model
            break
    if model_config is None:
        raise ValueError("Model name not found in config")

    # Using the first graph size for testing
    graph_size = config["graph_size_values"][0]

    # Load model
    model = load_model(model_config, graph_size, config["training_parameters"]["num_train"], world_size, rank, device_id)

    # Load progress and checkpoint to simulate resume (only rank 0 prints)
    progress = load_resume_progress(results_dir, model_name, graph_size)
    temp_ckpt = load_temp_checkpoint(results_dir, model_name, graph_size, map_location={"cuda:%d" % 0: "cuda:%d" % device_id})
    if rank == 0:
        print("Loaded progress:", progress)
        print("Loaded temp_ckpt keys:", list(temp_ckpt.keys()) if temp_ckpt else None)

    # Resume training from checkpoint
    if rank == 0:
        print("\n=== Simulating resume from checkpoint ===")
    train_model(
        model,
        config["training_parameters"],
        graph_size,
        config["p_correction_type"],
        mock_writer,
        model_name,
        results_dir,
        world_size=world_size,
        rank=rank,
        device_id=device_id,
        resume=True,
        exp_name_with_time="mock_resume_exp"
    )

    if rank == 0:
        print("\nTest complete.")

    # DDP cleanup
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
    
# ---------------------------
# How to run this test file (NOTE: "trainloop_checkpoint_test.py" must be run first, as it creates a checkpoint to resume from)
# ---------------------------
# Resource allocation (debug mode, has 30' limit by default):
# salloc --nodes=1 --gres=gpu:2 --mem=64G --ntasks=2 --cpus-per-task=8 -A Sis25_piasini -p boost_usr_prod --qos=boost_qos_dbg
# From the project root, run:
# torchrun --standalone --nproc_per_node=2 -m tests.train_loop.trainloop_resume_test