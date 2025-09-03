import torch
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# custom imports
from src.utils import load_config, load_model, save_test_results
from src.train_test import test_model

# # TO RUN FROM HOME DIRECTORY WITH DDP:
# torchrun --standalone --nproc_per_node=4 -m tests.test_loop.testloop_test_DDP

def run_test_only():    
    # reading configuration and model name:
    config = load_config(os.path.join("docs", "cnn_exp_config.yml"))  # CHANGE THIS
    model_name = "CNN_large"  # CHANGE THIS

    # DDP:
    rank = (
        torch.distributed.get_rank()
    )  # identifies processes (in this context, one process per GPU)
    device_id = rank % torch.cuda.device_count()
    print(f"Running test on device id: {device_id}.")
    world_size = torch.cuda.device_count()
    print("world size is: ", world_size)

    # reading model configuration:
    model_config = [
        model for model in config["models"] if model["model_name"] == model_name
    ][0]
    # creating the model:
    model = load_model(model_config, config["graph_size_values"][0],config["testing_parameters"]["num_test"], world_size, rank, device_id)

    # load the best model from the training process on all ranks
    # - defining file name and path:
    file_path = os.path.join(
        current_dir,
        "..",
        "..",
        "results",
        "data",
        "cnn_exp_2025-09-03_14-42-36",  # CHANGE THIS
        f"N{config['graph_size_values'][0]}",
        model_name,
        f"{model_name}_N{config['graph_size_values'][0]}_trained.pth"
    )
    # - making sure processes are synchronized on all devices
    torch.distributed.barrier()        
    # - configuring map location:
    map_location = {"cuda:%d" % 0: "cuda:%d" % device_id}
    print(map_location)

    # - loading the model:
    state_dict = torch.load(file_path, map_location=map_location)
    model.load_state_dict(state_dict)

    # - putting the model in evaluation mode before starting training:
    model.eval()

    # testing best model
    fraction_correct_results, metrics_results = test_model(
        model,
        config["testing_parameters"],
        config["graph_size_values"][0],
        config["p_correction_type"],
        model_name,
        # DDP:
        world_size,
        rank,
        device_id,
    )
    # - making sure processes are synchronized on all devices
    torch.distributed.barrier()

    # saving test results as csv file
    if rank == 0:
        save_test_results(
            fraction_correct_results,
            metrics_results,
            model_name,
            config["graph_size_values"][0],
            current_dir,
        )
        
            
if __name__ == "__main__":   
    # DDP (here, using one process per GPU):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")  # process group initialization
    
    run_test_only()
                
    # DDP:
    torch.distributed.destroy_process_group()