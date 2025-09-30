import torch
import os

# custom imports
from src.utils import load_config, load_model, save_features

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# # TO RUN FROM HOME DIRECTORY (NOTE: 1 GPU is sufficient in this case):
# # resource allocation:
# salloc --nodes=1 --gres=gpu:1 --mem=32G --ntasks=1 --cpus-per-task=8 -A Sis25_piasini -p boost_usr_prod --qos=boost_qos_dbg
# # command to run
# torchrun --standalone --nproc_per_node=1 -m tests.save_features.save_features_test

n_gpus = torch.cuda.device_count()

# DDP (here, using one process per GPU):
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.distributed.init_process_group("nccl")  # process group initialization
rank = (
    torch.distributed.get_rank()
)  # identifies processes (in this context, one process per GPU)
device_id = rank % torch.cuda.device_count()
print(f"Running 'save_features_test' on device id: {device_id}.")
world_size = torch.cuda.device_count()
print("world size is: ", world_size)

# reading configuration and model name:
config = load_config(os.path.join("docs", "cnn_exp_config.yml"))
model_name = "CNN_large"
# reading model configuration:
model_config = [
    model for model in config["models"] if model["model_name"] == model_name
][0]
# creating the model:
model = load_model(model_config, config["graph_size_values"][0], config["testing_parameters"]["num_test"], world_size, rank, device_id).to(device_id)

# load the best model from the training process on current rank
# - defining file name and path:
file_path = os.path.join(
    current_dir,
    "..",
    "..",
    "results",
    "data",
    "cnn_exp_2025-09-05_11-06-38",  # CHANGE THIS
    f"N{config['graph_size_values'][0]}",  # CHANGE THIS
    model_name,
    f"{model_name}_N{config['graph_size_values'][0]}_trained.pth"
)     
# - configuring map location:
map_location = {"cuda:%d" % 0: "cuda:%d" % device_id}
print(map_location)
# - loading the model:
state_dict = torch.load(file_path, map_location=map_location)
model.load_state_dict(state_dict)
# - putting the model in evaluation mode before starting save_features:
model.eval()

# call "save_features" function:
save_features(
    model,
    model_name,
    config['graph_size_values'][0],
    config["p_correction_type"],
    current_dir,
    device_id
)
