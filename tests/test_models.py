import unittest
import torch
import os
import datetime

from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import load_model, load_config
import src.graphs_generation as gen_graphs

# Load configuration file
configfile_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "docs",
    "cnn_exp_config.yml",
)
configfile = load_config(configfile_path)

# Define graph size for tests (choosing last, largest value)
graph_size = configfile["graph_size_values"][-1]

class ModelTest(unittest.TestCase):

    def generate_and_predict(self, model, model_name, device_id):
        """Helper to generate graphs and check predictions."""
        clique_size = int(
            graph_size
            * (configfile["training_parameters"]["max_clique_size_proportion"])
        )
        input_magnification = True if "CNN" in model_name else False
        # Generate local batch (2 graphs)
        graphs = gen_graphs.generate_batch(
            2,
            graph_size,
            [clique_size, clique_size],
            configfile["p_correction_type"],
            input_magnification,
        )[0].to(device_id)
        # Model inference
        prediction = model(graphs).squeeze()
        self.assertEqual(
            prediction.size(),
            torch.Size([2]),
            f"Output size mismatch for model {model_name}",
        )
        self.assertTrue(
            torch.all(prediction >= 0) and torch.all(prediction <= 1),
            f"Predictions out of range for model {model_name}",
        )
        # Correct definition of Scratch/pretrained ViT models
        if "ViTscratch" in model_name:
            # Check that all layers are trainable
            for name, param in model.named_parameters():
                self.assertTrue(param.requires_grad)
        elif "ViTpretrained" in model_name:
            # Pretrained model layers should not be trained
            for name, param in model.named_parameters():
                if any(key in name for key in ["cls_token", "embed", "head"]):
                    self.assertTrue(param.requires_grad)
                else:
                    self.assertFalse(param.requires_grad)        
        del graphs

    def trainability_check(self, model, model_name, world_size, rank, device_id):
        """Test model's ability to perform a forward/backward pass splitting the data across GPUs like during training."""

        input_magnification = True if "CNN" in model_name else False        
        
        # Optimizer and loss function
        optimizer = self.get_optimizer(model)
        criterion = self.get_loss_function()
        
        clique_size = int(
            graph_size
            * (configfile["training_parameters"]["max_clique_size_proportion"])
        )       
        
        # Split training data across GPUs, checking divisibility of batch size by world size
        if configfile["training_parameters"]["num_train"] % world_size != 0:
            raise ValueError(
                f"{model_name} Trainability test: Batch size of {configfile['training_parameters']['num_train']} is not evenly divisible by world_size={world_size}. "
                f"{model_name} Trainability test: Each rank requires an equal share of the data for DDP. Please adjust 'num_train' to be divisible by {world_size}."
            )
        # If no errors, proceed with splitting            
        local_batch_size = configfile["training_parameters"]["num_train"] // world_size
        start_idx_train = rank * local_batch_size
        end_idx_train = (rank + 1) * local_batch_size          

        # Generating training data on CPU (full batch)
        if rank == 0:
            print(f"{model_name} Trainability test, rank {rank}: Data generation start {datetime.datetime.now()}")        
        full_data = gen_graphs.generate_batch(
            configfile["training_parameters"]["num_train"],
            graph_size,
            [clique_size] * configfile["training_parameters"]["num_train"],
            configfile["p_correction_type"],
            input_magnification,
        )          
        if rank == 0:
            print(f"{model_name} Trainability test, rank {rank}: Data generation finish {datetime.datetime.now()}")                 

        # Partition data for the current rank
        split_data = (
            torch.Tensor(full_data[0][start_idx_train:end_idx_train]).to(device_id),
            torch.Tensor(full_data[1][start_idx_train:end_idx_train]).to(device_id),
        )
        if rank == 0:
            print(f"{model_name} Trainability test, rank {rank}: Data partitioned and send to device at {datetime.datetime.now()}")        
        
        # Barrier to ensure all processes reach this point
        torch.distributed.barrier()
        
        # Forward pass on training data
        train_pred = model(split_data[0]).squeeze()
        train_loss = criterion(train_pred.type(torch.float), torch.Tensor(split_data[1]).type(torch.float))            
        if rank == 0:
            print(f"{model_name} Trainability test, rank {rank}: Forward pass completed at {datetime.datetime.now()}")        

        # Synchronize before backward pass
        torch.distributed.barrier()

        # Backward pass
        train_loss.backward()   # DDP GRADIENT SYNCHRONIZATION HAPPENS HERE
        optimizer.step()
        if rank == 0:
            print(f"{model_name} Trainability test, rank {rank}: Backward pass and optimizer step completed at {datetime.datetime.now()}")        
        
        # Making sure batches are correctly split across GPUs
        print(f"{model_name} Trainability test (loss calculation), rank {rank}: GPU {device_id} is processing {split_data[0].shape[0]} graphs.")
        print(f"{model_name} Trainability test (loss calculation), rank {rank}: The full batch contains {full_data[0].shape[0]} graphs.")
        # Making sure global loss is computed correctly:
        print(f"{model_name} Trainability test (loss calculation), rank {rank}: Training loss on GPU {device_id} is {train_loss}.")           
        
        # Synchronize after backward pass
        torch.distributed.barrier()         

        # Aggregating training loss across GPUs to make sure it is computed correctly:
        train_loss_tensor = torch.tensor(train_loss.item(), device=device_id)
        torch.distributed.all_reduce(train_loss_tensor, op=torch.distributed.ReduceOp.SUM)        
        if rank == 0:
            global_train_loss = train_loss_tensor.item() / world_size                
            
            print(f"{model_name} Trainability test (loss calculation): Global training loss averaged across GPUs is {global_train_loss}.")                        
                        
        del full_data, split_data, train_pred

    def get_optimizer(self, model):
        """Retrieve optimizer from configuration."""
        optimizer_type = configfile["training_parameters"]["optimizer"]
        lr = configfile["training_parameters"]["learning_rate"]
        if optimizer_type == "Adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Invalid optimizer in configuration")

    def get_loss_function(self):
        """Retrieve loss function from configuration."""
        loss_function_type = configfile["training_parameters"]["loss_function"]
        if loss_function_type == "BCELoss":
            return torch.nn.BCELoss()
        elif loss_function_type == "CrossEntropyLoss":
            return torch.nn.CrossEntropyLoss()
        elif loss_function_type == "MSELoss":
            return torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss function in configuration")


def generate_ddp_tests():
    """Dynamically create test cases for each model in the configuration."""
    for idx, model_specs in enumerate(configfile["models"]):
        model_name = model_specs["model_name"]

        def test_case(self, model_specs=model_specs, model_name=model_name):
            
            # DDP:
            rank = torch.distributed.get_rank() # identifies processes (in this context, one process per GPU)
            device_id = rank % torch.cuda.device_count()
            if rank == 0:
                print(f"Started running tests with graph size {graph_size} on rank {rank} at {datetime.datetime.now()}")    
            world_size = torch.cuda.device_count() 
            
            try:
                if rank == 0:
                    print(f"Rank {rank}: Model loading start {datetime.datetime.now()}")
                # Load model
                model = load_model(model_specs, graph_size, configfile["training_parameters"]["num_train"], world_size, rank, device_id)
                if rank == 0:
                    print(f"Rank {rank}: Model loading finish {datetime.datetime.now()}")                                       
                # Synchronize after model loading
                torch.distributed.barrier()

                # Test prediction
                if rank == 0:
                    print(f"Rank {rank}: Prediction test start {datetime.datetime.now()}")                
                self.generate_and_predict(model, model_name, device_id)
                if rank == 0:
                    print(f"Rank {rank}: Prediction test finish {datetime.datetime.now()}")                  
                # Synchronize after prediction testing
                torch.distributed.barrier()

                # Test trainability (across GPUs)
                if rank == 0:
                    print(f"Rank {rank}: {model_name} Trainability test start {datetime.datetime.now()}")                  
                self.trainability_check(model, model_name, world_size, rank, device_id)
                if rank == 0:
                    print(f"Rank {rank}: {model_name} Trainability test finish {datetime.datetime.now()}")                  
                # Synchronize GPUs after trainability testing
                torch.distributed.barrier()
                del model

            finally:
                torch.cuda.empty_cache()

        test_name = f"test_{model_name}_ddp"
        setattr(ModelTest, test_name, test_case)


# Generate DDP test cases
generate_ddp_tests()

if __name__ == "__main__":
    unittest.main()