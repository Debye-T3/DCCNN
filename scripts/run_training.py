import yaml # structure of the config
import glob # search for paths
import torch # Neural Network Framework
import gc # Garbage Collector
import random # dataset splits
from datetime import datetime # gives current time for datanames
from pathlib import Path # build paths
from torch.utils.data import DataLoader, Subset # class of PyTorch for seperation of the Dataset in batches, shuffling, multithreading and iteration of batches

# Import externs
from modules.datasets.dataset import build_dataset_from_config
from modules.models.ccnn import CCNN

# import trainer
from train.trainer import train_model

def load_config(path):
    with open(path, "r", encoding='utf-8') as f:  # 加 encoding='utf-8' 修 GBK 错
      return yaml.safe_load(f)

def run_all_configs(config_dir):
    #configs = sorted(glob.glob(f"{config_dir}/*.yaml")) # searching for all .yaml files and gives back a list with all of them
    configs = sorted(glob.glob(f"{config_dir}/baseline_v3.yaml"))  # 只跑 v2
    

    for cfg_path in configs: # for every .yaml file
        
        # initialize all config values 
        cfg = load_config(cfg_path)

        # Convert numeric strings to proper types (fix YAML loading issue where scientific notation like "1e-04" loads as str)
        if "training" in cfg:
            training = cfg["training"]
            training["learning_rate"] = float(training["learning_rate"])
            training["alpha"] = float(training["alpha"])
            training["epochs"] = int(training["epochs"])
            training["batch_size"] = int(training["batch_size"])
            training["early_stopping_patience"] = int(training["early_stopping_patience"])
            if "val_split" in training:
                training["val_split"] = float(training["val_split"])

        if "model" in cfg:
            model = cfg["model"]
            model["kernel_size"] = int(model["kernel_size"])
            model["num_layers"] = int(model["num_layers"])

        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        path_cfg = cfg["paths"]
        output_dir = Path(path_cfg["output_dir"])
        csv_dir = output_dir / path_cfg.get("csv_subdir")
        model_dir = output_dir / "models"

        # create result folder and csv, model subfolder
        output_dir.mkdir(exist_ok=True)
        csv_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # current time for filename
        mask = f"{timestamp}_layers{model_cfg['num_layers']}_batch{train_cfg['batch_size']}_kernel{model_cfg['kernel_size']}_alpha{train_cfg['alpha']}" # mask for resulting model filename 
        csv_path = csv_dir / f"{mask}.csv" # path of the .csv
        model_path = model_dir / f"{mask}.pt" # path of the .pt (model)

        device = torch.device("cuda" if train_cfg["use_gpu"] and torch.cuda.is_available() else "cpu") # use GPU-RAM for cuda calculations
        dataset = build_dataset_from_config(path_cfg, cfg.get("data")) # load dataset

        total_len = len(dataset)
        val_ratio = train_cfg.get("val_split", 0.2)
        val_size = int(max(1, total_len * val_ratio)) if total_len > 1 else 0
        if val_size >= total_len:
            val_size = max(1, total_len // 5)
        indices = list(range(total_len))
        random.Random(train_cfg.get("split_seed", 42)).shuffle(indices)
        val_indices = indices[:val_size] if val_size > 0 else indices[:1]
        train_indices = indices[val_size:] if val_size > 0 else indices
        if not train_indices:
            train_indices = indices

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            drop_last=len(train_dataset) > train_cfg["batch_size"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(train_cfg["batch_size"], len(val_dataset)),
            shuffle=False,
            drop_last=False,
        )

        model = CCNN(model_cfg["kernel_size"], model_cfg["num_layers"]) # load model

        ##########################################################################################################################################
        ##########################################################################################################################################
        model, metrics = train_model(model, train_loader, val_loader, train_cfg["epochs"], train_cfg["learning_rate"], train_cfg["alpha"], device, train_cfg["early_stopping_patience"]) #############
        ##########################################################################################################################################
        ##########################################################################################################################################


        metrics.to_csv(csv_path, index=False) # save the .csv
        torch.save(model.state_dict(), model_path) # save model weights and biases


        del model, train_loader, val_loader, dataset # delete references for gc.collect()
        torch.cuda.empty_cache() # clears GPU-RAM
        gc.collect() # Python Garbage Collector clears CPU-RAM


        # NOW COMES THE NEXT YAML!

if __name__ == "__main__":
    run_all_configs("config")