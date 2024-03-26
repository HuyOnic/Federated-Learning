import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from dataset import prepare_datasets
import torch
import pickle
from pathlib import Path
import numpy as np
from visualize import visualize
from clients import generate_client_fn
import flwr as fl
from server import get_evaluate_fn, get_on_fit_config
@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg :DictConfig):
    print(OmegaConf.to_yaml(cfg))
    #prepare datasets
    train_loaders, val_loaders, test_loaders = prepare_datasets(cfg.num_clients, cfg.batch_size)
    print(len(train_loaders))
    print(len(train_loaders[0].dataset))

    # Data distribution of each clients
    # visualize(train_loaders,cfg.num_clients)
    # Define clients 
    client_fn = generate_client_fn(train_loaders,val_loaders, cfg.num_classes)

    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001, 
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(test_loaders, cfg.num_classes))
    #simualation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            'num_cpus':2.0,
            'num_gpus':2.0,
        }
    )

    #save result
    save_path = HydraConfig.get().runtime.output_dir
    result_path = Path(save_path)/'result.pkl'
    result = {'history': history}
    with open(str(result_path),'wb') as h:
        pickle.dump(result, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ ==  '__main__':
    main()

