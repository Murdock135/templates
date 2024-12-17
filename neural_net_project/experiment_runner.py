import torch
from src.utils import load_config, ResultsPlotter
from src.dataset_utils import CustomDataset
from src.model_trainer import ClassifierTrainer, RegressorTrainer, reset_model_weights
from src.models import CNN
import torch.optim as optim
import torch.nn as nn

def get_optimizer(model, optimizer_config: str):
    optimizer_type = optimizer_config['type']
    learning_rate = optimizer_config['learning_rate']

    if optimizer_config == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_config == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    
def get_criterion(criterion_config: str):
    criterion = criterion_config

    if criterion == "Cross Entropy":
        return nn.CrossEntropyLoss()
    elif criterion == "MSE":
        return nn.MSELoss()

if __name__ == "main":
    path_to_config = "configs/config.toml"
    config = load_config(path_to_config)
    results_path = "results/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset()
    model = CNN(1, 10).to(device)
    optimizer = get_optimizer(model, config['training']['optimizer'])
    criterion = get_criterion(config['training']['criterion'])
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    trainer = ClassifierTrainer(model=model, 
                                device=device, 
                                dataset=dataset, 
                                criterion=criterion, 
                                optimizer=optimizer,
                                epochs=epochs,
                                batch_size=batch_size)
    
    trainer.save_model_results(results_path)
    plotter = ResultsPlotter(results_path)
    plotter.plot_results()