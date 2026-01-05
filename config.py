import torch

class Config:
    # Model config
    in_channels = 3
    num_classes = 10
    
    # Training config
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset config
    data_dir = './data'
    num_workers = 2
    
    # Reproducibility
    seed = 42
    seeds = [42, 123, 456]  # For multi-run mode
    
    # Comparison settings
    save_models = False  # Set to True to save model weights
