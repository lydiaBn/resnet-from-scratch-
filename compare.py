import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from models.resnet import ResNet18
from models.cnn import CNN
from config import Config
from utils import train_one_epoch, evaluate

def set_seed(seed):
    """Set random seed for reproducibility.
    
    Note: Full determinism may reduce performance due to disabling CUDNN optimizations.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Full determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model_name, model, trainloader, testloader, device, num_epochs, seed):
    """Train a model and return training history.
    
    Note: train_one_epoch sets model.train()
          evaluate sets model.eval() and disables gradients
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (seed={seed})")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, trainloader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
    
    training_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times)
    
    print(f"\n{model_name} Training finished in {training_time:.2f}s")
    print(f"Average time per epoch: {avg_epoch_time:.2f}s")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")
    
    return history, training_time, avg_epoch_time

def plot_results(results, num_runs):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ResNet18 vs Vanilla CNN Comparison on CIFAR-10', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(results['cnn']['runs'][0]['history']['train_acc']) + 1)
    
    # Plot 1: Test Accuracy Comparison
    ax = axes[0, 0]
    for i in range(num_runs):
        ax.plot(epochs, results['cnn']['runs'][i]['history']['test_acc'], 
                color='blue', alpha=0.3, linewidth=1)
        ax.plot(epochs, results['resnet']['runs'][i]['history']['test_acc'], 
                color='red', alpha=0.3, linewidth=1)
    
    # Plot means
    cnn_test_acc_mean = np.mean([r['history']['test_acc'] for r in results['cnn']['runs']], axis=0)
    resnet_test_acc_mean = np.mean([r['history']['test_acc'] for r in results['resnet']['runs']], axis=0)
    
    ax.plot(epochs, cnn_test_acc_mean, color='blue', linewidth=2.5, label='CNN (mean)')
    ax.plot(epochs, resnet_test_acc_mean, color='red', linewidth=2.5, label='ResNet18 (mean)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Train Accuracy Comparison
    ax = axes[0, 1]
    cnn_train_acc_mean = np.mean([r['history']['train_acc'] for r in results['cnn']['runs']], axis=0)
    resnet_train_acc_mean = np.mean([r['history']['train_acc'] for r in results['resnet']['runs']], axis=0)
    
    ax.plot(epochs, cnn_train_acc_mean, color='blue', linewidth=2.5, label='CNN')
    ax.plot(epochs, resnet_train_acc_mean, color='red', linewidth=2.5, label='ResNet18')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Accuracy')
    ax.set_title('Train Accuracy Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Test Loss Comparison
    ax = axes[1, 0]
    cnn_test_loss_mean = np.mean([r['history']['test_loss'] for r in results['cnn']['runs']], axis=0)
    resnet_test_loss_mean = np.mean([r['history']['test_loss'] for r in results['resnet']['runs']], axis=0)
    
    ax.plot(epochs, cnn_test_loss_mean, color='blue', linewidth=2.5, label='CNN')
    ax.plot(epochs, resnet_test_loss_mean, color='red', linewidth=2.5, label='ResNet18')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Accuracy Distribution
    ax = axes[1, 1]
    cnn_final_accs = [r['final_test_acc'] for r in results['cnn']['runs']]
    resnet_final_accs = [r['final_test_acc'] for r in results['resnet']['runs']]
    
    positions = [1, 2]
    bp = ax.boxplot([cnn_final_accs, resnet_final_accs], 
                     positions=positions,
                     widths=0.6,
                     patch_artist=True,
                     labels=['CNN', 'ResNet18'])
    
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title(f'Final Test Accuracy Distribution ({num_runs} runs)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to comparison_plots.png")
    plt.close()

def main():
    # Set initial seed
    set_seed(Config.seed)
    
    # Configuration
    device = torch.device(Config.device)
    print(f"Using device: {device}")
    print(f"Running {len(Config.seeds)} independent runs with seeds: {Config.seeds}")
    print(f"Note: Full determinism enabled (may reduce performance)\n")

    # Data Preparation
    print("Preparing data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=Config.data_dir, train=True, download=True, transform=transform_train)
    
    testset = torchvision.datasets.CIFAR10(
        root=Config.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=Config.num_workers)

    # Initialize models to count parameters
    cnn_model_temp = CNN(num_classes=Config.num_classes)
    resnet_model_temp = ResNet18(num_classes=Config.num_classes)
    
    cnn_params = count_params(cnn_model_temp)
    resnet_params = count_params(resnet_model_temp)
    
    print(f"\nModel Parameters:")
    print(f"  CNN: {cnn_params:,} parameters")
    print(f"  ResNet18: {resnet_params:,} parameters")
    print(f"  Ratio: {resnet_params/cnn_params:.2f}x\n")
    
    del cnn_model_temp, resnet_model_temp

    # Multi-run training with full histories stored separately
    full_histories = {'cnn': [], 'resnet': []}
    results = {
        'cnn': {'runs': [], 'params': cnn_params},
        'resnet': {'runs': [], 'params': resnet_params}
    }
    
    for seed in Config.seeds:
        set_seed(seed)
        
        # Create DataLoader with seeded generator for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=Config.batch_size, shuffle=True, 
            num_workers=Config.num_workers, generator=g
        )
        
        # Train CNN
        cnn_model = CNN(num_classes=Config.num_classes).to(device)
        cnn_history, cnn_time, cnn_epoch_time = train_model(
            "Vanilla CNN", cnn_model, trainloader, testloader, device, Config.num_epochs, seed
        )
        
        # Store full history separately
        full_histories['cnn'].append({
            'seed': seed,
            'history': cnn_history
        })
        
        # Store summary only in main results
        results['cnn']['runs'].append({
            'seed': seed,
            'final_test_acc': cnn_history['test_acc'][-1],
            'training_time': cnn_time,
            'avg_epoch_time': cnn_epoch_time
        })
        
        # Train ResNet
        set_seed(seed)  # Reset seed for fair comparison
        g.manual_seed(seed)  # Reset generator
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=Config.batch_size, shuffle=True, 
            num_workers=Config.num_workers, generator=g
        )
        
        resnet_model = ResNet18(num_classes=Config.num_classes).to(device)
        resnet_history, resnet_time, resnet_epoch_time = train_model(
            "ResNet18", resnet_model, trainloader, testloader, device, Config.num_epochs, seed
        )
        
        # Store full history separately
        full_histories['resnet'].append({
            'seed': seed,
            'history': resnet_history
        })
        
        # Store summary only in main results
        results['resnet']['runs'].append({
            'seed': seed,
            'final_test_acc': resnet_history['test_acc'][-1],
            'training_time': resnet_time,
            'avg_epoch_time': resnet_epoch_time
        })
        
        # Optionally save models
        if Config.save_models:
            torch.save(cnn_model.state_dict(), f'./cnn_seed{seed}.pth')
            torch.save(resnet_model.state_dict(), f'./resnet_seed{seed}.pth')
    
    # Compute statistics
    cnn_accs = [r['final_test_acc'] for r in results['cnn']['runs']]
    resnet_accs = [r['final_test_acc'] for r in results['resnet']['runs']]
    cnn_times = [r['avg_epoch_time'] for r in results['cnn']['runs']]
    resnet_times = [r['avg_epoch_time'] for r in results['resnet']['runs']]
    
    results['cnn']['mean_acc'] = float(np.mean(cnn_accs))
    results['cnn']['std_acc'] = float(np.std(cnn_accs))
    results['cnn']['mean_epoch_time'] = float(np.mean(cnn_times))
    
    results['resnet']['mean_acc'] = float(np.mean(resnet_accs))
    results['resnet']['std_acc'] = float(np.std(resnet_accs))
    results['resnet']['mean_epoch_time'] = float(np.mean(resnet_times))
    
    # Save summary results (clean JSON)
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Optionally save full histories (for plotting, ignored by git)
    # Uncomment if needed:
    # import pickle
    # with open('raw_runs.pkl', 'wb') as f:
    #     pickle.dump(full_histories, f)
    
    # Generate plots (using full histories)
    # Temporarily add histories back for plotting
    for i, run in enumerate(results['cnn']['runs']):
        run['history'] = full_histories['cnn'][i]['history']
    for i, run in enumerate(results['resnet']['runs']):
        run['history'] = full_histories['resnet'][i]['history']
    
    plot_results(results, len(Config.seeds))
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Params':<15} {'Mean Acc':<20} {'Epoch Time':<15}")
    print(f"{'-'*70}")
    print(f"{'Vanilla CNN':<15} {cnn_params/1e6:.2f}M         {results['cnn']['mean_acc']:.4f} ±{results['cnn']['std_acc']:.4f}    {results['cnn']['mean_epoch_time']:.2f}s")
    print(f"{'ResNet18':<15} {resnet_params/1e6:.2f}M         {results['resnet']['mean_acc']:.4f} ±{results['resnet']['std_acc']:.4f}    {results['resnet']['mean_epoch_time']:.2f}s")
    print(f"{'-'*70}")
    
    improvement = (results['resnet']['mean_acc'] - results['cnn']['mean_acc']) * 100
    param_ratio = resnet_params / cnn_params
    time_ratio = results['resnet']['mean_epoch_time'] / results['cnn']['mean_epoch_time']
    acc_per_param_cnn = results['cnn']['mean_acc'] / (cnn_params / 1e6)
    acc_per_param_resnet = results['resnet']['mean_acc'] / (resnet_params / 1e6)
    
    print(f"\nResNet18 vs CNN:")
    print(f"  Accuracy improvement: {improvement:+.2f}%")
    print(f"  Parameter ratio: {param_ratio:.2f}x")
    print(f"  Training time ratio: {time_ratio:.2f}x")
    print(f"  Accuracy per million params (CNN): {acc_per_param_cnn:.4f}")
    print(f"  Accuracy per million params (ResNet): {acc_per_param_resnet:.4f}")
    
    print(f"\nResults saved to comparison_results.json")
    print(f"Plots saved to comparison_plots.png")
    
    if Config.save_models:
        print(f"\nModels saved with seed suffixes")
    
    print(f"\nNote: Both models were trained under identical hyperparameters")
    print(f"      to isolate architectural differences; no per-model")
    print(f"      hyperparameter tuning was performed.")

if __name__ == '__main__':
    main()
