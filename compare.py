import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import time
from models.resnet import ResNet18
from models.cnn import CNN
from config import Config
from utils import train_one_epoch, evaluate

def train_model(model_name, model, trainloader, testloader, device, num_epochs):
    """Train a model and return training history."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
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
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        scheduler.step()
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )
    
    training_time = time.time() - start_time
    
    print(f"\n{model_name} Training finished in {training_time:.2f}s")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")
    
    return history, training_time

def main():
    # Configuration
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # Data Preparation
    print("\nPreparing data...")
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
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=Config.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=Config.num_workers)

    # Train CNN
    cnn_model = CNN(num_classes=Config.num_classes).to(device)
    cnn_history, cnn_time = train_model(
        "Vanilla CNN", cnn_model, trainloader, testloader, device, Config.num_epochs
    )
    
    # Train ResNet
    resnet_model = ResNet18(num_classes=Config.num_classes).to(device)
    resnet_history, resnet_time = train_model(
        "ResNet18", resnet_model, trainloader, testloader, device, Config.num_epochs
    )
    
    # Save results
    results = {
        'cnn': {
            'history': cnn_history,
            'training_time': cnn_time,
            'final_test_acc': cnn_history['test_acc'][-1]
        },
        'resnet': {
            'history': resnet_history,
            'training_time': resnet_time,
            'final_test_acc': resnet_history['test_acc'][-1]
        }
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Final Test Acc':<20} {'Training Time':<15}")
    print(f"{'-'*60}")
    print(f"{'Vanilla CNN':<15} {cnn_history['test_acc'][-1]:.4f} ({cnn_history['test_acc'][-1]*100:.2f}%)    {cnn_time:.2f}s")
    print(f"{'ResNet18':<15} {resnet_history['test_acc'][-1]:.4f} ({resnet_history['test_acc'][-1]*100:.2f}%)    {resnet_time:.2f}s")
    print(f"{'-'*60}")
    
    improvement = (resnet_history['test_acc'][-1] - cnn_history['test_acc'][-1]) * 100
    print(f"\nResNet18 improvement: {improvement:+.2f}%")
    print(f"\nResults saved to comparison_results.json")
    
    # Save models
    torch.save(cnn_model.state_dict(), './cnn.pth')
    torch.save(resnet_model.state_dict(), './resnet_comparison.pth')
    print("Models saved to ./cnn.pth and ./resnet_comparison.pth")

if __name__ == '__main__':
    main()
