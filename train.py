import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.resnet import ResNet18
from config import Config
from utils import train_one_epoch, evaluate

def main():
    # Configuration
    device = torch.device(Config.device)
    print(f"Using device: {device}")

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
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=Config.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=Config.num_workers)

    # Model
    print("Building model...")
    net = ResNet18(num_classes=Config.num_classes)
    net = net.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=Config.learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs
    )

    # Training Loop
    print("Starting training...")
    for epoch in range(Config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            net, trainloader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(net, testloader, criterion, device)
        
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.num_epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    print("Training finished.")
    
    # Save model
    torch.save(net.state_dict(), './resnet.pth')
    print("Model saved to ./resnet.pth")

if __name__ == '__main__':
    main()
