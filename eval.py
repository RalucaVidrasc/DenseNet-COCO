# -*- coding: utf-8 -*-
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from train import COCODataset, get_densenet_model
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

def evaluate_model(model, val_loader, device, num_classes=80):
    model.eval()
    # keep the nr of correctly classified examples and the total nr of examples for each class
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            for class_index in range(num_classes):
                total_per_class[class_index] += labels[:, class_index].sum().item()
                correct_per_class[class_index] += ((predicted[:, class_index] == labels[:, class_index]) & (labels[:, class_index] == 1)).sum().item()

    accuracies_per_class = correct_per_class / total_per_class
    return accuracies_per_class

def main(data_dir, densenet_version, epochs):
    val_dir = os.path.join(data_dir, 'val2017')
    ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    print(f"val_dir: {val_dir}")
    print(f"ann_file: {ann_file}")

    assert os.path.exists(val_dir), "Validation directory not found!"
    assert os.path.exists(ann_file), "Annotation file not found!"

    print("All paths are set correctly!")
    print(f"data_dir: {data_dir}")
    print(f"densenet_version: {densenet_version}")
    print(f"epochs: {epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = COCODataset(val_dir, ann_file, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

    categories = ['cat', 'person', 'car', 'laptop']
    category_indices = [17, 1, 2, 63]  # indices for some chosen categories

    general_accuracies = []
    class_accuracies = {category: [] for category in categories}

    for epoch in range(1, epochs + 1):
        model_path = f'densenet_{densenet_version}_{epoch}epochs.pth'
        print(f"Evaluating model for epoch {epoch}")
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, skipping.")
            continue

        model = get_densenet_model(densenet_version, 80).to(device)
        model.load_state_dict(torch.load(model_path))

        accuracies_per_class = evaluate_model(model, val_loader, device)
        general_accuracy = accuracies_per_class.mean().item()
        general_accuracies.append(general_accuracy)
        print(f"General Accuracy for epoch {epoch}: {general_accuracy}")

        for category, index in zip(categories, category_indices):
            class_accuracy = accuracies_per_class[index].item()
            class_accuracies[category].append(class_accuracy)
            print(f"Accuracy for {category} at epoch {epoch}: {class_accuracy}")

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, epochs + 1), general_accuracies, label='General Accuracy')
    for category in categories:
        plt.plot(range(1, epochs + 1), class_accuracies[category], label=f'{category.capitalize()} ')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title(f'The accuracy of  {densenet_version} depending on the number of epochs ')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DenseNet on COCO')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the COCO dataset directory')
    parser.add_argument('--densenet_version', type=str, default='densenet121', help='DenseNet version (densenet121, densenet169, densenet201)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to evaluate')
    
    args = parser.parse_args()
    main(args.data_dir, args.densenet_version, args.epochs)
