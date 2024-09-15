import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from train import COCODataset, get_densenet_model  
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main(data_dir, densenet_version, model_path):
    val_dir = os.path.join(data_dir, 'val2017')
    ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')

    assert os.path.exists(val_dir), "Validation directory not found!"
    assert os.path.exists(ann_file), "Annotation file not found!"

    print("All paths are set correctly!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The used device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = COCODataset(val_dir, ann_file, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

    model = get_densenet_model(densenet_version, 80).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # an empty confusion matrix
    cm = np.zeros((80, 80))

    
    for i in range(all_labels.shape[0]):
        true_classes = np.where(all_labels[i] == 1)[0]
        pred_classes = np.where(all_preds[i] == 1)[0]
        for t in true_classes:
            for p in pred_classes:
                cm[t, p] += 1

    # normalize the confusion matrix
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    coco = COCO(ann_file)
    class_names = [coco.cats[catId]['name'] for catId in coco.getCatIds()]

    plt.figure(figsize=(20, 20))
    cmap = sns.color_palette("flare", as_cmap=True)    
    sns.heatmap(cm, cmap=cmap, xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Preddicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix for DenseNet169 - 20 epochs')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplots_adjust(bottom=0.3, left=0.3)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DenseNet on COCO')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the COCO dataset directory')
    parser.add_argument('--densenet_version', type=str, default='densenet121', help='DenseNet version (densenet121, densenet169)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved DenseNet model ')
    
    args = parser.parse_args()
    main(args.data_dir, args.densenet_version, args.model_path)
