
import os
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import argparse
from datetime import datetime

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.coco_category_to_label = {category['id']: i for i, category in enumerate(self.coco.loadCats(self.coco.getCatIds()))}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        labels = np.zeros((80,), dtype=np.float32)
        for ann in anns:
            category_id = ann['category_id']
            if category_id in self.coco_category_to_label:
                label_index = self.coco_category_to_label[category_id]
                labels[label_index] = 1
        
        return img, torch.tensor(labels)

    def __len__(self):
        return len(self.ids)

def get_densenet_model(version, num_classes):
    if version == 'densenet121':
        model = models.densenet121(weights=None)
    elif version == 'densenet169':
        model = models.densenet169(weights=None)
    else:
        raise ValueError("Unsupported DenseNet version")
    
    # modify the final classification layer to have a nr of outputs equal to the nr of classes 
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, num_classes)
    return model

def main(data_dir, densenet_version, num_epochs, batch_size, num_workers):
    train_dir = os.path.join(data_dir, 'train2017')
    val_dir = os.path.join(data_dir, 'val2017')
    ann_file = os.path.join(data_dir, 'annotations/instances_train2017.json')

    assert os.path.exists(train_dir), "Train directory not found!"
    assert os.path.exists(val_dir), "Validation directory not found!"
    assert os.path.exists(ann_file), "Annotation file not found!"

    print("All paths are set correctly!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = COCODataset(train_dir, ann_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = get_densenet_model(densenet_version, 80).to(device)
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

    criterion = torch.nn.BCEWithLogitsLoss()
    # dynamically adjust the lr of each parameter based on the gradients estimated from previous iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        start_time = datetime.now()
        print(f"Starting epoch {epoch+1} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(images.to(device))  # move the images on the GPU
            
            loss = criterion(outputs, labels.to(device))  # move the labels on the GPU
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss}")
        
        # save the model after every epoch
        model_file = f'densenet_{densenet_version}_{epoch+1}epochs.pth'
        torch.save(model.state_dict(), model_file)
        print(f'Model saved to: {model_file}')
        
        torch.cuda.empty_cache()

    final_model_file = f'densenet_{densenet_version}_{num_epochs}epochs.pth'
    torch.save(model.state_dict(), final_model_file)
    print(f'Final model saved to: {final_model_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DenseNet on COCO')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the COCO dataset directory')
    parser.add_argument('--densenet_version', type=str, default='densenet121', help='DenseNet version (densenet121, densenet169, densenet201)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    main(args.data_dir, args.densenet_version, args.num_epochs, args.batch_size, args.num_workers)
