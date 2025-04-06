import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 50
MIN_SAMPLES_PER_CLASS = 15  # Minimum samples required to keep a class

# Enhanced Dataset Class with class filtering
class VehicleDamageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path, sep='|')
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter rare classes
        self._filter_rare_classes()
        
    def _filter_rare_classes(self):
        """Remove samples from classes with insufficient examples"""
        for task in ['Tipos de Daño', 'Piezas del Vehículo', 'Sugerencia']:
            class_counts = self.data[task].value_counts()
            valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
            self.data = self.data[self.data[task].isin(valid_classes)]
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        
        labels = {
            'damage': torch.tensor(self.data.iloc[idx, 1] - 1, dtype=torch.long),
            'part': torch.tensor(self.data.iloc[idx, 2] - 1, dtype=torch.long),
            'suggestion': torch.tensor(self.data.iloc[idx, 3] - 1, dtype=torch.long)
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

# Enhanced Multi-Task Model with Simplified Architecture
class DamageClassifier(nn.Module):
    def __init__(self, num_damage_types, num_parts, num_suggestions):
        super().__init__()
        
        # Simpler pretrained backbone
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Shared layers with more regularization
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Task-specific heads
        self.damage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_damage_types)
        )
        self.part_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, num_parts)
        )
        self.suggestion_head = nn.Linear(256, num_suggestions)
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        
        return {
            'damage': self.damage_head(shared),
            'part': self.part_head(shared),
            'suggestion': self.suggestion_head(shared)
        }

# Enhanced Training Loop
def train_model(model, train_loader, val_loader, num_epochs):
    # Class-weighted loss functions
    damage_weights = get_class_weights(train_dataset, 'damage')
    part_weights = get_class_weights(train_dataset, 'part')
    suggestion_weights = get_class_weights(train_dataset, 'suggestion')
    
    criterion = {
        'damage': nn.CrossEntropyLoss(weight=damage_weights),
        'part': nn.CrossEntropyLoss(weight=part_weights),
        'suggestion': nn.CrossEntropyLoss(weight=suggestion_weights)
    }
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    best_metrics = {'damage': 0, 'part': 0, 'suggestion': 0}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = {k: v.to(DEVICE) for k, v in labels.items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Weighted multi-task loss
            loss = 0.4 * criterion['damage'](outputs['damage'], labels['damage']) + \
                   0.4 * criterion['part'](outputs['part'], labels['part']) + \
                   0.2 * criterion['suggestion'](outputs['suggestion'], labels['suggestion'])
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {running_loss/len(train_loader):.4f}')
        for task in val_metrics:
            print(f'{task} Accuracy: {val_metrics[task]:.4f}')
            
        # Update learning rate
        avg_acc = sum(val_metrics.values()) / 3
        scheduler.step(avg_acc)
        
    return model

def get_class_weights(dataset, task):
    """Calculate inverse frequency class weights"""
    class_counts = Counter(getattr(dataset, f'get_{task}_counts')())
    weights = 1.0 / torch.tensor([class_counts[i] for i in range(len(class_counts))], dtype=torch.float)
    return (weights / weights.sum()).to(DEVICE)

def evaluate_model(model, loader):
    """Enhanced evaluation with per-task metrics"""
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for task in ['damage', 'part', 'suggestion']:
            all_preds = []
            all_labels = []
            
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                
                _, preds = torch.max(outputs[task], 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels[task].cpu().numpy())
            
            # Calculate metrics
            metrics[task] = accuracy_score(all_labels, all_preds)
            
            # Generate classification report
            print(f'\nClassification Report for {task}:')
            print(classification_report(all_labels, all_preds, zero_division=0))
            
            # Plot confusion matrix
            plot_confusion_matrix(all_labels, all_preds, task)
    
    return metrics

def plot_confusion_matrix(true, pred, task_name):
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Main execution
if __name__ == '__main__':
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    train_dataset = VehicleDamageDataset(
        'data/fotos_siniestros/datasets/train.csv',
        'data/fotos_siniestros/',
        data_transforms['train']
    )
    
    val_dataset = VehicleDamageDataset(
        'data/fotos_siniestros/datasets/val.csv',
        'data/fotos_siniestros/',
        data_transforms['val']
    )
    
    # Create data loaders with balanced sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            weights=get_sample_weights(train_dataset),
            num_samples=len(train_dataset),
            replacement=True
        ),
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = DamageClassifier(
        num_damage_types=len(label_to_cls_danos),
        num_parts=len(label_to_cls_piezas),
        num_suggestions=len(label_to_cls_sugerencia)
    ).to(DEVICE)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, NUM_EPOCHS)
    
    # Save model
    torch.save(trained_model.state_dict(), 'improved_damage_classifier_by_blackboxAI.pth')
