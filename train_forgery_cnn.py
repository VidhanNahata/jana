import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class TransformSubset(Dataset):
    """
    Safely wraps a random_split Subset with a specific augmentation transform 
    to prevent Validation data from being inadvertently argumented.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        # Base ImageFolder yields (PIL Image, int)
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def build_cnn_model(device):
    """
    Leverages Transfer Learning loading ResNet-50 weights from ImageNet computations.
    """
    print("Loading pretrained ResNet-50 architecture...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Overwrite the final classification head
    # We strictly enforce a 2-neuron structure as requested. 
    num_ftrs = model.fc.in_features
    # Note: Using nn.CrossEntropyLoss automatically applies SoftMax mathematically to these raw Logit neurons
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model.to(device)

def plot_and_save_confusion_matrix(cm, classes, output_path="confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Image Forgery Classification Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[Artifact Computed] Saved confusion matrix plot to '{output_path}'")

def train_cnn_pipeline():
    """
    Main training execution block utilizing Early Stopping, Adam optimization, 
    and 80/20 train-val splitting over the preprocessed ELA dataset.
    """
    data_dir = "ela_output/"
    
    if not os.path.exists(data_dir):
        print(f"Error: Target ELA input directory '{data_dir}' not found.")
        print("Please ensure the initial preprocessing pipeline has generated the data structure.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CUDA status: Using acceleration device ({device})")

    # 1. Pipeline Definition: Augmentation + Normalization bounds 
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),         # Random Horizontal Flip
            transforms.RandomRotation(15),                  # Random Degree Rotation bounds
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Jitter adjustments
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # 2. Extract ground truth structure dynamically
    base_dataset = datasets.ImageFolder(data_dir, transform=None)
    class_names = base_dataset.classes
    print(f"Located active classes mapping: {class_names}")

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], 
                                          generator=torch.Generator().manual_seed(42))

    # Apply strict segregated Transforms (never augment evaluation metrics)
    train_dataset = TransformSubset(train_subset, transform=data_transforms['train'])
    val_dataset = TransformSubset(val_subset, transform=data_transforms['val'])

    # 3. Construct multi-threaded data loaders
    # Batch size explicitly locked to 32 as requested
    batch_size = 32
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    print(f"Structured Loaders -> Train: {train_size} | Validation: {val_size}\n")
    
    # Init configurations
    model = build_cnn_model(device)
    
    # Binary Cross Entropy optimization mechanism (mathematically combined internally by CE against a [x, y] 2-neuron target matrix)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Variables for state tracking and Early Stopping (Monitor: Val Acc)
    num_epochs = 20
    patience = 5
    epochs_no_improve = 0
    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    start_time = time.time()

    # --------------- EXECUTION LOOP -----------------
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} -> ", end="")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Step operations
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Autograd calculation boundary
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropogate only if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            # Aggregate metrics
            dataset_size = train_size if phase == 'train' else val_size
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}  |  ", end="")

            # Evaluation conditional logic
            if phase == 'val':
                print("") # Nextline formatting
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Evaluate Early Stop criteria 
        if epochs_no_improve >= patience:
            print(f"\n[!] Triggered Early Stopping. Val Accuracy has stalled for {patience} epochs.")
            break

    # --------------- TERMINATION -----------------
    time_elapsed = time.time() - start_time
    print(f"\nOptimization Finished (Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s)")
    print(f"Target Threshold (>= 0.9400) vs Peak Accuracy ({best_acc:.4f})")
    
    # Save parameters block
    model.load_state_dict(best_model_weights)
    target_disk_path = "ela_cnn_model.pth"
    torch.save(model.state_dict(), target_disk_path)
    print(f"Exported model weights successfully to '{target_disk_path}'")

    # Final Classification Report and Matrix Compilation
    print("\n--- Constructing Final Classifier Report Validation Output ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n[Scikit-Learn Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Plot formatting graph output
    plot_and_save_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    train_cnn_pipeline()
