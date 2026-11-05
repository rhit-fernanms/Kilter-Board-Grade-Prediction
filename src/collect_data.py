import torch, torchvision, re, os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler

class ImageDataset(Dataset):
    def __init__(self, images, angles, grades):
        self.images = images
        self.angles = np.array([angles])
        self.grades = grades
        
    def __len__(self):
        return len(self.grades)

    def __getitem__(self, idx):
        return (self.images[idx], self.angles[:, idx], self.grades[idx])


def collect_data(data_dir, preprocess: torchvision.transforms.Compose):
    images, climb_ids, angles, labels = [], [], [], []
    for filename in sorted(os.listdir(data_dir)):
        # Images are titled 'id=[id]_angle=[angle]_grade=[grade].png'
        id, angle, grade = re.findall(r'id=(.+)_angle=(.+)_grade=(.+)\.png', filename)[0]
        climb_ids.append(id)
        angles.append(int(angle))
        labels.append(int(grade) - 10) # Standardize to [0, 22]
        
        img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
        images.append(preprocess(img).numpy())
    
    images =  np.stack(images)
    angles = np.stack(angles)
    grades = np.array(labels, dtype=np.int64)
    
    # TODO: shuffle data before splitting

    # Separate train/val/test splits
    images_train = images[:80]; angles_train = angles[:80]; grades_train = grades[:80]
    images_val = images[80:]; angles_val = angles[80:]; grades_val = grades[80:]
    
    
    
    # Hyperparameters
    batch_size = 64

    # Use a WeightedRandomSampler
    # SELECT COUNT(DISTINCT climbs.climbUuid) FROM climbs INNER JOIN climb_stats ON climb_stats.climbUuid = climbs.climbUuid WHERE ascentCount >= 5 AND climb_stats.product_layout_uuid = '10' AND currentDifficultyId >= 10 GROUP BY currentDifficultyId;
    grade_counts = [334, 795, 916, 1324, 1609, 2003, 2932, 2501, 2854, 2420, 2933, 2367, 3640, 2749, 2249, 1318, 1453, 995, 664, 246, 46, 1, 0]
    # Use weights that pick the less common grades more often
    class_weights = [0 if grade_counts[i] == 0 else 1 / grade_counts[i] for i in range(len(grade_counts))]
    train_weights = [class_weights[grades_train[i]] for i in range(len(grades_train))]

    train_sampler = BatchSampler(WeightedRandomSampler(torch.DoubleTensor(train_weights), len(grades_train)), batch_size, drop_last=False)
    train_dataset = ImageDataset(torch.from_numpy(images_train), torch.from_numpy(angles_train), torch.from_numpy(grades_train))
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    # Use uniform random selection for validation set
    val_dataset = ImageDataset(torch.from_numpy(images_val), torch.from_numpy(angles_val), torch.from_numpy(grades_val))
    val_loader = DataLoader(val_dataset, batch_size=len(images_val)) # Treat whole dataset as a batch; allows reuse of loops/code

    # test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)), batch_size=batch_size)
    
    return train_loader, val_loader
    
    
import plotly.graph_objects as go
import time
    
# Taken from Lesson 19
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path):
    """Train, keep the checkpoint with the lowest validation loss, and plot the losses."""
    training_losses = []
    validation_losses = []
    training_accs = []
    validation_accs = []
    best_validation_loss = float('inf')
    best_validation_epoch = -1
    training_start = time.time()
    records = len(train_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        num_batches = 0
        for images, angles, labels in train_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            outputs_class = torch.argmax(outputs, dim=1)
            optimizer.zero_grad() 
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            
            epoch_loss += loss.item() * len(images)
            epoch_correct += (labels == outputs_class).sum().item()
            # print(f'{epoch_correct:.1f} correct predictions during current epoch')
            num_batches += 1

        avg_loss = epoch_loss / records
        train_acc = epoch_correct / records
        training_losses.append(avg_loss)
        training_accs.append(train_acc)

        # Compute validation loss. Don't need gradients here.
        model.eval()
        with torch.no_grad():
            val_correct = 0
            for images, angles, labels in val_loader: # loads full validiaiton datset as a batch
                images = images.to(device)
                angles = angles.to(device)
                labels = labels.to(device)
                val_outputs = model(images, angles)
                validation_loss = criterion(val_outputs, labels).item() 
                val_outputs_class = torch.argmax(val_outputs, dim=1)
                val_correct = (labels == val_outputs_class).sum().item()
                validation_losses.append(validation_loss)
            val_acc = val_correct / records
            validation_accs.append(val_acc)
            
            

        # Save the model if this is the best validation loss so far
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            best_marker = ' *'
        else:
            best_marker = ''

        if epoch % (num_epochs / 10) == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.5f}, Val Loss: {validation_loss:.5f}{best_marker}, Train Acc: {train_acc:.5f}, Validation Acc: {val_acc:.5f}')

    total_time = time.time() - training_start
    print(f'Average time per epoch: {total_time / num_epochs:.2f}s')
    print(f'Best validation loss: {best_validation_loss:.5f} (epoch {best_validation_epoch})')

    # Loss plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(len(training_losses))), y=training_losses, mode='lines', name='Training Loss'))
    fig1.add_trace(go.Scatter(x=list(range(len(validation_losses))), y=validation_losses, mode='lines', name='Validation Loss'))
    fig1.update_layout(title="Training and Validation Loss Over Time", xaxis_title="Epoch", yaxis_title="Loss")
    fig1.show()
    # Accuracy plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(training_accs))), y=training_accs, mode='lines', name='Training Accuracy'))
    fig2.add_trace(go.Scatter(x=list(range(len(validation_accs))), y=validation_accs, mode='lines', name='Validation Accuracy'))
    fig2.update_layout(title="Training and Validation Accuracy Over Time", xaxis_title="Epoch", yaxis_title="Accuracy")
    fig2.show()
    