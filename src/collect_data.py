import torch, torchvision, re, os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
from random import shuffle
import copy

# Hyperparameters
BATCH_SIZE = 64

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
    images, angles, grades = [], [], []
    i = 0
    for filename in sorted(os.listdir(data_dir)):
        # Images are titled 'id=[id]_angle=[angle]_grade=[grade].png'
        id, angle, grade = re.findall(r'id=(.*)_angle=(.*)_grade=(.+)\.png', filename)[0]
        if angle == '': # Some climbs don't have angle data
            continue

        angles.append(int(angle))
        grades.append(int(grade) - 10) # Standardize to [0, 22]
        
        img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
        images.append(preprocess(img).numpy())
        
        i += 1
        if i % 1000 == 0:
            print(f'{i} images processed')
        if i == 12000: # Stop at 12000 images to save memory
            break
    
    # Shuffle data before splitting
    shuffled = list(zip(images, angles, grades))
    shuffle(shuffled)
    images, angles, grades = zip(*shuffled)
    
    images =  np.stack(images)
    angles = np.stack(angles)
    grades = np.array(grades, dtype=np.int64)

    # Separate train/val/test splits (70 / 15 / 15)
    train_count = int(len(images) * 0.70)
    val_count = int(len(images) * 0.15)
    images_train = images[:train_count]; angles_train = angles[:train_count]; grades_train = grades[:train_count]
    images_val = images[train_count:train_count + val_count]; angles_val = angles[train_count:train_count + val_count]; grades_val = grades[train_count:train_count + val_count]
    images_test = images[train_count + val_count:]; angles_test = angles[train_count + val_count:]; grades_test = grades[train_count + val_count:]
    
    

    # Use a WeightedRandomSampler
    # SELECT COUNT(DISTINCT climbs.climbUuid) FROM climbs INNER JOIN climb_stats ON climb_stats.climbUuid = climbs.climbUuid WHERE ascentCount >= 5 AND climb_stats.product_layout_uuid = '10' AND currentDifficultyId >= 10 GROUP BY currentDifficultyId;
    grade_counts = [334, 795, 916, 1324, 1609, 2003, 2932, 2501, 2854, 2420, 2933, 2367, 3640, 2749, 2249, 1318, 1453, 995, 664, 246, 46, 1, 0]
    # Use weights that pick the less common grades more often
    class_weights = [0 if grade_counts[i] == 0 else 1 / grade_counts[i] for i in range(len(grade_counts))]
    train_weights = [class_weights[grades_train[i]] for i in range(len(grades_train))]

    train_sampler = BatchSampler(WeightedRandomSampler(torch.DoubleTensor(train_weights), len(grades_train)), BATCH_SIZE, drop_last=False)
    train_dataset = ImageDataset(torch.from_numpy(images_train), torch.from_numpy(angles_train), torch.from_numpy(grades_train))
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    # Use uniform random selection for validation set
    val_dataset = ImageDataset(torch.from_numpy(images_val), torch.from_numpy(angles_val), torch.from_numpy(grades_val))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) # calculates by batch, but will be aggregated to be as though it were a single batch

    test_dataset = ImageDataset(torch.from_numpy(images_test), torch.from_numpy(angles_test), torch.from_numpy(grades_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE) # calculates by batch, but will be aggregated to be as though it were a single batch
    
    return train_loader, val_loader, test_loader
    
    
import plotly.graph_objects as go
from plotly.offline import plot
import time
    
# Taken from Lesson 19
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path, graph_save_directory=None):
    """Train, keep the checkpoint with the lowest validation loss, and plot the losses."""
    training_losses = []
    validation_losses = []
    training_accs = []
    validation_accs = []
    training_off_1 = []
    validation_off_1 = []
    best_validation_loss = float('inf')
    best_validation_epoch = -1
    training_start = time.time()
    records = len(train_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_off_1 = 0
        num_batches = 0
        for images, angles, labels in train_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            outputs_class = torch.argmax(outputs, dim=1)
            optimizer.zero_grad() 
            if isinstance(criterion, torch.nn.MSELoss): loss = criterion(outputs, labels.unsqueeze(1))
            else: loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            
            epoch_loss += loss.item() * len(images)
            if isinstance(criterion, torch.nn.MSELoss): 
                epoch_correct += (labels.unsqueeze(1) == outputs.round()).sum().item()
                epoch_off_1 += (abs(labels.unsqueeze(1) - outputs.round()) <= 1).sum().item()
            else: 
                epoch_correct += (labels == outputs_class).sum().item()
                epoch_off_1 += (abs(labels - outputs_class) <= 1).sum().item()
            # print(f'{epoch_correct:.1f} correct predictions during current epoch')
            num_batches += 1

        avg_loss = epoch_loss / records
        train_acc = epoch_correct / records
        train_off_1 = epoch_off_1 / records
        training_losses.append(avg_loss)
        training_accs.append(train_acc)
        training_off_1.append(train_off_1)

        # Compute validation loss. Don't need gradients here.
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_loss = 0
            val_off_1 = 0
            for images, angles, labels in val_loader: 
                images = images.to(device)
                angles = angles.to(device)
                labels = labels.to(device)
                val_outputs = model(images, angles)

                if isinstance(criterion, torch.nn.MSELoss): val_loss += criterion(val_outputs, labels.unsqueeze(1)).item() * len(images)
                else: val_loss += criterion(val_outputs, labels).item() * len(images)
                val_outputs_class = torch.argmax(val_outputs, dim=1)
                if isinstance(criterion, torch.nn.MSELoss): 
                    val_correct += (labels.unsqueeze(1) == val_outputs.round()).sum().item()
                    val_off_1 += (abs(labels.unsqueeze(1) - val_outputs.round()) <= 1).sum().item()
                else: 
                    val_correct += (labels == val_outputs_class).sum().item()
                    val_off_1 += (abs(labels - val_outputs_class) <= 1).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            validation_losses.append(val_loss)
            val_acc = val_correct / len(val_loader.dataset)
            validation_accs.append(val_acc)
            val_off_1 = val_off_1 / len(val_loader.dataset)
            validation_off_1.append(val_off_1)
            
            

        # Save the model if this is the best validation loss so far
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_validation_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            
            best_marker = ' *'
        else:
            best_marker = ''

        # if epoch % (num_epochs / 10) == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.5f}, Val Loss: {val_loss:.5f}{best_marker}, Train Acc: {train_acc:.5f}, Validation Acc: {val_acc:.5f}')

    torch.save(best_model, checkpoint_path)
    total_time = time.time() - training_start
    print(f'Average time per epoch: {total_time / num_epochs:.2f}s')
    print(f'Best validation loss: {best_validation_loss:.5f} (epoch {best_validation_epoch})')

    # Loss plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(len(training_losses))), y=training_losses, mode='lines', name='Training Loss'))
    fig1.add_trace(go.Scatter(x=list(range(len(validation_losses))), y=validation_losses, mode='lines', name='Validation Loss'))
    fig1.update_layout(title="Training and Validation Loss Over Time", xaxis_title="Epoch", yaxis_title="Loss")
    if graph_save_directory is None:
        fig1.show()
    else:
        plot(fig1, filename=os.path.join(graph_save_directory, 'Loss.html'))
    
    # Accuracy plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(training_accs))), y=training_accs, mode='lines', name='Training Accuracy'))
    fig2.add_trace(go.Scatter(x=list(range(len(validation_accs))), y=validation_accs, mode='lines', name='Validation Accuracy'))
    fig2.update_layout(title="Training and Validation Accuracy Over Time", xaxis_title="Epoch", yaxis_title="Accuracy")
    if graph_save_directory is None:
        fig2.show()
    else:
        plot(fig2, filename=os.path.join(graph_save_directory, 'Accuracy.html'))

    # Off-by-one accuracy plot
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(range(len(training_off_1))), y=training_off_1, mode='lines', name='Training Off-by-One Accuracy'))
    fig3.add_trace(go.Scatter(x=list(range(len(validation_off_1))), y=validation_off_1, mode='lines', name='Validation Off-by-One Accuracy'))
    fig3.update_layout(title="Training and Validation Off-by-One Accuracy Over Time", xaxis_title="Epoch", yaxis_title="Off-by-One Accuracy")
    if graph_save_directory is None:
        fig3.show()
    else:
        plot(fig3, filename=os.path.join(graph_save_directory, 'Off_by_One_Accuracy.html'))

def test_best(test_loader, best_filename):
    pass