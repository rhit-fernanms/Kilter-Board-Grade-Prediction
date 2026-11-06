import torch

def test_model(model_obj, model_path, criterion, test_loader, device):
    model_obj.load_state_dict(torch.load(model_path, weights_only=True))
    
    model_obj.eval()
    with torch.no_grad():
        correct = 0
        off_1 = 0
        loss = 0
        for images, angles, labels in test_loader: 
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)
            outputs = model_obj(images, angles)

            if isinstance(criterion, torch.nn.MSELoss): loss += criterion(outputs, labels.unsqueeze(1)).item() * len(images)
            else: loss += criterion(outputs, labels).item() * len(images)
            outputs_class = torch.argmax(outputs, dim=1)

            if isinstance(criterion, torch.nn.MSELoss): 
                correct += (labels.unsqueeze(1) == outputs.round()).sum().item()
                off_1 += (abs(labels.unsqueeze(1) - outputs.round()) <= 1).sum().item()
            else: 
                correct += (labels == outputs_class).sum().item()
                off_1 += (abs(labels - outputs_class) <= 1).sum().item()

        loss = loss / len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
        off_1_acc = off_1 / len(test_loader.dataset)
        
        print(f'Loss: {loss}')
        print(f'Accuracy: {acc}')
        print(f'Off-by-One Accuracy: {off_1_acc}')