import torch, torchvision
import numpy as np
import os, copy
from PIL import Image

difficulties = {
  10: "4a",
  11: "4b",
  12: "4c",
  13: "5a",
  14: "5b",
  15: "5c",
  16: "6a",
  17: "6a+",
  18: "6b",
  19: "6b+",
  20: "6c",
  21: "6c+",
  22: "7a",
  23: "7a+",
  24: "7b",
  25: "7b+",
  26: "7c",
  27: "7c+",
  28: "8a",
  29: "8a+",
  30: "8b",
  31: "8b+",
  32: "8c",
  33: "8c+",
}

class ResNetTransfer(torch.nn.Module):
    def __init__(self, resnetmodel):
        super(ResNetTransfer, self).__init__()
        # ResNet structure (without classifier)
        self.resnet = torch.nn.Sequential(resnetmodel.conv1,
                                            resnetmodel.bn1,
                                            resnetmodel.relu,
                                            resnetmodel.maxpool,
                                            resnetmodel.layer1,
                                            resnetmodel.layer2,
                                            resnetmodel.layer3,
                                            resnetmodel.layer4,
                                            resnetmodel.avgpool)
        # 2048 from the resnet convolution and 1 from the angles
        self.regressor = torch.nn.Sequential(torch.nn.Linear(2049, 256),
                                             torch.nn.Linear(256, 1))
    
    def forward(self, X, angles):
        r = self.resnet(X)
        f = torch.flatten(r, 1)
        f = torch.cat([f, angles], dim=1)
        y = self.regressor(f)
        return y

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load trained model
resnet50 = torchvision.models.resnet50(weights=None)
model_path = "trained_models/resnet_finetuned_regressor_double_FULL.pth"
model = ResNetTransfer(copy.deepcopy(resnet50))
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
model.eval()

def collect_data(filename, angle, preprocess: torchvision.transforms.Compose):
    images, angles = [], []
    img = Image.open(filename).convert('RGB')
    images.append(preprocess(img).numpy())
    angles.append([angle])
    images = np.stack(images)
    angles = np.stack(angles)
    return torch.from_numpy(images), torch.from_numpy(angles)

def get_prediction(tensor):
    index = int(tensor.round().item())
    return difficulties.get(index + 10)

def pass_image(filename, angle):
    image_data, angle_data = collect_data(filename, angle, preprocess)
    with torch.no_grad():
        output = model(image_data, angle_data)
    print(f"prediction: {output}")
    return get_prediction(output)

# o = pass_image("temp/temp_climb_png.png", 0)
# print(o)
    