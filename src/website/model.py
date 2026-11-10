import torch, torchvision
import numpy as np
import os
from PIL import Image


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
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2049, 23))
    
    def forward(self, X, angles):
        r = self.resnet(X)
        f = torch.flatten(r, 1)
        print(f.size())
        f = torch.cat([f, angles], dim=1)
        y = self.classifier(f)
        return y

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Load pretrained ResNet model
resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model_path = "trained_models/resnet_finetuned.pth"
model = ResNetTransfer(resnet50)
print("loading model")
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
model.eval()
print("model loaded!")

def collect_data(filename, angle, preprocess: torchvision.transforms.Compose):
    images, angles = [], []
    img = Image.open(filename).convert('RGB')
    images.append(preprocess(img).numpy())
    angles.append([angle])
    images = np.stack(images)
    angles = np.stack(angles)
    return torch.from_numpy(images), torch.from_numpy(angles)

def pass_image(filename, angle):
    image_data, angle_data = collect_data(filename, angle, preprocess)
    print("after collect_data")
    print(f"{image_data.size()}, {angle_data.size()}")
    output = model(image_data, angle_data)
    return output

o = pass_image("temp/temp_climb_png.png", 0)
print(o)
    