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

class ResNetTransferRegressor2Layer(torch.nn.Module):
    def __init__(self, resnetmodel):
        super(ResNetTransferRegressor2Layer, self).__init__()
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
    
class ResNetTransferRegressor1Layer(torch.nn.Module):
    def __init__(self, resnetmodel):
        super(ResNetTransferRegressor1Layer, self).__init__()
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
        self.regressor = torch.nn.Sequential(torch.nn.Linear(2049, 1))
    
    def forward(self, X, angles):
        r = self.resnet(X)
        f = torch.flatten(r, 1)
        f = torch.cat([f, angles], dim=1)
        y = self.regressor(f)
        return y
    
class ResNetTransferClassifier1Layer(torch.nn.Module):
    def __init__(self, resnetmodel):
        super(ResNetTransferClassifier1Layer, self).__init__()
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
        f = torch.cat([f, angles], dim=1)
        y = self.classifier(f)
        return y

class ResNetTransferClassifier2Layer(torch.nn.Module):
    def __init__(self, resnetmodel):
        super(ResNetTransferClassifier2Layer, self).__init__()
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
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2049, 256),
                                              torch.nn.Linear(256, 23))
    
    def forward(self, X, angles):
        r = self.resnet(X)
        f = torch.flatten(r, 1)
        f = torch.cat([f, angles], dim=1)
        y = self.classifier(f)
        return y

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pretrained ResNet model
resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

# classifier 1 layer
model_path_classifier_1layer = "trained_models/resnet_finetuned_FULL2.pth"
model_classifier_1layer = ResNetTransferClassifier1Layer(resnet50)
model_classifier_1layer.load_state_dict(torch.load(model_path_classifier_1layer, weights_only=True, map_location=torch.device("cpu")))
model_classifier_1layer.eval()

# classifier trained on 100 climbs
model_path_classifier_100climbs = "trained_models/resnet_finetuned.pth"
model_classifier_100climbs = ResNetTransferClassifier1Layer(resnet50)
model_classifier_100climbs.load_state_dict(torch.load(model_path_classifier_100climbs, weights_only=True, map_location=torch.device("cpu")))
model_classifier_100climbs.eval()

# regressor 1 layer
model_path_regressor_1layer = "trained_models/resnet_finetuned_regressor_FULL.pth"
model_regressor_1layer = ResNetTransferRegressor1Layer(resnet50)
model_regressor_1layer.load_state_dict(torch.load(model_path_regressor_1layer, weights_only=True, map_location=torch.device("cpu")))
model_regressor_1layer.eval()

# classifier 2 layers
model_path_classifier_2layer = "trained_models/resnet_finetuned_double_FULL.pth"
model_classifier_2layer = ResNetTransferClassifier2Layer(resnet50)
model_classifier_2layer.load_state_dict(torch.load(model_path_classifier_2layer, weights_only=True, map_location=torch.device("cpu")))
model_classifier_2layer.eval()

# regressor 2 layers
model_path_regressor_2layer = "trained_models/resnet_finetuned_regressor_double_FULL.pth"
model_regressor_2layer = ResNetTransferRegressor2Layer(resnet50)
model_regressor_2layer.load_state_dict(torch.load(model_path_regressor_2layer, weights_only=True, map_location=torch.device("cpu")))
model_regressor_2layer.eval()

def collect_data(filename, angle, preprocess: torchvision.transforms.Compose):
    images, angles = [], []
    img = Image.open(filename).convert('RGB')
    images.append(preprocess(img).numpy())
    angles.append([angle])
    images = np.stack(images)
    angles = np.stack(angles)
    return torch.from_numpy(images), torch.from_numpy(angles)

def get_prediction(tensor, isClassifier):
    if isClassifier:
        index = int(torch.argmax(tensor, dim=1).item())
    else:
        index = int(tensor.round().item())
    return difficulties.get(index + 10)

def pass_image(filename, angle, model_type_string):
    image_data, angle_data = collect_data(filename, angle, preprocess)
    isClassifier = "classifier" in model_type_string
    if model_type_string == "classifier_100":
        output = model_classifier_100climbs(image_data, angle_data)
    elif model_type_string == "classifier1":
        output = model_classifier_1layer(image_data, angle_data)
    elif model_type_string == "regressor1":
        output = model_regressor_1layer(image_data, angle_data)
    elif model_type_string == "regressor2":
        output = model_regressor_2layer(image_data, angle_data)
    else:
        output = model_classifier_2layer(image_data, angle_data)
    print(f"prediction: {output}")
    return get_prediction(output, isClassifier)

# o = pass_image("temp/temp_climb_png.png", 0)
# print(o)
    