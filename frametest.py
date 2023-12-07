import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import re

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def FeatureExtractor(frames_tensor):
    activation = {}
    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()
    vid = frames_tensor
    batch = preprocess(vid).unsqueeze(0).to(device)
 
    h2 = model.avgpool.register_forward_hook(getActivation('avgpool'))
    out = model(batch)

    feature_vector_tensor = activation['avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
    feature_vector_np = feature_vector_tensor.detach().cpu().numpy()

    h2.remove()
    return feature_vector_np

label_mapping = {
    "antiClockwise": 0,
    "blahblah": 1,
    "clockwise": 2,
    "come": 3,
    "down": 4,
    "like": 5,
    "NoActivities": 6,
    "stop": 7,
    "super": 8,
    "up": 9,
    "victory": 10
}

inverse_label_mapping = {v: k for k, v in label_mapping.items()}

def predictor(frames_tensor):
    feature_vector = FeatureExtractor(frames_tensor)
    tensor_features = torch.from_numpy(feature_vector)

    input_size = len(feature_vector[0])
    hidden_size = 64  
    num_classes = len(label_mapping)
    model = Classifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load('resnet3Dann.pth'))
    model.eval()

    with torch.no_grad():
        output = model(tensor_features)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = inverse_label_mapping[predicted.item()]
    return predicted_label, confidence.item()

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

frames_list = []
frame_count = 0
image_dir = "C:\\Users\\sazza\\OneDrive\\Documents\\newMay31\\frametest"
image_files = os.listdir(image_dir)
image_files.sort(key=numerical_sort)  # Sort the images to process them in correct sequence

for image_file in image_files:
    print("Loading frame: ", image_file)
    image_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(image_path)
    frames_list.append(frame)
    frame_count += 1

    if frame_count == 40:
        frames_array = np.array(frames_list)
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))
        frames_tensor = torch.from_numpy(frames_array)
        prediction, confidence = predictor(frames_tensor)
        print("Prediction: ", prediction)
        print("Confidence: ", confidence)
        print('\n')

        frames_list = []
        frame_count = 0
