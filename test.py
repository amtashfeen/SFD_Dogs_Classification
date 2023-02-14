import os
import cv2
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from matplotlib import pyplot as plt
from PIL import Image
from skimage import img_as_ubyte
from torchsummary import summary
from sklearn import metrics
import matplotlib

# Cuda as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initial path to the stanford dogs dataset folder
path = '/home/abdul/Desktop/projects/Blockchain'
save_path = '/home/abdul/Desktop/projects/Blockchain/saved_model.pth'

test_list = []
test_label = []

# Loading adresses of the data in lists
data1 = loadmat(path + '/lists/test_list.mat')
for i in range(len(data1['file_list'])):
    test_list.append('./images/Images/'+str(data1['file_list'][i][0][0]))
    test_label.append(data1['labels'][i][0]-1)

# Test data transform
test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data loading
class DS(Dataset):

    def __init__(self, directory, labels, transform=test_transform):
        self.dir = directory
        self.labels = labels
        self.transform = transform
        self.image_files_list = []

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):

        img = cv2.imread(self.dir[idx])
        img = Image.fromarray(img)
        img = self.transform(img)

        target = self.labels[idx]

        return img, target

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()

        self.n_classes = 120
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.n_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                            nn.Linear(self.n_inputs, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(1024, self.n_classes),)

    def forward(self, x):
        out = self.resnet(x)
        return out

# Accuracy metric
def get_accuracy(y_true, y_prob):
    _,pred = torch.max(y_prob, dim=1)
    return torch.sum(pred==y_true).item()

batch_size = 32

testset = DS(test_list, test_label, transform = test_transform)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=8)

# Model Testing
model = network()
model = model.to(device)

# Loading Model
model.load_state_dict(torch.load(save_path))

batch_loss = 0
total_t=0
correct_t=0
test_accuracy = []

with torch.no_grad():
    epoch_test_loss = 0
    epoch_test_accuracy = 0
    model.eval()

    for img, label in test_loader:
        img = img.float()
        img = img.cuda()
        label = label.long()
        label = label.cuda()

        test_out = model(img)

        _,pred_t = torch.max(test_out, dim=1)
        correct_t += torch.sum(pred_t==label).item()
        total_t += label.size(0)
    test_epoch_acc = 100 * (correct_t / total_t)  
    test_accuracy.append(test_epoch_acc)

    
    print('Test , Test_accuracy : {}'.format(test_epoch_acc),"\n")