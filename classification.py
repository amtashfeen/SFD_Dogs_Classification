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

#Initial path to the stanford dogs dataset folder
path = '/home/abdul/Desktop/projects/Blockchain'

train_list = []
test_list = []
train_label = []
test_label = []

# Loading adresses of the data in lists
data = loadmat(path + '/lists/train_list.mat')
for i in range(len(data['file_list'])):
    train_list.append('./images/Images/'+str(data['file_list'][i][0][0]))
    train_label.append(data['labels'][i][0]-1)

data1 = loadmat(path + '/lists/test_list.mat')
for i in range(len(data1['file_list'])):
    test_list.append('./images/Images/'+str(data1['file_list'][i][0][0]))
    test_label.append(data1['labels'][i][0]-1)

# Train data transform (Preprocessing)
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Test data transform
test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data loading
class DS(Dataset):

    def __init__(self, directory, labels, transform=transform):
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

batch_size = 32
epochs = 100

trainset = DS(train_list, train_label)
testset = DS(test_list, test_label, transform = test_transform)

train_loader = DataLoader(dataset=trainset,shuffle=True, batch_size=batch_size, num_workers=8)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=8)

# Classification model (Based on ResNet, pretrained on ImageNet dataset)
#Output size = (32, 120)
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

# Loading model
model = network()
model = model.to(device)

#Adam optimizer with learning rate of 1x10e-4
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

train_loss = []
train_acc = []
test_loss = []
test_accuracy = []

# Model training
for i in range(epochs):
    running_loss = 0
    epoch_accuracy = 0
    correct_tensor = 0
    total = 0
    train_ac = 0
    total_step = len(train_loader)
    model.train()
    for it,(img,label) in enumerate(train_loader):
        img = img.float()
        label = label.long()
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        loss = criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(out, dim=1)

        correct_tensor += torch.sum(pred==label).item()
        total += label.size(0)
    epoch_accuracy = 100 * (correct_tensor / total)
    epoch_loss = running_loss/total_step
    train_acc.append(epoch_accuracy)
    train_loss.append(epoch_loss)
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(i, epoch_accuracy,epoch_loss))

    batch_loss = 0
    total_t=0
    correct_t=0

    #Model Testing
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
            t_loss = criterion(test_out, label)
            batch_loss += t_loss.item()

            _,pred_t = torch.max(test_out, dim=1)
            correct_t += torch.sum(pred_t==label).item()
            total_t += label.size(0)
        test_epoch_acc = 100 * (correct_t / total_t)  
        test_accuracy.append(test_epoch_acc)

        test_epoch_loss = batch_loss/len(test_loader)
        test_loss.append(test_epoch_loss)
        
        print('Test , Test_accuracy : {}, Test_loss : {}'.format(test_epoch_acc,test_epoch_loss),"\n")

#Plotting the loss graph
plt.plot(train_loss,'r', label='Training loss')
plt.plot(test_loss, 'b', label='Test loss')
plt.title('Loss vs Epochs')
plt.legend(loc=0)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('loss_graph.png')

#Plotting accuracy graph
plt.figure()
plt.plot(train_acc,'r', label='Training accuracy')
plt.plot(test_accuracy, 'b', label='Test accuracy')
plt.title('ACC vs Epochs')
plt.legend(loc=0)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('accuracy_graph.png')
# plt.show()
