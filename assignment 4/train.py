import os
import torch
import cv2 as cv
from PIL import Image
from skimage import io
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)        # 162
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 8, 7)        # 2646
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(8, 4, 3)        # 405
        self.fc1 = nn.Linear(4* 2 * 2, 20)    # 1600
        self.fc2 = nn.Linear(20, 4)            # 80

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = x.view(-1, 4 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

validation = False
data_transform = transforms.Compose([transforms.ToTensor()])
gesture_dataset = datasets.ImageFolder(root="/Users/tanmaygoyal/Downloads/vision4/test_copy_copy", transform=data_transform)
training_dataset_loader = torch.utils.data.DataLoader(gesture_dataset, batch_size=100, shuffle=True, num_workers=4)
test_set = datasets.ImageFolder(root="/Users/tanmaygoyal/Downloads/vision4/test_data_copy", transform=data_transform)
test_dataset_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=True, num_workers=4)

label_names = ["Other","Next","Previous", "Stop"]

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''for epoch in range(40):
    print epoch
    running_loss = 0.0
    for i, data in enumerate(training_dataset_loader, 0):
        inputs, labels = data
        #print(data)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print "[", (epoch + 1) , ", ", (i+1), "] loss: ", (running_loss / 9.0)
            running_loss = 0.0

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)'''
PATH = './cifar_net.pth'
dataiter = iter(test_dataset_loader)
images, labels = dataiter.next()

imshow(utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % label_names[labels[j]] for j in range(500)))
net = Net()
netPath = "/Users/tanmaygoyal/Downloads/vision4/model/gesture_net_cluttered_5.pth"
net.load_state_dict(torch.load(netPath))
outputs = net(images)
print(outputs)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % label_names[predicted[j]]
                              for j in range(500)))

for i in range(500):
    if('%5s' % label_names[labels[i]] != '%5s' % label_names[predicted[i]]):
        print('%5s' % label_names[labels[i]] + "  "+'%5s' % label_names[predicted[i]])

correct = 0
total = 0
with torch.no_grad():
    for data in test_dataset_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
