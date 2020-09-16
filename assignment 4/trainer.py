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

validation = False
data_transform = transforms.Compose([transforms.ToTensor()])
gesture_dataset = datasets.ImageFolder(root="/Users/tanmaygoyal/Downloads/vision4/test_data_copy_copy", transform=data_transform)
print gesture_dataset.class_to_idx
dataset_loader = torch.utils.data.DataLoader(gesture_dataset, batch_size=100, shuffle=True, num_workers=4)
if validation:
	testset = datasets.ImageFolder(root="/Users/tanmaygoyal/Downloads/vision4/test", transform=data_transform)
	testset_loader = torch.utils.data.DataLoader(gesture_dataset, num_workers=4)

print gesture_dataset.__len__()

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

lossGraph = os.path.join(os.getcwd(), "model/loss_cluttered_5.csv")
modelPath = os.path.join(os.getcwd(), "model/gesture_net_cluttered_5.pth")
fields = ["epoch", "i", "loss"]
loss_csv = []

for epoch in range(100):
    print epoch
    running_loss = 0.0
    for i, data in enumerate(dataset_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 49:
            print "[", (epoch + 1) , ", ", (i+1), "] loss: ", (running_loss / 49.0)
            loss_csv.append([epoch + 1, ((i + 1)*100/gesture_dataset.__len__()), running_loss / 49.0])
            running_loss = 0.0
            
    if validation:
		correct = 0
		total = 0
		tl = 0
		data_size = 0
		with torch.no_grad():
			for data in testset_loader:
				images, labels = data
				outputs = net(images)
				test_loss = criterion(outputs, labels)
				tl += test_loss.item()
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				data_size += 1
				correct += (predicted == labels).sum().item()

		print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
		print('Loss of the network on the test images: %.3f %%' % tl/data_size)

print('Finished Training')
torch.save(net.state_dict(), modelPath)
with open(lossGraph, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields) 
    csvwriter.writerows(loss_csv)

