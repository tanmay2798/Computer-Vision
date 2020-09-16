import numpy as np
import cv2
import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils, datasets

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

cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
ret, frame = cap.read()

h, w = frame.shape[:2]
c1 = w/2 - h/2
c2 = w/2 + h/2
frame_sq = frame[:,c1:c2,:]
frame1=frame_sq.copy()
down = int (math.log((h/50),2))
netPath = "/Users/tanmaygoyal/Downloads/vision4/model/gesture_net_cluttered_5.pth"
net = Net()
net.load_state_dict(torch.load(netPath))
loader = transforms.Compose([transforms.ToTensor()])
label_names = ['Next', 'Other', 'Previous', 'Stop']
while(cap.isOpened()):
	
    for i in range(5):
        ret, frame = cap.read()
        
    if ret==True:
        frame_sq = frame[:,c1:c2,:]
        #frame_sq = cv2.flip(frame_sq, 1)
        #frame_sq = cv2.absdiff(frame1,frame_sq)
        frame_small = []
        for j in range(down):
            frame_small = cv2.pyrDown(frame_sq)
        frame_50 = cv2.resize(frame_sq, (50, 50))
        frame_50 = loader(frame_50).float()
        inputs = frame_50.unsqueeze(0)
        with torch.no_grad():
        	outputs = net(inputs)
        	_, predicted = torch.max(outputs.data, 1)
        	print label_names[predicted]
        frame_sq = cv2.putText(frame_sq , label_names[predicted], (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame_sq)
        print outputs
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

