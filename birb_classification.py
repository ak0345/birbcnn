# Initialising Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchsummary
import torchmetrics

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(device)

"""# Formatting images"""

torch.manual_seed(42)

transformrot = torchvision.transforms.RandomRotation((0,60))
transformflip = torchvision.transforms.RandomHorizontalFlip(p = 0.3)
transformresize = torchvision.transforms.Resize((32,32))
transform = torchvision.transforms.Compose([transformrot, transformflip, transformresize, torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(r'C:\Users\foosh\Desktop\projects\train', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

"""# Defining Class for Classifier Model"""

class conv(nn.Module):
    def __init__(self):
        super().__init__()
        #defining layers
        self.conv1 = torch.nn.Conv2d(3, 64, 5)
        self.batch1 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(64, 128, 5)
        self.batch2 = torch.nn.BatchNorm2d(128)
        self.flat = torch.nn.Flatten()
        self.hc1 = torch.nn.Linear(3200,2048)
        self.hc2 = torch.nn.Linear(2048,1024)
        self.out = torch.nn.Linear(1024, 525)

    def forward(self, x):
        #defining model
        x = self.pool(self.batch1(F.relu(self.conv1(x))))
        x = self.pool(self.batch2(F.relu(self.conv2(x))))
        x = torch.relu(self.hc1(self.flat(x)))
        x = torch.tanh(self.hc2(x))
        x = F.softmax(self.out((x)), dim = 1)
        return x

#outputting model summary
testm = conv()
testm.to(device)
torchsummary.summary(testm, input_size=(3, 32, 32), device = device)

# Training Classifier Model

torch.manual_seed(42)

lr = 1e-4 #1e-4
epochs = 80

#assiging model to variable
model = conv()
model.load_state_dict(torch.load(r'C:\Users\foosh\Desktop\projects\model.pth'))
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr = lr)
accs = torchmetrics.Accuracy(task="multiclass", num_classes=525)

for e in range(epochs):
    print(f' Epoch: {e+1}\n ')
    losseslist = []
    acclist = []
    for images, labels in dataloader:
        #sending varaibles to gpu
        images = images.to(device)
        labels = labels.to(device)

        #training model
        outputs = model(images)
        loss = criterion(outputs, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Calculating accuracy
        outputs2 = outputs.to(cpu).clone().detach()
        labels2 = labels.to(cpu).clone().detach()
        losseslist.append(loss.item())
        acclist.append(accs(outputs2, labels2).numpy())

    #printing loss of each batch and average accuracy in each epoch
    print(f" Loss: {losseslist}\n Accuracy: {np.mean(np.array(acclist))*100}%\n")

    torch.save(model.state_dict(), r'C:\Users\foosh\Desktop\projects\model.pth')
    print('saved epoch')

# Testing Classifier Model

torch.manual_seed(42)

transformrot = torchvision.transforms.RandomRotation((0,30))
transformflip = torchvision.transforms.RandomHorizontalFlip(p = 0.4)
transformresize = torchvision.transforms.Resize((32,32))
transform = torchvision.transforms.Compose([transformrot, transformflip, transformresize, torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(r'C:\Users\foosh\Desktop\projects\test', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
accs = torchmetrics.Accuracy(task="multiclass", num_classes=525)

for images, labels in dataloader:
     #running model on test data
     images = images.to(device)
     labels = labels.to(device)
     outputs = model(images).to(device)
     outputs2 = outputs.to(cpu).detach()
     labels = labels.to(cpu).clone().detach()

     #printing accuracy of model on test batch
     print(f'Accuracy: {accs(outputs2,labels)*100}%')

torch.save(model.state_dict(), r'C:\Users\foosh\Desktop\projects\model.pth')