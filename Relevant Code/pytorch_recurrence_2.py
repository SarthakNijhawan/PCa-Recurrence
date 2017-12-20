import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
#Hyper-parameters
learning_rate=0.0001
num_epochs = 30
batch_size=100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Data Loader (Input Pipeline)
data_train = ImageFolder(root='train', transform=transform)
train_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)

data_val = ImageFolder(root='valid', transform=transform)

val_loader = torch.utils.data.DataLoader(dataset=data_val,
                                          batch_size=batch_size, 
                                          shuffle=False)

#CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=11, padding=5),
            # nn.Dropout2d(p=0.25),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=9, padding=4),
            # nn.Dropout2d(p=0.4),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(20, 30, kernel_size=5, padding=2),
        #     nn.Dropout2d(p=0.55),
        #     nn.BatchNorm2d(30),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        
        # self.fc1 = nn.Sequential(
        # 	nn.Linear(4320, 128),
        # 	nn.ReLU())
        # self.drop1 = nn.Dropout(p=0.6)
        # self.fc2 = nn.Sequential(
        # 	nn.Linear(128, 32),
        # 	nn.ReLU())
        # self.drop2 = nn.Dropout(p=0.6)
        self.fc_final = nn.Linear(12500, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # print out.size()
        # out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.drop1(out)
        # out = self.fc2(out)
        # out = self.drop2(out)
        # print out.size()
        out = out.view(out.size(0), -1)
        out = self.fc_final(out)
        return out
        
model = CNN()
model =model.cuda()

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the Model
for epoch in range(num_epochs):
    model.train()
    running_corrects=0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        outputs=outputs.data.max(1,keepdim=True)[1]
        # print outputs
        # print labels#.max(0,keepdim=True)[1]
        running_corrects += outputs.eq(labels.data.view_as(outputs)).sum()
        if (i+1) % 1000 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f r_c : %d' 
                   %(epoch+1, num_epochs, i+1, len(data_train)//batch_size, loss.data[0],running_corrects))

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = Variable(images.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Train accuraccy :%.4f val Accuracy : %.4f %%' % ((100.0 * running_corrects / len(data_train)),(100.0 * correct / total)))
    val_acc=100.0 * correct / total
    # Save the Trained Model
    modelsave_name='./weights/rec_pred'+'_'+str(val_acc)+'_'+'2_.pkl'
    torch.save(model.state_dict(), modelsave_name)


