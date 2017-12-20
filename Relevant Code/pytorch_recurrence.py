import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np
import torch.utils.data as data

from PIL import Image
import os
import os.path
import time
import shutil


#Hyper-parameters
learning_rate=0.01
num_epochs = 10
batch_size=100



transform = transforms.Compose([
    # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.25),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
])
transform_val = transforms.Compose([
    # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.25),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
])
####


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def convert_hsv(path):
    # return img.convert("HSV")
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('HSV'))
def convert_gray(path):
    # return img.convert('L')
    with open(path, 'rb') as f:
    	with Image.open(f) as img:
        	return np.array(np.expand_dims(img.convert('L'),axis=2))

class ImageFolder_custom(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mode,transform=None, target_transform=None,
                 loader=default_loader,gray=convert_gray,hsv=convert_hsv):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.gray = convert_gray
        self.hsv = convert_hsv
        # self.mode=mode
        if(mode=='train'):
            self.ProcessedDataPath = './ProcessedDataFolder/train'
        elif(mode=='val'):
            self.ProcessedDataPath = './ProcessedDataFolder/val'
        else:
            self.ProcessedDataPath = './ProcessedDataFolder/test'

    def __getitem__(self, index):
        #print('getting item, ',index)
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        ################################
        if os.path.exists(os.path.join(self.ProcessedDataPath,str(index))):
            img_stacked, target = torch.load(os.path.join(self.ProcessedDataPath,str(index)))
            img_stacked = torch.ByteTensor(img_stacked).float()/255.
            return img_stacked, target


        ###############################
        #print('index:',index)
        path, target = self.imgs[index]
        path_token = path.rsplit("/")
        # print path_token
        image_token = path_token[-1].rsplit("_")
        #print image_token
        path_e="../DB_E_101_anno_cent/"+path_token[1]+"/"+image_token[0]+"_"+path_token[1]+"/"+path_token[-1]
        path_h = "../DB_H_101_anno_cent/"+path_token[1]+"/"+image_token[0]+"_"+path_token[1]+"/"+path_token[-1]
        #img = self.loader(path)
        # img.save('./image.bmp', 'bmp')
        # print np.array(img).shape
        #print('reading img_e')
        img_e = self.loader(path_e)
        # print np.array(img_e).shape
        #print('reading img_h')
        img_h = self.loader(path_h)
        # print np.array(img_h).shape
        #print('reading img_hsv')
        img_hsv = self.hsv(path)
        # print img_hsv
        #img_hsv.save('./hsv_image.bmp', 'bmp')
        # print np.max(np.array(img_hsv),axis=0)
        #print('reading img_gray')
        img_gray = self.gray(path)

        # print np.array(np.expand_dims(img_gray,axis=2)).shape
        #print('concatinating')
        img_stacked = np.concatenate((img_hsv,img_h,img_e,img_gray),axis=2)
        # print np.array(img_stacked).shape
        # img_stacked =np.float32(np.transpose(img_stacked,(2,0,1)))
        if self.transform is not None:
            img_stacked = self.transform(img_stacked)
        if self.target_transform is not None:
            target = self.target_transform(target)


        with open(os.path.join(self.ProcessedDataPath,str(index)), 'wb') as f:
            DataSample = (255*img_stacked).byte().numpy(), target
            torch.save(DataSample, f)

        return img_stacked, target
        # return img, target

    def __len__(self):
        return len(self.imgs)


####
# Data Loader (Input Pipeline)
data_train = ImageFolder_custom(root='train', transform=transform,mode='train')
train_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=16)

data_val = ImageFolder_custom(root='valid', transform=transform_val,mode='val')

val_loader = torch.utils.data.DataLoader(dataset=data_val,
                                          batch_size=batch_size, 
                                          shuffle=False,num_workers=16)
'''
#CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(10, 30, kernel_size=11, padding=5),
            nn.Dropout2d(p=0.25),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 48, kernel_size=9, padding=4),
            nn.Dropout2d(p=0.25),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 70, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.55),
            nn.BatchNorm2d(70),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))
        
        self.fc1 = nn.Sequential(
            nn.Linear(10080, 128),
            nn.LeakyReLU(0.1))
        self.drop1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1))
        self.drop2 = nn.Dropout(p=0.6)
        self.fc_final = nn.Linear(32, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print out.size()
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        # print out.size()
        out = out.view(out.size(0), -1)
        out = self.fc_final(out)
        return out
        
model = CNN()
model =model.cuda()


'''
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(10,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3]*3*3, num_classes)
        self.softmax = nn.Softmax()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # print "conv1 done"
        out = self.layer1(out)
        # print "layer1 done"
        out = self.layer2(out)
        # print "layer2 done"
        out = self.layer3(out)
        # print "layer3 done"
        out = F.relu(self.bn1(out))
        # print "bn1 done"
        out = F.avg_pool2d(out, 8)
        # print "avg_pool2d done"
        # print out.size()
        out = out.view(out.size(0), -1)
        # print "out view done"
        # print out.size()
        out = self.softmax(self.linear(out))
        # print "final"

        return out

net=Wide_ResNet(10, 2, 0.6, 2)
model=net.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5.0, gamma=0.5)

def save_checkpoint(state, is_best, filename='./weights/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './weights/model_best.pth.tar')

val_acc_prev=0
# change accordingly to resume from the previous model
checkpointer = './weights/model_best.pth.tar'

if (~(checkpointer==None)):
    if os.path.isfile(checkpointer):
        print("=> loading checkpoint '{}'".format(checkpointer))
        checkpoint = torch.load(checkpointer)
        # args.start_epoch = checkpoint['epoch']
        val_acc_prev = checkpoint['val_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpointer, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpointer))


# Train the Model
for epoch in range(num_epochs):
    t1 = time.time()
    model.train()
    scheduler.step()
    running_corrects=0
    time_taken=0
    time_taken_val=0
    print "training"
    #t_batch=time.time()
    for i, (images, labels) in enumerate(train_loader):
        t2=time.time()
        #print(t2-t_batch)
        #t_batch=t2
        # print images 
        # print labels
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        # print outputs
        loss = criterion(outputs, labels)
        # print loss
        loss.backward()
        optimizer.step()
        outputs=outputs.data.max(1,keepdim=True)[1]
        # print outputs
        running_corrects += outputs.eq(labels.data.view_as(outputs)).sum()
        t3=time.time()
        time_taken+=(t3-t2)
        if (i+1) % 100 == 0:            
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f r_c : %d  Time_taken : %.4f' 
                   %(epoch+1, num_epochs, i+1, len(data_train)//batch_size, loss.data[0],running_corrects,time_taken/100.))
            time_taken=0

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    print "validating ..."
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        t4=time.time()
        images = Variable(images.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        t5 = time.time()
        time_taken_val+=(t5-t4)
        if (i+1) % 100 ==0:
            print('Epoch [%d/%d], Val Iter [%d/%d]  running corrects : %d  Time_taken %.4f' % (epoch+1, num_epochs,i+1, len(data_val)//batch_size,correct,time_taken_val/100.))
            time_taken_val=0
    t6=time.time()
    print('Train accuraccy :%.4f val Accuracy : %.4f Time_taken %.4f' % ((100.0 * running_corrects / len(data_train)),(100.0 * correct / total),t6-t1))
    val_acc=100.0 * correct / total
    is_best=val_acc>val_acc_prev
    val_acc_prev=val_acc
    # Save the Trained Model
    print "saving model..."
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'wrn_10_2_06',
        'state_dict': model.state_dict(),
        'val_acc': val_acc,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
    print "saved model"
    # modelsave_name='./weights/rec_pred'+'_'+str(val_acc)+'_'+'_multi_channel_wrn_10_2_06.pkl'
    # torch.save(model.state_dict(), modelsave_name)


