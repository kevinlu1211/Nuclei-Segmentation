
# coding: utf-8

# In[1]:

import pickle
import random
from glob import glob

import cv2
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

# get_ipython().magic('matplotlib inline')


# In[2]:

TRAIN_PATH = 'data/stage1_train'
TEST_PATH = 'data/stage1_test'


# In[3]:

image_folders = glob(f"{TRAIN_PATH}/*")


# In[4]:

sample_folder = random.choice(image_folders)


# In[5]:

def get_masks(folder_path):
    masks_paths = glob(f"{folder_path}/masks/*") 
    return [mpimg.imread(p) for p in masks_paths]

def get_image(folder_path): 
    image_path, = glob(f"{folder_path}/images/*")
    return mpimg.imread(image_path)

def stack_masks(masks):
    stacked_mask = np.zeros_like(masks[0])
    for mask in masks:
        stacked_mask += mask
    stacked_mask = stacked_mask > 0
    return stacked_mask.astype(np.uint8)


# ## Test on one folder

# In[6]:

# image = get_image(sample_folder)
# masks = get_masks(sample_folder)
# stacked_mask = stack_masks(masks)


# In[7]:

# plt.imshow(image)


# In[8]:

# plt.imshow(stack_masks(masks), cmap='gray')


# ## Now create preprocess training data

# In[9]:

# all_images = []
# for path in tqdm(image_folders):
#     all_images.append(get_image(path))
# all_images = np.array(all_images)
#
#
# # In[10]:
#
# all_masks = []
# for path in tqdm(image_folders):
#     all_masks.append(get_masks(path))
#
#
# # In[11]:
#
# all_stacked_masks = []
# for masks in tqdm(all_masks):
#     all_stacked_masks.append(stack_masks(masks))
# all_stacked_masks = np.array(all_stacked_masks)


# In[12]:

# pickle.dump(all_images, open("data/training_images.p", "wb"))


# In[13]:

# pickle.dump(all_stacked_masks, open("data/stacked_masks.p", "wb"))


# ## Create the dataset

# In[14]:

all_images = pickle.load(open("data/training_images.p", "rb"))


# In[15]:

all_stacked_masks = pickle.load(open("data/stacked_masks.p", "rb"))


# In[16]:

from sklearn.model_selection import KFold


# In[17]:

kf = KFold(n_splits=8, shuffle=True)


# In[18]:

for train_idx, test_idx in kf.split(all_images):
    train_images, val_images = all_images[train_idx], all_images[test_idx]
    train_masks, val_masks = all_stacked_masks[train_idx], all_stacked_masks[test_idx]
    break


# In[19]:

def convert_image_to_tensor(image):
    return torch.from_numpy(image).type(torch.FloatTensor)

def resize_image(image):
    try:
        resized_image = cv2.resize(image, (128, 128))
    except cv2.error:
        print("how?")
    return cv2.resize(image, (128, 128))

def convert_to_CHW(tensor):
    return tensor.permute(2, 0, 1)

class NucleiDataset(data.Dataset):
    def __init__(self, input_image, stacked_masks, is_train):
        self.X = input_image
        self.y = stacked_masks
        self.input_transform = transforms.Compose([
            resize_image,
            convert_image_to_tensor, 
            convert_to_CHW,
        ])
        self.mask_transform = transforms.Compose([
            resize_image, 
            convert_image_to_tensor
        ])
        self.is_train = is_train
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
         return {
             "input": self.input_transform(self.X[i]),
             "mask": self.mask_transform(self.y[i])
         }


# In[20]:

BATCH_SIZE = 16
train_ds = NucleiDataset(train_images, train_masks, is_train=True)
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ds = NucleiDataset(val_images, val_masks, is_train=False)
val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ## Check if the input image and associated mask is correct after transforms

# In[21]:

sample_data = random.choice(train_ds)


# In[22]:

sample_input = sample_data['input'].permute(1, 2, 0)
sample_mask = sample_data['mask']


# In[23]:

# plt.imshow(sample_input)


# In[24]:

# plt.imshow(sample_mask)


# ## Now we define the U-Net architecture

# ![title](U-Net.png)

# In[42]:

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_bn_relu_1 = ConvBnRelu(in_channels, in_channels)
        self.conv_bn_relu_2 = ConvBnRelu(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x_down_sampled = self.pool(x)
        return x_down_sampled, x



class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)        
        self.conv_bn_relu_1 = ConvBnRelu(in_channels, out_channels)
        self.conv_bn_relu_2 = ConvBnRelu(out_channels, out_channels)
    
    def forward(self, x, encoded):
        x = self.conv_transpose(x)
        x = torch.cat([x, encoded], 1)
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, n_input_channels, n_classes):
        super().__init__()
        self.down_sample_1 = DownSample(n_input_channels, 64)
        self.down_sample_2 = DownSample(64, 128)
        self.down_sample_3 = DownSample(128, 256)
        self.down_sample_4 = DownSample(256, 512)
        self.middle = nn.Sequential(
            nn.Conv2d(512, 1024, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(1024)
        )
        self.up_sample_1 = UpSample(1024, 512)
        self.up_sample_2 = UpSample(512, 256)
        self.up_sample_3 = UpSample(256, 128)
        self.up_sample_4 = UpSample(128, 64)
        self.output = nn.Sequential(
            nn.Conv2d(64, n_classes, padding=1, kernel_size=3, stride=1),
        )
    
    def forward(self, x):
        x, x_1 = self.down_sample_1(x)
        x, x_2 = self.down_sample_2(x)
        x, x_3 = self.down_sample_3(x)
        x, x_4 = self.down_sample_4(x)
        x = self.middle(x)
        x = self.up_sample_1(x, x_4)
        x = self.up_sample_2(x, x_3)
        x = self.up_sample_3(x, x_2)
        x = self.up_sample_4(x, x_1)
        x = self.output(x)
        return x


# In[43]:

# next(iter(train_dl))


# In[44]:



def calculate_thresholded_precision(pred, target, thresholds):
    batch_size = pred.size()[0]
    thresholded_precisions = [calculate_IOU(pred, target, t / 100.0) for t in thresholds]
    avg_precision_per_sample = torch.cat(thresholded_precisions, 0).view(len(thresholded_precisions),
                                                                         batch_size).permute(1, 0).mean(1)
    avg_precision = avg_precision_per_sample.mean()
    return avg_precision

def calculate_IOU(pred, target, threshold):
    batch_size = pred.size()[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    tp = torch.sum(((pred_flat >= threshold) * (target_flat == 1)).float(), 1)
    fp = torch.sum(((pred_flat >= threshold) * (target_flat == 0)).float(), 1)
    fn = torch.sum(((pred_flat < threshold) * (target_flat == 1)).float(), 1)
    precision = tp.float()/(tp + fp + fn).float()
    return precision

def create_thresholded_mask(pred, thresholds):
    return {str(t / 100.0): (pred >= t/100.0).float().squeeze(1).data.numpy() for t in thresholds}

def cudarize(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

from model.unet import UNet
NUM_EPOCHS = 4
thresholds = range(50, 100, 5)
u_net = cudarize(UNet(n_input_channels=4, n_classes=1))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(u_net.parameters())



for i in range(NUM_EPOCHS):
    train_masks = []
    train_precisions = []
    train_losses = []
    for train_idx, d in enumerate(tqdm(train_dl)):
        optimizer.zero_grad()
        u_net.train()
        batch_size = len(d['input'])
        inp = cudarize(Variable(d['input']))
        pred = F.sigmoid(u_net(inp))
        target = cudarize(Variable(d['mask'])).unsqueeze(1) # to keep same dims as pred
        precision = calculate_thresholded_precision(pred.cpu(), target.cpu(), thresholds)
        masks = create_thresholded_mask(pred.cpu(), thresholds)
        # train_masks.append(masks)
        loss = criterion(pred, target)
        train_precisions.append(precision.data.numpy())
        train_losses.append(loss.cpu().data.numpy())
        loss.backward()
        optimizer.step()
        tqdm.write(f"Precision: {round(precision.data[0], 3)} Loss: {round(loss.data[0], 3)}")
        # print(f"Average precision for training batch {train_idx + 1}/{len(train_dl)} is: {round(precision.data[0], 5)}")
        # print(f"Average loss for training batch {train_idx + 1}/{len(train_dl)} is: {round(loss.data[0], 5)}")


    val_masks = []
    val_precisions = []
    val_losses = []
    for val_idx, d in enumerate(tqdm(val_dl)):
        u_net.eval()
        batch_size = len(d['input'])
        inp = cudarize(Variable(d['input']))
        pred = F.sigmoid(u_net(inp))
        target = cudarize(Variable(d['mask'])).unsqueeze(1)  # to keep same dims as pred
        precision = calculate_thresholded_precision(pred.cpu(), target.cpu(), thresholds)
        masks = create_thresholded_mask(pred.cpu(), thresholds)
        # val_masks.append(masks)
        loss = criterion(pred, target)
        val_precisions.append(precision.data.numpy())
        val_losses.append(loss.cpu().data.numpy())
        tqdm.write(f"Precision: {round(precision.data[0], 3)} Loss: {round(loss.data[0], 3)}")


    torch.save(u_net.state_dict(), f"model_checkpoints/epoch_{i+1}.loss_{round(np.mean(val_losses), 2)}.precision_{round(np.mean(val_precisions), 2)}")