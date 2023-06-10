#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
torchvision.__version__
torch.__version__


# In[ ]:


_exp_name="sample"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import gc
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
#cross validation
from sklearn.model_selection import KFold
# This is for the progress bar.
from tqdm.auto import tqdm
import random
    
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.ElasticTransform(),
    transforms.RandomRotation(degrees=180),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
            
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im,label

class Wide_resnet50_2(nn.Module):
    def __init__(self) -> None:
        super(Wide_resnet50_2,self).__init__()
        self.model=torchvision.models.wide_resnet50_2(weights=None)
        self.model.conv1=nn.Conv2d(3,64,3,1,1)
        self.model.fc=nn.Linear(2048,11)
    
    def forward(self,x):
        return self.model(x)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

#TPU
# device = xm.xla_device()

# The number of batch size.
batch_size = 40

# Initialize a model, and put it on the device specified.
model_1 = Wide_resnet50_2()
model_2=torchvision.models.alexnet(weights=None,num_classes=11)
model_3=torchvision.models.vgg16_bn(weights=None,num_classes=11)
# sample_best='/kaggle/usr/lib/2023mlspringhw_3/sample_best.ckpt'
# model.load_state_dict(torch.load(sample_best,map_location=device))
model_1.load_state_dict(torch.load('/kaggle/input/ensemble-for-ntu-hw3-cnn/wide_resnet50_2_best.ckpt',map_location=device))
model_2.load_state_dict(torch.load('/kaggle/input/ensemble-for-ntu-hw3-cnn/AlexNet_best.ckpt',map_location=device))
model_3.load_state_dict(torch.load('/kaggle/input/ensemble-for-ntu-hw3-cnn/VGG16_bn_best.ckpt',map_location=device))

class Ensemble(nn.Module):
    def __init__(self) -> None:
        super(Ensemble,self).__init__()
        self.model1 = model_1
        self.model2 = model_2
        self.model3 = model_3
        
        # self.model1 = torch.load("wide_resnet50_2")
        # self.model2 = torch.load("AlexNet")
        # self.model3 = torch.load("VGG16_bn")
        path = '/kaggle/input/ensemble-for-ntu-hw3-cnn/'
        self.model1.load_state_dict(torch.load(f'{path}wide_resnet50_2_best.ckpt', map_location=device))
        self.model2.load_state_dict(torch.load(f'{path}AlexNet_best.ckpt', map_location=device))
        self.model3.load_state_dict(torch.load(f'{path}VGG16_bn_best.ckpt', map_location=device))

        self.fc=nn.Linear(11,11)
        
    def forward(self,x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        return (self.fc(out1+out2+out3))
        # return (out1+out2+out3)/3

# The number of training epochs.
n_epochs = 7

# If no improvement in 'patience' epochs, early stop.
patience = 5

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset("/kaggle/input/ml2023spring-hw3/train", tfm=train_tfm)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset("/kaggle/input/ml2023spring-hw3/valid", tfm=test_tfm)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
dataset = ConcatDataset([train_set, valid_set])
k = 3
splits = KFold(n_splits=k, shuffle=True, random_state=30)


# In[ ]:


def train_and_valid(model, model_name, train_loader, valid_loader, epochs = n_epochs):
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-6)
    
    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0
    for epoch in range(epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()
            # xm.mark_step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train {model_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        del train_loss, train_acc
        
        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid {model_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid {model_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid {model_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch+1}, saving model")
            torch.save(model.state_dict(), f"{model_name}_best.ckpt") # only save best to prevent output memory exceed error
            # xm.save(model.state_dict(), f"{model_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
                
        del valid_loss, valid_acc
        gc.collect()
        


# Train and Cross Validation

# In[ ]:


def train():
    models = [model_1, model_2, model_3]
    models_name = ['wide_resnet50_2', 'AlexNet', "VGG16_bn", _exp_name]
    train_epochs = [7, 10, 8, 5]
    for i in range(len(models)+1):
        for fold, (train_idx, valid_idx) in enumerate(splits.split(dataset)):
                # train_and_valid(model=xmp.MpModelWrapper(models[i]).to(device),model_name=models_name[i])
                print(f"FOLD:{fold+1}")
                
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)
                train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
                valid_loader = DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler)
                
                if i==3:
                    Ensemble().to(device)
                    Ensemble.load_state_dict(torch.load('/kaggle/input/ensemble-for-ntu-hw3-cnn/sample_best.ckpt'), map_location=device)
                
                train_and_valid(model = models[i].to(device),
                                model_name = models_name[i],
                                train_loader = train_loader,
                                valid_loader = valid_loader,
                                epochs = train_epochs[i])
            
# train()


# In[ ]:


model=Ensemble().to(device)
model.load_state_dict(torch.load('/kaggle/usr/lib/2023mlspringhw_3/sample_best.ckpt', map_location = device))

for fold, (train_idx, valid_idx) in enumerate(splits.split(dataset)):
    print(f'FOLD {fold+1}')
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
    valid_loader = DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler)

    train_and_valid(model = model,
                    model_name = _exp_name,
                    train_loader = train_loader,
                    valid_loader = valid_loader,
                    epochs = 7
                   )

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("/kaggle/input/ml2023spring-hw3/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = model.to(device)
# model_best.load_state_dict(torch.load('/kaggle/input/ensemble-for-ntu-hw3-cnn/sample_best.ckpt'))
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)

