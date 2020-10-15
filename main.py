# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:56:41 2020

@author: Yun, Junhyuk
"""

from __future__ import print_function
import torch
from torch import nn, optim, cuda
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

# GPU settings
device = 'cuda' if cuda.is_available() else 'cpu'
print("Device: ", device, "\n");

# Training settings
batch_size = 60

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

latent_space_size=200

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_space_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.relu = nn.ReLU()
#        self.elu = nn.ELU(0.02, inplace=True)
#        self.tanh =  nn.Tanh()
        
    def forward(self, x): # 200
        x = self.fc1(x)      # 256
        x = self.relu(x)
        x = self.fc2(x)      # 512
        x = self.relu(x)
        x = self.fc3(x)      # 1024
        x = self.relu(x)
        x = self.fc4(x)      # 784
        x = self.relu(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.do = nn.Dropout(inplace=True)
#        self.elu = nn.ELU(0.02, inplace=True)
        self.relu = nn.ReLU()
        self.sm = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc4(x)
        x = self.sm(x)
        return x

G = Generator().to(device)
D = Discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=0.0001)
D_optimizer = optim.Adam(D.parameters(), lr=0.0001)
criterion = nn.BCELoss()

epochs=100
G_loss_arr=[]
D_loss_arr=[]
for epoch in range(epochs):
    G_loss_avg=0
    D_loss_avg=0
    for batch_idx, (real_data, _) in enumerate(train_loader):
        
        real_data = real_data.to(device)
          
        is_real = Variable(torch.ones(batch_size, 1, device=device))
        is_fake = Variable(torch.zeros(batch_size, 1, device=device))
        
        # Scenario1
        z = Variable(torch.randn(batch_size, latent_space_size, device=device))
        fake_data = G(z)
        
        D_result_from_fake = D(fake_data)
        G_loss = criterion(D_result_from_fake, is_real)
        
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
        # Scenario2
        D_result_from_real = D(real_data)
        D_loss_real = criterion(D_result_from_real, is_real)
    
        z = Variable(torch.randn(batch_size, latent_space_size, device=device))
        fake_data = G(z)
        D_result_from_fake = D(fake_data)
        D_loss_fake = criterion(D_result_from_fake, is_fake)

        D_loss = D_loss_real + D_loss_fake
        
        
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        
        G_loss_avg += G_loss
        D_loss_avg += D_loss
        
        if batch_idx % 200 == 0:
            print('Epoch: {} [{}/{}({:.0f}%)] G_loss: {:.2f} D_loss: {:.2f}'.format(
                epoch+1, batch_idx * len(real_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), G_loss.item(), D_loss.item()))

        
    G_loss_avg /= batch_size
    D_loss_avg /= batch_size
    G_loss_arr.append(G_loss)
    D_loss_arr.append(D_loss)

    if (epoch+1) % 10 == 1 or epoch+1<=10:
        fake_img = fake_data.reshape([batch_size, 1, 28, 28])
        img_grid = make_grid(fake_img, nrow=10, normalize=True)
        save_image(img_grid, "VanillaGAN-fake-img/epoch%03d.png"%(epoch+1))
        print('Epoch: {}\tG_loss: {:.2f} D_loss: {:.2f}'.format(epoch+1, G_loss_avg, D_loss_avg))
        
from matplotlib import pyplot as plt

plt.plot([i+1 for i in range(len(G_loss_arr))], G_loss_arr)
plt.plot([i+1 for i in range(len(D_loss_arr))], D_loss_arr)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('VanillaGAN')
plt.legend(['G_loss', 'D_loss'])
plt.show()
