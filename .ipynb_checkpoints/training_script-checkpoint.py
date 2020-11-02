import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision.utils import make_grid
#import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import numpy as np
import argparse
import random
import pickle as pkl
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim

    

from torchvision.models import vgg19

from collections import OrderedDict


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values
        channels: the number of channels throughout the residual block, a scalar
    '''

    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.layers(x)
    
    
    
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        base_channels: number of channels throughout the generator, a scalar
        n_ps_blocks: number of PixelShuffle blocks, a scalar
        n_res_blocks: number of residual blocks, a scalar
    '''

    def __init__(self, base_channels=64, n_ps_blocks=2, n_res_blocks=16):
        super().__init__()
        # Input layer
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(base_channels)]

        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # PixelShuffle blocks
        ps_blocks = []
        for _ in range(n_ps_blocks):
            ps_blocks += [
                nn.Conv2d(base_channels, 4 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # Output layer
        self.out_layer = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.out_layer(x)
        return x
    
    
    
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        base_channels: number of channels in first convolutional layer, a scalar
        n_blocks: number of convolutional blocks, a scalar
    '''

    def __init__(self, base_channels=64, n_blocks=3):
        super().__init__()
        self.blocks = [
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        cur_channels = base_channels
        flatten=Flatten()
        for i in range(n_blocks):
            self.blocks += [
                nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(2 * cur_channels, 2 * cur_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            cur_channels *= 2

        self.blocks += [
            # You can replicate nn.Linear with pointwise nn.Conv2d
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * cur_channels, 1, kernel_size=1, padding=0),

            # Apply sigmoid if necessary in loss function for stability
            flatten,
        ]

        self.layers = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.layers(x)


from torchvision.models import vgg19

class Loss(nn.Module):
    '''
    Loss Class
    Implements composite content+adversarial loss for SRGAN
    Values:
        device: 'cuda' or 'cpu' hardware to put VGG network on, a string
    '''

    def __init__(self, device='cuda'):
        super().__init__()

        vgg = vgg19(pretrained=True).to(device)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    @staticmethod
    def img_loss(x_real, x_fake):
        return F.mse_loss(x_real, x_fake)

    def adv_loss(self, x, is_real):
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)

    def vgg_loss(self, x_real, x_fake):
        return F.mse_loss(self.vgg(x_real), self.vgg(x_fake))

    def forward(self, generator, discriminator, hr_real, lr_real):
        ''' Performs forward pass and returns total losses for G and D '''
        hr_fake = generator(lr_real)
        #my line
        hr_fake=F.interpolate(hr_fake,size=(218,178))
        #--------
        fake_preds_for_g = discriminator(hr_fake.detach())
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())

        g_loss = (
            0.005 * self.adv_loss(fake_preds_for_g, False) + \
            0.09 * self.vgg_loss(hr_real, hr_fake) + \
            1.3*self.img_loss(hr_real, hr_fake)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )

        return g_loss, d_loss, hr_fake
    
    

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
def train(gen,disc,loss,n_epochs,train_dataloader,models_to_save_dir):
    
    for epoch in range(n_epochs):
        
        for i, batch in enumerate(train_dataloader):
            
            gen_opt.zero_grad()
            super_res=batch[0]
            modes=['nearest','bilinear','bicubic','area']
            modee=random.choice(modes)
            if modee=='bilinear':
                x=True
            else:
                x=False
            crap=F.interpolate(super_res,size=(70,70),mode='nearest')
            fake=gen(crap)
            fake=F.interpolate(input=fake,size=super_res[0][1].shape)
            g_loss,d_loss,hr_fake=loss(gen,disc,super_res,crap)
            gen_opt.zero_grad()
            g_loss.backward()
            gen_opt.step()

            disc_opt.zero_grad()
            d_loss.backward()
            disc_opt.step()  
            
            fake_2=gen(crap)
            
            print('gen_loss {}, disc_loss {}'.format(g_loss,d_loss))
            
            if i% 30 == 0:
                torch.save(gen,os.path.join(
                    models_to_save_dir,'gen{}_{}.pb'.format(epoch,g_loss)))
                
  

    
    
    
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=32,
                       help='input batch-size for training(default:128)')
    parser.add_argument('--n-epochs',type=int,default=10,
                       help='no of epochs to train for ..')
    parser.add_argument('--img-size',default=(32,32),
                       help='img-size to crop ..')
    
  
    parser.add_argument('--lr',type=float,default=0.000002,
                       help='the learning rate')
    parser.add_argument('--beta1',type=float,default=0.2,
                       help='the beta1')
    parser.add_argument('--beta2',type=float,default=0.7,
                       help='the beta2')    
    
    
    parser.add_argument('--data-dir',type=str,default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args=parser.parse_args()

    gen=Generator()
    disc=Discriminator()
    
    beta1=args.beta1
    beta2=args.beta2
    # Create optimizers for the discriminator D and generator G
    disc_opt = optim.Adam(disc.parameters(), args.lr, [beta1, beta2])
    gen_opt = optim.Adam(gen.parameters(),args.lr, [beta1, beta2])
    
    trans=torchvision.transforms.ToTensor()
    dataset=torchvision.datasets.ImageFolder(root=args.data_dir,transform=trans)
    train_dataloader =DataLoader(dataset,args.batch_size)
    
    loss=Loss(device='cpu')
    train(gen,disc,loss,args.n_epochs,train_dataloader,args.model_dir)