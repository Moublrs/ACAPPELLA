__author__ = "Mouad"
'''my first torch module hope not the last one :p'''

import torch
import torch.nn as nn
import numpy as np


def conv(in_chan, out_chan):
    #Kernel    padding for /2
    # 3            1
    # 5            2
    # 7            3
    # 2K+1         K
    return nn.Sequential(
        nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(num_features=out_chan),
        nn.LeakyReLU(0.2,inplace=True)

    )

def deconv(in_chan, out_chan,dropout=False):
    #Typically, dropout is applied after the non-linear activation function #Sebastien RASCHKA
    if dropout==False:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=5, stride=2, padding=2,output_padding=1),
            nn.BatchNorm2d(num_features=out_chan),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.BatchNorm2d(num_features=out_chan),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )



def final_conv(in_chan, out_chan):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
                         nn.Sigmoid())


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Hout = H+2(P)−K+1 / S
        #Wout = W+2(P)−K+1 / S

        self.down_conv1=conv(1,16)
        self.down_conv2=conv(16,32)
        self.down_conv3=conv(32,64)
        self.down_conv4=conv(64,128)
        self.down_conv5=conv(128,256)
        self.down_conv6=conv(256,512)
        self.deconv1=deconv(512,256,dropout=True)
        self.deconv2=deconv(512,128,dropout=True)
        self.deconv3=deconv(256,64,dropout=True)
        self.deconv4=deconv(128,32)
        self.deconv5=deconv(64,16)

        self.final_conv=final_conv(32,1)

    def forward(self, image):

        # encoder
        x1 = self.down_conv1(image)
        #print(f"x1 (down1) : {x1.shape}")

        x2 = self.down_conv2(x1)
        #print(f"x2 (down2) : {x2.shape}")

        x3 = self.down_conv3(x2)
        #print(f"x3 (down3) : {x3.shape}")

        x4 = self.down_conv4(x3)
        #print(f"x4 (down4) : {x4.shape}")

        x5 = self.down_conv5(x4)
        #print(f"x5 (down5) : {x5.shape}")

        x6 = self.down_conv6(x5)
        #print(f"x6 (down6) : {x6.shape}")

        # decoder
        x7 = self.deconv1(x6)
        #print(f"x7 (up1)   : {x7.shape}  | skip x5 : {x5.shape}")

        x8 = self.deconv2(torch.cat((x7, x5), 1))
        #print(f"x8 (up2)   : {x8.shape}  | skip x4 : {x4.shape}")

        x9 = self.deconv3(torch.cat((x8, x4), 1))
        #print(f"x9 (up3)   : {x9.shape}  | skip x3 : {x3.shape}")

        x10 = self.deconv4(torch.cat((x9, x3), 1))
        #print(f"x10 (up4)  : {x10.shape} | skip x2 : {x2.shape}")

        x11 = self.deconv5(torch.cat((x10, x2), 1))
        #print(f"x11 (up5)  : {x11.shape} | skip x1 : {x1.shape}")

        final_layer = self.final_conv(torch.cat((x11, x1), 1))
        #print(f"final      : {final_layer.shape}")

        return final_layer


# if __name__ == '__main__':
#     #[batch_size, channels, height, width]
#     im=np.load("../data/spec_data/train/Leaf - Come Around_mixture/mixture_db.npy")[:-1,:128]
#     im=im[np.newaxis,np.newaxis,:,:]
#
#     model = UNet()
#     print(model(torch.from_numpy(im).float()))