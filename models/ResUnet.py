#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:49:33 2023

@author: venkatesh
"""

import torch.nn as nn
import torch





class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
#%%

#%%
import torch.nn.functional as F

def _upsample_like(src,tar):
    ##print(tar.shape[2:])
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

#%%

class ResUnet_updated(nn.Module):
    def __init__(self, channel, out_channel=1,filters=[64, 128, 256, 512, 512]):
        super(ResUnet_updated, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.residual_conv_4 = ResidualConv(filters[3], filters[4], 2, 1)

        self.bridge = ResidualConv(filters[4], filters[4], 2, 1)

        self.up_residual_conv1 = ResidualConv(2*filters[4] , filters[3], 1, 1)
        self.up_residual_conv2 = ResidualConv(2*filters[3] , filters[2], 1, 1)
        self.up_residual_conv3 = ResidualConv(2*filters[2] , filters[1], 1, 1)
        self.up_residual_conv4 = ResidualConv(2*filters[1] , filters[0], 1, 1)
        self.up_residual_conv5 = ResidualConv(2*filters[0] , filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        #print(x.shape)
        x1 = self.input_layer(x) + self.input_skip(x)
        #print('x1.shape',x1.shape)

        #64->128
        x2 = self.residual_conv_1(x1)
        #print('x2.shape',x2.shape)

        #128->256

        x3 = self.residual_conv_2(x2)
        #print('x3.shape',x3.shape)

        #256->512

        x4 = self.residual_conv_3(x3)
        #print('x4.shape',x4.shape)

        #512->512

        x5 = self.residual_conv_4(x4)
        #print('x5.shape',x5.shape)

        #512->512
        x6 = self.bridge(x5)
        #print('x6.shape',x6.shape)

        # Decode
        x6up = _upsample_like(x6,x5)
        #print('x6up.shape',x6up.shape)

        temp_x = torch.cat([x6up, x5], dim=1)
        #print('temp_x.shape',temp_x.shape)
        dx5 = self.up_residual_conv1(temp_x)
        #print('dx5.shape',dx5.shape)

        dx5_up = _upsample_like(dx5,x4)

        #print('dx5_up.shape',dx5_up.shape)
        temp_x = torch.cat([dx5_up, x4], dim=1)
        #print('temp_x.shape',temp_x.shape)
        dx4 = self.up_residual_conv2(temp_x)
        #print('dx4.shape',dx4.shape)


        dx4_up = _upsample_like(dx4,x3)

        #print('dx4_up.shape',dx4_up.shape)
        temp_x = torch.cat([dx4_up, x3], dim=1)
        #print('temp_x.shape',temp_x.shape)
        dx3 = self.up_residual_conv3(temp_x)
        #print('dx3.shape',dx3.shape)


        dx3_up = _upsample_like(dx3,x2)

        #print('dx3_up.shape',dx3_up.shape)
        temp_x = torch.cat([dx3_up, x2], dim=1)
        #print('temp_x.shape',temp_x.shape)
        dx2 = self.up_residual_conv4(temp_x)
        #print('dx2.shape',dx2.shape)

        dx2_up = _upsample_like(dx2,x1)

        #print('dx2_up.shape',dx2_up.shape)
        temp_x = torch.cat([dx2_up, x1], dim=1)
        #print('temp_x.shape',temp_x.shape)
        dx1 = self.up_residual_conv5(temp_x)
        # print('dx1.shape',dx1.shape)

    
        output = self.output_layer(dx1)
        #print('output.shape',output.shape)

        # return output,dx1
        return output

#     #%%\
# def tic():
#     # Homemade version of matlab tic and toc functions
#     import time
#     global startTime_for_tictoc
#     startTime_for_tictoc = time.time()

# def toc():
#     import time
#     if 'startTime_for_tictoc' in globals():
#         print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
#         print(str(time.time() - startTime_for_tictoc) )
#     else:
#         print("Toc: start time not set")
     
#%%
# model=ResUnet_updated(channel=3)
# model=model.cuda()

# input_data=torch.rand(1,3,448,320)
# input_data=input_data.cuda()

# output=model(input)
# model=model.float()
# model=model.cuda()


# with torch.no_grad():
#     for i in range(1000):
#         tic()
#         output=model(input_data)
#         toc()
# torch.save(model.state_dict(), 'UNet_updated.pth')

# #%%
# from torchsummary import summary

# summary(model, (3, 448, 320))
# #%%
# from ptflops import get_model_complexity_info
# from pthflops import count_ops

# macs, params = get_model_complexity_info(model, (3, 448, 320), as_strings=True,
# print_per_layer_stat=True, verbose=True)
# # Extract the numerical value
# flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# # Extract the unit
# flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

# print('Computational complexity: {:<8}'.format(macs))
# print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
# print('Number of parameters: {:<8}'.format(params))
