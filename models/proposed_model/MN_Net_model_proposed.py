#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:01:31 2023

@author: venkatesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    #print(tar.shape[2:])
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

#%%

# This is the proposed model 

class MN_Net_proposed(nn.Module):

    def __init__(self,in_ch=3,out_ch=1,no_channels_dealing_in_and_out=64,no_channels_dealing_in_the_middle=16):
        super(MN_Net_proposed,self).__init__()

        self.stage1 = RSU7(in_ch,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)

        # decoder
        self.stage5d = RSU4F(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage4d = RSU4(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage3d = RSU5(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage2d = RSU6(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage1d = RSU7(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)


        self.side1 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(7*out_ch,out_ch,1)



        #   motivated from Half UNet Decoder.......
        self.up_hx2_to_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_hx3_to_x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hx4_to_x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_hx5_to_x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_hx6_to_x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)



        self.final_stage_of_half_UNet = RSU7(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.outconv_Half_of_UNet = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,1)
        self.decision=nn.Sigmoid()







    def forward(self,x):
        # print('This is the proposed model')
        hx = x
        #print('input:','source image',hx.shape)

        #stage 1
        hx1 = self.stage1(hx)
        #print('stage1:','RSU7','hx1.shape',hx1.shape)
        hx = self.pool12(hx1)
        #print('After maxpooling:',hx.shape)

        #stage 2
        hx2 = self.stage2(hx)
        #print('stage2:','RSU6','hx2.shape',hx2.shape)

        hx = self.pool23(hx2)
        #print('maxpooling',hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        #print('stage3:','RSU5','hx3.shape',hx3.shape)

        hx = self.pool34(hx3)
        #print('maxpooling',hx.shape)

        #stage 4
        hx4 = self.stage4(hx)
        #print('stage4:','RSU4','hx4.shape',hx4.shape)

        hx = self.pool45(hx4)
        #print('maxpooling',hx.shape)

        #stage 5
        hx5 = self.stage5(hx)
        #print('stage5:','RSU4F','hx5.shape:',hx5.shape)

        hx = self.pool56(hx5)
        #print('maxpooling',hx.shape)

        #stage 6
        hx6 = self.stage6(hx)
        #print('stage6:','RSU4F','hx6.shape',hx6.shape)
















        #print("#"*50)
        hx6up = _upsample_like(hx6,hx5)
        #print('_upsample_like:','hx6up',hx6up.shape)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        #print('stage5d:','RSU4F','hx5d.shape',hx5d.shape)

        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        #print('stage4d:','RSU4','hx4d.shape',hx4d.shape)

        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        #print('stage3d:','RSU5','hx3d.shape',hx3d.shape)

        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        #print('stage2d:','RSU6','hx2d.shape',hx2d.shape)

        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        # print('stage1d:','RSU7','hx1d.shape',hx1d.shape)


        #side output
        d1 = self.side1(hx1d)
        # print('d1.shape',d1.shape)
        d2 = self.side2(hx2d)
        #print('d2.shape',d2.shape)

        d2 = _upsample_like(d2,d1)
        #print('d2.shape',d2.shape)

        d3 = self.side3(hx3d)
        #print('d3.shape',d3.shape)

        d3 = _upsample_like(d3,d1)
        #print('d3.shape',d3.shape)

        d4 = self.side4(hx4d)
        #print('d4.shape',d4.shape)

        d4 = _upsample_like(d4,d1)
        #print('d4.shape',d4.shape)

        d5 = self.side5(hx5d)
        #print('d5.shape',d5.shape)

        d5 = _upsample_like(d5,d1)
        #print('d5.shape',d5.shape)

        d6 = self.side6(hx6)
        #print('d6.shape',d6.shape)
        d6 = _upsample_like(d6,d1)
        #print('d6.shape',d6.shape)

        

        # Half UNet decoder adding
        #print('Half UNet Decooder implementation')
        hx2_upsample_to_orig_shape=self.up_hx2_to_x(hx2)
        #print(hx2_upsample_to_orig_shape.shape)

        hx3_upsample_to_orig_shape=self.up_hx3_to_x(hx3)
        #print(hx3_upsample_to_orig_shape.shape)

        hx4_upsample_to_orig_shape=self.up_hx4_to_x(hx4)
        #print(hx4_upsample_to_orig_shape.shape)

        hx5_upsample_to_orig_shape=self.up_hx5_to_x(hx5)
        #print(hx5_upsample_to_orig_shape.shape)

        hx6_upsample_to_orig_shape=self.up_hx6_to_x(hx6)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape

        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)
        d7=self.outconv_Half_of_UNet(temp_output_1)

        output_of_Half_UNet=self.decision(d7)

        
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6,d7),1))
        concated_feature=torch.cat((d1,d2,d3,d4,d5,d6,d7),1)
        #print('output_of_Half_UNet.shape:',output_of_Half_UNet.shape)
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6),output_of_Half_UNet
        # return d0,d1,d2,d3,d4,d5,d6,d7,concated_feature,hx1d

        # return F.sigmoid(d0)
#%%

#%%




def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")
#%%
from torchsummary import summary
from pthflops import count_ops

input_data=torch.randn(1,3,448,320)
input_data=input_data.float()
input_data=input_data.cuda()


model = MN_Net_proposed(in_ch=3,
                        out_ch=1,
                        no_channels_dealing_in_and_out=64,
                        no_channels_dealing_in_the_middle=32)
model=model.float()
model=model.cuda()

with torch.no_grad():
    for i in range(100):

        #print(i)
        tic()
        # output,d1,d2,d3,d4,d5,d6,d7=model(input)
        output=model(input_data)
        toc()
#%%
summary(model, (3, 448, 320))
#%%
torch.save(model.state_dict(),'U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels_updated_architecture_64_32.pth')
#%%
import re
from ptflops import get_model_complexity_info
from pthflops import count_ops

macs, params = get_model_complexity_info(model, (3, 448, 320), as_strings=True,
print_per_layer_stat=True, verbose=True)
# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))
#%%
from thop import profile

flops, params = profile(model, inputs=(input_data,))

print(f"Total FLOPs: {flops}")
print(f"Total parameters: {params}")
#%%
from torchprofile import profile_macs

flops = profile_macs(model, input_data)
print("Total FLOPs:", flops)
