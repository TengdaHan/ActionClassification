import torch.nn as nn
import torch


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),
                              stride=(1,stride,stride),padding=(0,padding,padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),
                               stride=(stride,1,1),padding=(padding,0,0), bias=False)

        self.bn1=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        # init
        self.conv1.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.conv2.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        return x

class SepInception(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SepInception, self).__init__()

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [num_out_0_0a, 
        num_out_1_0a, num_out_1_0b,
        num_out_2_0a, num_out_2_0b, 
        num_out_3_0b] = out_planes

        self.branch0 = nn.Sequential(
            BasicConv3d(in_planes, num_out_0_0a, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_planes, num_out_1_0a, kernel_size=1, stride=1),
            STConv3d(num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_planes, num_out_2_0a, kernel_size=1, stride=1),
            STConv3d(num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(in_planes, num_out_3_0b, kernel_size=1, stride=1),
        )

        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):

    def __init__(self, input_channel=3):
        super(S3D, self).__init__()
	# assume input C,T,H,W == (3, 64, 224, 224)

        self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3) 

        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)
            
        ###################################

        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Conv_2b = BasicConv3d(64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = STConv3d(64, 192, kernel_size=3, stride=1, padding=1) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a, # (64, 32, 56, 56)
            self.Conv_2b,    # (64, 32, 56, 56)
            self.Conv_2c)    # (192, 32, 56, 56)

        ###################################
        
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = SepInception(in_planes=192, out_planes=[64, 96, 128, 16, 32, 32])
        self.Mixed_3c = SepInception(in_planes=256, out_planes=[128, 128, 192, 32, 96, 64])

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32, 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        ###################################
        
        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = SepInception(in_planes=480, out_planes=[192, 96, 208, 16, 48, 64])
        self.Mixed_4c = SepInception(in_planes=512, out_planes=[160, 112, 224, 24, 64, 64])
        self.Mixed_4d = SepInception(in_planes=512, out_planes=[128, 128, 256, 24, 64, 64])
        self.Mixed_4e = SepInception(in_planes=512, out_planes=[112, 144, 288, 32, 64, 64])
        self.Mixed_4f = SepInception(in_planes=528, out_planes=[256, 160, 320, 32, 128, 128])

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        ###################################
        
        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = SepInception(in_planes=832, out_planes=[256, 160, 320, 32, 128, 128])
        self.Mixed_5c = SepInception(in_planes=832, out_planes=[384, 192, 384, 48, 128, 128])

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x 

