import torch
import torch.nn as nn
import torchvision.models as models

class Res18BB(nn.Module):
    def __init__(self, num_classes):
        super(Res18BB, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(16)

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        
        x = self.classifier(x)

        return x
    


class ConvBlock(nn.Module):
    '''convolutional block'''
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_c)
        self.batch_n = True

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        if self.batch_n:
            x = self.bn(x)

        x = self.conv2(x)
        x = self.relu(x)
        if self.batch_n:
            x = self.bn(x)
                
        return x

class EncoderBlock(nn.Module):
    '''encoder block'''
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, c):

        c = self.conv(c)
        p = self.pool(c)

        return c, p # return convolutional (c) part for concatenating

class DecoderBlock(nn.Module):
    '''
    decoder block
    skip_features:: result from conv block to concatenate
    '''
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()
        
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_c, in_c//2, kernel_size=2, stride=2)#args.stride)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip_features):
   
        x = self.up(x)
        c = torch.cat([x, skip_features], dim=1)
        c = self.conv(c)

        return c

class FSUNet(nn.Module):

    def __init__(self, in_c, out_c, num_classes):
        super(FSUNet, self).__init__()
        
        self.enc1 = EncoderBlock(3, out_c)

        self.enc2 = EncoderBlock(out_c, int(2*out_c))
        self.enc3 = EncoderBlock(int(2*out_c), int(2*2*out_c))
        self.enc4 = EncoderBlock(int(2*2*out_c), int(2*2*2*out_c))

        self.conv = ConvBlock(int(2*2*2*out_c), int(2*2*2*2*out_c))

        self.dec1 = DecoderBlock(int(2*2*2*2*out_c), int(2*2*2*out_c))
        self.dec2 = DecoderBlock(int(2*2*2*out_c), int(2*2*out_c))
        self.dec3 = DecoderBlock(int(2*2*out_c), int(2*out_c))
        self.dec4 = DecoderBlock(int(2*out_c), out_c)

        self.output = nn.Conv2d(out_c, num_classes, kernel_size=1)


    def forward(self, img):
        # input shape (bs, channels, height, width)
        c1, p1 = self.enc1(img)
        c2, p2 = self.enc2(p1)
        c3, p3 = self.enc3(p2)
        c4, p4 = self.enc4(p3)

        b = self.conv(p4)
        
        x = self.dec1(b, c4)
        x = self.dec2(x, c3)
        x = self.dec3(x, c2)
        x = self.dec4(x, c1) 
        
        x = self.output(x)

        return x
    


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)



# # Create the network with the desired number of output classes
# num_classes = 20  # Adjust this number based on your dataset
# fcn = FCN(num_classes=num_classes)


    