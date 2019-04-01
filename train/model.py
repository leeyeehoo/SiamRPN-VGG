import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=0, bias=False)
        
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.padding = nn.ReplicationPad2d(1)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding(out)
        out = self.conv2(out)
        
        
        
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,
                               bias=False)
        self.padding1 = nn.ReplicationPad2d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.padding1(x)
        
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

##upsample
class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out
class backend(nn.Module):
    def __init__(self,upblock, num_layers):
        super(backend, self).__init__()
        self.in_channels = 2048
        self.uplayer1 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        
        
    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels*2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)    
    def forward(self, x):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        return x
    
    
class SiamRes(nn.Module):
    def __init__(self):
        super(SiamRes, self).__init__()
        self.model = resnet101(pretrained=False)
        #self.backend = backend(DeconvBottleneck, [3, 4, 23, 2])
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        
        
        #self.up_sample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.bn_adjust = nn.BatchNorm2d(1)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)
    def forward_once(self,x):
        x = self.model(x)
        #x = self.backend(x)
        output = self.conv1(x)
        
        return output
    def forward(self, z, x):
        z = self.forward_once(z)
        x = self.forward_once(x)
        output = self.xcorr(z, x)
        output = self.bn_adjust(output)
        
        return output

class SiamResnet(nn.Module):
    def __init__(self):
        super(SiamResnet, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        #self.backend = backend(DeconvBottleneck, [3, 4, 23, 2])
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        
        self.model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        
        #self.up_sample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.bn_adjust = nn.BatchNorm2d(1)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)
    def forward_once(self,x):
        x = self.model(x)
        #x = self.backend(x)
        output = self.conv1(x)
        
        return output
    def forward(self, z, x):
        z = self.forward_once(z)
        x = self.forward_once(x)
        output = self.xcorr(z, x)
        output = self.bn_adjust(output)
        
        return output
    
class SiamResnet101(nn.Module):
    def __init__(self):
        super(SiamResnet101, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 23, 3])
        #self.backend = backend(DeconvBottleneck, [3, 4, 23, 2])
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        
        self.model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        
        #self.up_sample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.bn_adjust = nn.BatchNorm2d(1)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)
    def forward_once(self,x):
        x = self.model(x)
        #x = self.backend(x)
        output = self.conv1(x)
        
        return output
    def forward(self, z, x):
        z = self.forward_once(z)
        x = self.forward_once(x)
        output = self.xcorr(z, x)
        output = self.bn_adjust(output)
        
        return output    
    
class SiamVGG(nn.Module):
    def __init__(self):
        super(SiamVGG, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backend = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
 
        
        
        #self.out_bias = Variable(torch.zeros(1,1,1,1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)

    def forward_once(self,x):
        output = self.frontend(x)
        output = self.backend(output)
        return output
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = self.xcorr(output1, output2)
        
        output = self.bn_adjust(output)

        return output
    
    
class SiamVGGwith1by1(nn.Module):
    def __init__(self):
        super(SiamVGGwith1by1, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        self.middelend = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1),
                                      nn.Conv2d(256, 512, kernel_size=1, stride=1),
                                      nn.Conv2d(512, 256, kernel_size=1, stride=1),
                                      nn.Conv2d(256, 512, kernel_size=1, stride=1),
                                      nn.Conv2d(512, 256, kernel_size=1, stride=1),
                                      nn.Conv2d(256, 512, kernel_size=1, stride=1),)
        
        self.backend = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
 
        
        
        #self.out_bias = Variable(torch.zeros(1,1,1,1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)

    def forward_once(self,x):
        output = self.frontend(x)
        output = self.middelend(output)
        output = self.backend(output)
        return output
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = self.xcorr(output1, output2)
        
        output = self.bn_adjust(output)

        return output
    
class SiamVGGwithpadding(nn.Module):
    def __init__(self):
        super(SiamVGGwithpadding, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backend = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        
        self.bn_adjust = nn.BatchNorm2d(1)
        
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
 
        
        
        #self.out_bias = Variable(torch.zeros(1,1,1,1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)

    def forward_once(self,x):
        output = self.frontend(x)
        output = self.backend(output)
        return output
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = self.xcorr(output1, output2)
        
        output = self.bn_adjust(output)

        return output    
    
class DualVGGwithpadding(nn.Module):
    def __init__(self):
        super(DualVGGwithpadding, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontendx = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backendx = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        self.frontendz = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backendz = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        self.bn_adjust = nn.BatchNorm2d(1)
        
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontendx.state_dict().items())):
            self.frontendx.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
            self.frontendz.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
 
        
        
        #self.out_bias = Variable(torch.zeros(1,1,1,1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)

    def forward_once_x(self,x):
        output = self.frontendx(x)
        output = self.backendx(output)
        return output
    def forward_once_z(self,z):
        output = self.frontendz(z)
        output = self.backendz(output)
        return output
    def forward(self, input1, input2):
        output1 = self.forward_once_z(input1)
        output2 = self.forward_once_x(input2)
        output = self.xcorr(output1, output2)
        
        output = self.bn_adjust(output)

        return output    


class DualResnet(nn.Module):
    def __init__(self):
        super(DualResnet, self).__init__()
        self.modelx = ResNet(Bottleneck, [3, 4, 6, 3])
        #self.backend = backend(DeconvBottleneck, [3, 4, 23, 2])
        self.conv1x = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.modelz = ResNet(Bottleneck, [3, 4, 6, 3])
        #self.backend = backend(DeconvBottleneck, [3, 4, 23, 2])
        self.conv1z = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        
        self.modelx.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.modelz.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        
        #self.up_sample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.bn_adjust = nn.BatchNorm2d(1)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)
    def forward_once_x(self,x):
        x = self.modelx(x)
        #x = self.backend(x)
        output = self.conv1x(x)
        
        return output
    def forward_once_z(self,x):
        x = self.modelz(x)
        #x = self.backend(x)
        output = self.conv1z(x)
        
        return output
    def forward(self, z, x):
        z = self.forward_once_z(z)
        x = self.forward_once_x(x)
        output = self.xcorr(z, x)
        output = self.bn_adjust(output)
        
        return output    
class DaSiam(nn.Module):
    def __init__(self):
        super(DaSiam, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        
        
        
        
        #self.up_sample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.bn_adjust = nn.BatchNorm2d(1)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)
    def forward_once_x(self,x):
        x = self.featureExtract(x)
        #x = self.backend(x)
        output = self.conv1(x)
        
        return output
    def forward_once_z(self,x):
        x = self.featureExtract(x)
        #x = self.backend(x)
        output = self.conv1(x)
        
        return output
    def forward(self, z, x):
        z = self.forward_once_z(z)
        x = self.forward_once_x(x)
        output = self.xcorr(z, x)
        output = self.bn_adjust(output)
        
        return output        
    
class DifferentialNet(nn.Module):
    def __init__(self):
        super(DifferentialNet, self).__init__()
        self.modelx = ResNet(Bottleneck, [3, 4, 6, 3])
        #self.backend = backend(DeconvBottleneck, [3, 4, 23, 2])
        self.conv1x = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.frontendz = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backendz = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        
        self.modelx.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontendz.state_dict().items())):
            self.frontendz.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
            
        #self.up_sample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.bn_adjust = nn.BatchNorm2d(1)
    def xcorr(self,z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))        
        return torch.cat(out, dim=0)
    def forward_once_x(self,x):
        x = self.modelx(x)
        #x = self.backend(x)
        output = self.conv1x(x)
        
        return output
    def forward_once_z(self,x):
        x = self.frontendz(x)
        #x = self.backend(x)
        output = self.backendz(x)
        
        return output
    def forward(self, z, x):
        z = self.forward_once_z(z)
        x = self.forward_once_x(x)
        #print z.shape
        #print x.shape
        output = self.xcorr(z, x)
        output = self.bn_adjust(output)
        
        return output  
    
class SiamVGGyolo(nn.Module):
    def __init__(self):
        super(SiamVGGyolo, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backend_z = nn.Sequential(nn.Conv2d(512, 256*5, kernel_size=1, stride=1))
        self.backend_x = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))
        
        self.bn_adjust = nn.BatchNorm2d(5)
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
 
        
        
        #self.out_bias = Variable(torch.zeros(1,1,1,1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def xcorr(self,z, x):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0).view(5,256,kernel_size,kernel_size)))        
        return torch.cat(out, dim=0)

    def forward_once_x(self,x):
        output = self.frontend(x)
        output = self.backend_x(output)
        return output
    
    def forward_once_z(self,x):
        output = self.frontend(x)
        output = self.backend_z(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once_z(input1)
        output2 = self.forward_once_x(input2)
        output = self.xcorr(output1, output2)
        
        output = self.bn_adjust(output)

        return output    
    
class SiamVGGyolo2(nn.Module):
    def __init__(self):
        super(SiamVGGyolo2, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        
        self.backend_z = nn.Sequential(nn.Conv2d(512, 256*5, kernel_size=3, stride=1,padding = 0))
        self.backend_x = nn.Sequential(nn.Conv2d(512, 256*5, kernel_size=3, stride=1,padding = 0))
        self.bn_adjust = nn.BatchNorm2d(5)
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
 
        
        
        #self.out_bias = Variable(torch.zeros(1,1,1,1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def xcorr(self,z, x):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0).view(5,256,kernel_size,kernel_size),groups = 5))        
        return torch.cat(out, dim=0)

    def forward_once_x(self,x):
        output = self.frontend(x)
        output = self.backend_x(output)
        return output
    
    def forward_once_z(self,x):
        output = self.frontend(x)
        output = self.backend_z(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once_z(input1)
        output2 = self.forward_once_x(input2)
        output = self.xcorr(output1, output2)
        output = self.bn_adjust(output)

        return output   
    
class SiamVGGyolo2plus(nn.Module):
    def __init__(self):
        super(SiamVGGyolo2plus, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        
        self.conv_cls1 = nn.Conv2d(512, 512 * 2 * 5, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(512, 512 * 4 * 5, kernel_size=3, stride=1, padding=0)
        self.conv_cls2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * 5, 4 * 5, 1)
        
        
        
        
        
        
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
 
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def xcorr(self,z, x,channels):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0).view(channels,512,kernel_size,kernel_size)))     
                       
        return torch.cat(out, dim=0)


    
    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.frontend(template)
        detection_feature = self.frontend(detection)

        kernel_score = self.conv_cls1(template_feature)
        kernel_regression = self.conv_r1(template_feature)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)
        
        pred_score = self.xcorr(kernel_score, conv_score,10)
        pred_regression = self.regress_adjust(self.xcorr(kernel_regression, conv_regression,20))
        
        
        
        
        

        return pred_score, pred_regression
    
class YNetYolo(nn.Module):
    def __init__(self):
        super(YNetYolo, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.backend = nn.Sequential(
            
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
#             nn.Conv2d(512, 256, kernel_size=1, stride=1)
        )

#        self.oup = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1))
#         self.conv_cls1 = nn.Conv2d(512, 512 * 1 * 5, kernel_size=3, stride=1, padding=0)
#         self.conv_r1 = nn.Conv2d(512, 512 * 4 * 5, kernel_size=3, stride=1, padding=0)
#         self.conv_cls2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
#         self.conv_r2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
#         self.regress_adjust = nn.Conv2d(4 * 5, 4 * 5, 1)
        
        self.conv_cls = nn.Conv2d(512, 2 * 5, kernel_size=3, stride=1, padding=1)
        self.conv_r = nn.Conv2d(512, 4 * 5, kernel_size=3, stride=1, padding=1)
        
        self.regress_adjust = nn.Conv2d(4 * 5, 4 * 5, 1)
        
        
        
        self._initialize_weights()
        
        mod = models.vgg16_bn(pretrained = True)
        
        start = 0
        
        for i in xrange(len(self.frontend.state_dict().items())):
            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
            start = start + 1
        
        for i in xrange(len(self.backend.state_dict().items())):
            self.backend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[start+i][1].data[:]
            
         
        
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def xcorr(self,z, x):
        out = []
        kernel_size = z.data.size()[-1]
        channels = z.data.size()[1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0).view(channels,1,kernel_size,kernel_size),groups = channels))     
                       
        return torch.cat(out, dim=0)


    
    def forward(self, template, detection):
        t = self.frontend(template)
        d = self.frontend(detection)
        
        
        f = self.xcorr(t,d)
        
        
        
        f = self.backend(f)
        #f = self.oup(f)
        
        pred_score = self.conv_cls(f)
        pred_regression = self.regress_adjust(self.conv_r(f))
        
        
        
        return pred_score, pred_regression     
            