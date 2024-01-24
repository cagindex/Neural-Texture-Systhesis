from typing import Any, List, Union
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import copy
from pathlib import Path

import torchvision.models as models
from IPython import embed
from PIL import Image

'''
Modified VGG19
'''
PATH    = (Path(__file__) / '..' / 'vgg19-dcbb9e9d.pth').resolve()

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # define structure
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        # define conv
        self.conv      = self.__make_layers(cfg)
        self.avgpool   = nn.AdaptiveAvgPool2d( ( 7, 7 ) )
        # define fc 
        self.fc        = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
    
    def __make_layers(self, cfg : List[Union[str, int]]):
        layers = []
        in_channel = 3
        for item in cfg:
            if ( item == 'M' ):
                layers += [ nn.MaxPool2d( kernel_size = 2, stride = 2 ) ]
            else:
                layers += [
                    nn.Conv2d( in_channel, item, kernel_size = 3, padding = 1 ),
                    nn.ReLU(True)
                ]
                in_channel = item
        return nn.Sequential(*layers)
    
    def forward(self, x : torch.Tensor):
        '''
        前向传播,不考虑全连接层
        '''
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def Load(self, PATH):
        '''
        加载预处理数据 (位置在PATH)
        ps: 这里会处理一些dict不匹配的问题
        '''
        model_dict1 = self.state_dict()
        model_dict2 = torch.load(PATH)

        model_list1 = list(model_dict1.keys())
        model_list2 = list(model_dict2.keys())

        model1_idx, model2_idx = 0, 0
        while ( model1_idx != len(model_list1) and model2_idx != len(model_list2) ):
            name1 = model_list1[model1_idx]
            name2 = model_list2[model2_idx]

            if ( re.search( 'batches', name1 ) != None ):
                model1_idx += 1
                continue

            assert model_dict1[name1].shape == model_dict2[name2].shape
            model_dict1[name1] = model_dict2[name2]

            model1_idx += 1
            model2_idx += 1
        
        self.load_state_dict(model_dict1)
        self.eval()

def get_model(device, middle_layer):
    vgg = VGG19()
    vgg.Load(PATH)
    vgg = vgg.conv.to(device).eval()
    vgg = copy.deepcopy(vgg)

    model = []
    i = 0
    tmp = nn.Sequential()
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
            layer.padding_mode = 'reflect'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)

            tmp.add_module(name, layer)
            model.append(tmp)
            tmp  = nn.Sequential()
            continue
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
            if middle_layer == 'M':
                layer = nn.MaxPool2d(kernel_size=2, stride=2)
            elif middle_layer == 'A':
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                raise RuntimeError(f'Unrecognized middle layer')
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
        tmp.add_module(name, layer)
    return nn.Sequential(*model)



class Modified_VGG19(nn.Module):
    def __init__(self, device, middle_layer = 'A', rescale = False):
        super().__init__()
        self.model = get_model(device, middle_layer)[:-2]

        self.feature_maps = [None for i in range(len(self.model))]


    def forward(self, x):
        # imagenet
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inp = normalize(x)
        # inp = x.clone()
        # inp[:, 0:1, ...] = (x[:, 0:1, ...] - 0.485) / 0.229
        # inp[:, 1:2, ...] = (x[:, 1:2, ...] - 0.456) / 0.224
        # inp[:, 2:3, ...] = (x[:, 2:3, ...] - 0.406) / 0.225

        # inp[0:1, :, ...] = (x[0:1, :, ...] - 0.485) / 0.229
        # inp[1:2, :, ...] = (x[1:2, :, ...] - 0.456) / 0.224
        # inp[2:3, :, ...] = (x[2:3, :, ...] - 0.406) / 0.225
        res = []

        for idx in range(len(self.model)):
            inp = self.model[idx](inp)
            res.append(inp)
            self.feature_maps[idx] = inp.clone()

        return res
    
    def get_feature_maps(self):
        return [ item.clone() for item in self.feature_maps ]
    
    def get_gram_matrices(self):
        feature_maps_flattened = [ torch.flatten(feature, start_dim=1) for feature in self.feature_maps ] 
        Gram_Matrices = [ torch.matmul(item, torch.transpose(item, 0, 1)) for item in feature_maps_flattened ]
        return Gram_Matrices



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Modified_VGG19(device)

    image = Image.open((Path(__file__) / '..' / '..' / 'Images' / 'pebbles.jpg').resolve())

    # embed()

    structure = torch.nn.Sequential(*list(model.children())[:])
    print(structure)



