from PIL import Image
import numpy as np
import random
import torch
import torchvision.transforms as transforms


def read_image(PATH : str) -> torch.Tensor:
    img = Image.open(PATH)
    return transforms.ToTensor()(img)

def create_white_noise_image(ref_img : torch.Tensor) -> torch.Tensor:
    return torch.rand(ref_img.shape)

def show_image(img : torch.Tensor, show = True):
    if img.device == 'cpu':
        image = img.detach().numpy()
    else:
        image = img.detach().cpu().numpy()
    # 防止数据溢出
    image = np.maximum(image, 0)
    image = np.minimum(image, 1)

    image = (image * 255).astype(np.uint8).transpose((1, 2, 0))
    image = Image.fromarray(image)
    if (show) :
        image.show()
    return image
       
