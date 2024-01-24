import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

from Model.Model import Modified_VGG19
from Functions.ImageOps import *
from tqdm import tqdm

from IPython import embed

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    ImagePath = (Path(__file__) / '..' / 'Images' / 'pebbles.jpg').resolve()

    model = Modified_VGG19(device=device)

    image_data = read_image(ImagePath)
    white_data = create_white_noise_image(image_data)

    embed()



