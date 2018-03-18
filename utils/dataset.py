from torchvision import transforms
import torch
import cv2
from torch.utils import data

def convert_image_to_tensor(image):
    return torch.from_numpy(image).type(torch.FloatTensor)


def resize_image(image):
    return cv2.resize(image, (128, 128))


def convert_to_CHW(tensor):
    return tensor.permute(2, 0, 1)


def create_dataloader(images, masks, is_train, batch_size=16, shuffle=True):
    ds = NucleiDataset(images, masks, is_train=is_train)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

class NucleiDataset(data.Dataset):
    def __init__(self, input_image, stacked_masks, is_train):
        self.X = input_image
        self.y = stacked_masks
        self.input_transform = transforms.Compose([
            resize_image,
            convert_image_to_tensor,
            convert_to_CHW,
        ])
        self.mask_transform = transforms.Compose([
            resize_image,
            convert_image_to_tensor
        ])
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {
            "input": self.input_transform(self.X[i]),
            "mask": self.mask_transform(self.y[i])
        }
