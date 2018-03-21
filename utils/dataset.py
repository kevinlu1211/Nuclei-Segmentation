from torchvision import transforms
import torch
import cv2


def convert_image_to_tensor(image):
    return torch.from_numpy(image).type(torch.FloatTensor)


def resize_image(image):
    return cv2.resize(image, (128, 128))


def convert_to_CHW(tensor):
    return tensor.permute(2, 0, 1)


def create_dataloader(data, is_test, batch_size=16, shuffle=True):
    ds = NucleiDataset(data, is_test)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


class NucleiDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.input_transform = transforms.Compose([
            resize_image,
            convert_image_to_tensor,
            convert_to_CHW,
        ])
        self.mask_transform = transforms.Compose([
            resize_image,
            convert_image_to_tensor
        ])
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.input_transform(self.data[i]["image"])
        shape = self.data[i]["shape"]
        id = self.data[i]["id"]
        ret = dict(input=image,
                   id=id,
                   shape=shape)
        if not self.is_test:
            ret.update(dict(mask=self.mask_transform(self.data[i]["mask"])))
        return ret
