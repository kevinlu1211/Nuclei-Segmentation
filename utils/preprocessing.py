import pickle
import matplotlib.image as mpimg
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold

def get_masks(folder_path):
    masks_paths = glob(f"{folder_path}/masks/*")
    return [mpimg.imread(p) for p in masks_paths]

def get_image(folder_path):
    image_path, = glob(f"{folder_path}/images/*")
    return mpimg.imread(image_path)[:, :, :3]

def get_shape(image):
    return image.shape

def stack_masks(masks):
    if len(masks) == 0:
        return []
    stacked_mask = np.zeros_like(masks[0])
    for mask in masks:
        stacked_mask += mask
    stacked_mask = stacked_mask > 0
    return stacked_mask.astype(np.uint8)

def load_data(load_path):
    data = pickle.load(open(load_path, "rb"))
    return data

def create_data(data_path):
    image_folder_paths = glob(f"{data_path}/*")
    all_ids = np.array([path.split("/")[-1] for path in image_folder_paths])
    all_images = np.array([get_image(path) for path in image_folder_paths])
    all_shapes = np.array([get_shape(image) for image in all_images])

    all_masks = []
    for path in tqdm(image_folder_paths):
        all_masks.append(get_masks(path))

    all_stacked_masks = []
    for masks in tqdm(all_masks):
        all_stacked_masks.append(stack_masks(masks))
    all_stacked_masks = np.array(all_stacked_masks)

    data = []
    for id, image, mask, shape in zip(all_ids, all_images, all_stacked_masks, all_shapes):
        data.append(dict(id=id, image=image, mask=mask, shape=shape))

    return np.array(data)

def train_val_split(data, n_splits=8):
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_idx, val_idx in kf.split(data):
        train = data[train_idx]
        val = data[val_idx]
        break

    return train, val

