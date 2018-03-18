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
    return mpimg.imread(image_path)

def stack_masks(masks):
    stacked_mask = np.zeros_like(masks[0])
    for mask in masks:
        stacked_mask += mask
    stacked_mask = stacked_mask > 0
    return stacked_mask.astype(np.uint8)

def load_data(image_load_path, mask_load_path):
    imgs, masks = pickle.load(open(image_load_path, "rb")), pickle.load(open(mask_load_path, "rb"))
    return imgs, masks

def create_data(data_path):
    image_folders = glob(f"{data_path}/*")
    all_images = np.array([get_image(path) for path in image_folders])

    all_masks = []
    for path in tqdm(image_folders):
        all_masks.append(get_masks(path))

    all_stacked_masks = []
    for masks in tqdm(all_masks):
        all_stacked_masks.append(stack_masks(masks))
    all_stacked_masks = np.array(all_stacked_masks)

    return all_images, all_stacked_masks

def train_val_split(images, masks, n_splits=8):
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_idx, test_idx in kf.split(images):
        train_images, val_images = images[train_idx], images[test_idx]
        train_masks, val_masks = masks[train_idx], masks[test_idx]
        break

    return train_images, train_masks, val_images, val_masks

