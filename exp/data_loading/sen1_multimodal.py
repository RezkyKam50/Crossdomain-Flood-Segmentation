import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import os
import numpy as np
import rasterio


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func
  
    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])
  
    def __len__(self):
        return len(self.data_list)

S1_BANDS = (0, 1)
S2_BANDS = (1, 2, 3, 8, 11, 12)
INPUT_SIZE = 224

S1_MEANS = [
    0.5772618749337688, 
    0.6414101609712768
]
S1_STDS  = [
    0.042344421839689894, 
    0.04924232261091114
]

S2_MEANS = [
    0.1396034706145535, 
    0.13640611840858108, 
    0.12182284795095258, 
    0.3077482214814722, 
    0.20306476302430385, 
    0.11791660724933717
]

S2_STDS = [
    0.07390945268999569, 
    0.07352482387806976, 
    0.08649366947564742, 
    0.119644255309065, 
    0.09808706133414245, 
    0.0764608365341021
]

DEM_MEAN = 0.015424964308526004
DEM_STD = 0.014092320382345137

def processAndAugmentDualWithAux(data):
    """Process training data with proper modality separation"""
    img_s1, img_s2, label, dem, water_occur = data
    
    # Extract bands and convert
    img_s1 = img_s1.astype(np.float32)[S1_BANDS, :, :]
    img_s2 = img_s2.astype(np.float32)[S2_BANDS, :, :]
    dem = dem.astype(np.float32)[:1, :, :]
    label = label.squeeze().astype(np.int16)
    water_occur = water_occur[:1, :, :].astype(np.float32)  # (1, H, W)
    
    # Convert to tensors
    img_s1 = torch.tensor(img_s1)
    img_s2 = torch.tensor(img_s2)
    dem = torch.tensor(dem)
    label = torch.tensor(label)
    water_occur = torch.tensor(water_occur)
    
    # Normalize each modality separately
    norm_s1 = transforms.Normalize(S1_MEANS, S1_STDS)
    norm_s2 = transforms.Normalize(S2_MEANS, S2_STDS)
    norm_dem = transforms.Normalize([DEM_MEAN], [DEM_STD])
    
    img_s1 = norm_s1(img_s1)
    img_s2 = norm_s2(img_s2)
    dem = norm_dem(dem)
    
    # Apply SAME random crop to all modalities (to keep spatial alignment)
    i, j, h, w = transforms.RandomCrop.get_params(img_s1, (INPUT_SIZE, INPUT_SIZE))
    
    sar_img = F.crop(img_s1, i, j, h, w)        # Stream 1: SAR data (2 channels)
    optical_img = F.crop(img_s2, i, j, h, w)    # Stream 2: Optical data (6 channels)
    elevation_img = F.crop(dem, i, j, h, w)     # Stream 3: Elevation data (1 channel)
    label = F.crop(label, i, j, h, w)
    water_occur = F.crop(water_occur, i, j, h, w)
    
    # Apply SAME random augmentations to all (to keep spatial alignment)
    if random.random() > 0.5:
        sar_img = F.hflip(sar_img)
        optical_img = F.hflip(optical_img)
        elevation_img = F.hflip(elevation_img)
        label = F.hflip(label)
        water_occur = F.hflip(water_occur)
    
    if random.random() > 0.5:
        sar_img = F.vflip(sar_img)
        optical_img = F.vflip(optical_img)
        elevation_img = F.vflip(elevation_img)
        label = F.vflip(label)
        water_occur = F.vflip(water_occur)
    
    return sar_img, optical_img, elevation_img, label, water_occur

def processTestMM(data):
    """Process test/validation data with proper modality separation"""
    img_s1, img_s2, label, dem, water_occur = data
    
    # Extract bands and convert
    img_s1 = img_s1.astype(np.float32)[S1_BANDS, :, :]
    img_s2 = img_s2.astype(np.float32)[S2_BANDS, :, :]
    dem = dem.astype(np.float32)[:1, :, :]
    label = label.squeeze().astype(np.int16)
    water_occur = water_occur[:1, :, :].astype(np.float32)  # (1, H, W)

    # Convert to tensors
    img_s1 = torch.tensor(img_s1)
    img_s2 = torch.tensor(img_s2)
    dem = torch.tensor(dem)
    label = torch.tensor(label)
    water_occur = torch.tensor(water_occur)
    
    # Normalize each modality separately
    norm_s1 = transforms.Normalize(S1_MEANS, S1_STDS)
    norm_s2 = transforms.Normalize(S2_MEANS, S2_STDS)
    norm_dem = transforms.Normalize([DEM_MEAN], [DEM_STD])
    
    sar_img = norm_s1(img_s1)
    optical_img = norm_s2(img_s2)
    elevation_img = norm_dem(dem)
    
    # Extract 4 corner crops for test-time evaluation
    sar_crops = [
        F.crop(sar_img, 0, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(sar_img, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
        F.crop(sar_img, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(sar_img, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    ]
    
    optical_crops = [
        F.crop(optical_img, 0, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(optical_img, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
        F.crop(optical_img, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(optical_img, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    ]
    
    elevation_crops = [
        F.crop(elevation_img, 0, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(elevation_img, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
        F.crop(elevation_img, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(elevation_img, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    ]
    
    labels = [
        F.crop(label, 0, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(label, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
        F.crop(label, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(label, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    ]
    
    water_occurs = [
        F.crop(water_occur, 0, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(water_occur, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
        F.crop(water_occur, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE),
        F.crop(water_occur, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    ]
    
    # Stack crops
    sar_batch = torch.stack(sar_crops)
    optical_batch = torch.stack(optical_crops)
    elevation_batch = torch.stack(elevation_crops)
    labels_batch = torch.stack([l.squeeze() for l in labels])
    water_batch = torch.stack(water_occurs)
    
    return sar_batch, optical_batch, elevation_batch, labels_batch, water_batch

def getArrFlood(fname):
    return rasterio.open(fname).read()

LABEL_DIR = 'data/LabelHand'
IMAGE_DIR_S2 = 'data/S2L1CHand'
IMAGE_DIR_S1 = 'data/S1GRDHand'
WATER_OCCUR_DIR = 'data/JRCWaterHand'
DEM_ELEVATION = 'data/CopernicusDEM'
DATASET_DIR = 'splits'

def load_dual_flood_data_with_aux(path, dataset_type):
    fpath = os.path.join(path, DATASET_DIR, f'flood_{dataset_type}_data.txt')
    
    with open(fpath) as f:
        data_files = []
        for line in f:
            identifier = line.strip()
            if identifier:
                s1_path = os.path.join(path, IMAGE_DIR_S1, f"{identifier}_S1Hand.tif")
                s2_path = os.path.join(path, IMAGE_DIR_S2, f"{identifier}_S2Hand.tif")
                label_path = os.path.join(path, LABEL_DIR, f"{identifier}_LabelHand.tif")
                dem_path = os.path.join(path, DEM_ELEVATION, f"{identifier}_DEM.tif")
                water_path = os.path.join(path, WATER_OCCUR_DIR, f"{identifier}_JRCWaterHand.tif")
                data_files.append((s1_path, s2_path, label_path, dem_path, water_path))
    
    return download_dual_flood_data_with_aux_from_list(data_files)

def download_dual_flood_data_with_aux_from_list(l):
    flood_data = []
    for (s1_path, s2_path, mask_path, dem_path, water_path) in l:
        if not os.path.exists(s1_path):
            raise ValueError(f"S1 file not found: {s1_path}")
        if not os.path.exists(s2_path):
            raise ValueError(f"S2 file not found: {s2_path}")
        if not os.path.exists(mask_path):
            raise ValueError(f"Mask file not found: {mask_path}")
        if not os.path.exists(dem_path):
            raise ValueError(f"DEM file not found: {dem_path}")
        if not os.path.exists(water_path):
            raise ValueError(f"Water occurrence file not found: {water_path}")
        
        arr_s1 = np.nan_to_num(getArrFlood(s1_path))
        arr_s2 = np.nan_to_num(getArrFlood(s2_path))
        arr_y = getArrFlood(mask_path)
        arr_y[arr_y == -1] = 255
        
        arr_dem = np.nan_to_num(getArrFlood(dem_path))
        arr_water = np.nan_to_num(getArrFlood(water_path))

        flood_data.append((arr_s1, arr_s2, arr_y, arr_dem, arr_water))
    
    return flood_data

def get_train_loader_MM(data_path, args):
    train_data = load_dual_flood_data_with_aux(data_path, 'train')
    train_dataset = InMemoryDataset(train_data, processAndAugmentDualWithAux)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True, 
        drop_last=False
    )
    return train_loader


def collate_fn(batch):
    sar_batch = torch.cat([item[0] for item in batch], 0)
    optical_batch = torch.cat([item[1] for item in batch], 0)
    elevation_batch = torch.cat([item[2] for item in batch], 0)
    labels_batch = torch.cat([item[3] for item in batch], 0)
    water_batch = torch.cat([item[4] for item in batch], 0)
    return sar_batch, optical_batch, elevation_batch, labels_batch, water_batch


def get_test_loader_MM(data_path, type):
    valid_data = load_dual_flood_data_with_aux(data_path, type)
    valid_dataset = InMemoryDataset(valid_data, processTestMM)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=3, 
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True, 
        drop_last=False
    )
    return valid_loader

def get_loader_MM(data_path, type, args):
    if type == 'train':
        return get_train_loader_MM(data_path, args)
    else:
        return get_test_loader_MM(data_path, type)