
import random
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from time import time

from typing import Tuple
from typing import List

import torch
from  torch.utils.data import Dataset   # from torch.data.dataset import Dataset
from torch.utils.data import DataLoader # from torch.data.dataset import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from zipfile import ZipFile
from h5py import File
from constant import *

# getting csv data files
with ZipFile( DATA_DIR, mode='r') as zip_file:
     with zip_file.open( train_csv_path ) as csv_file:
          train_df = pd.read_csv( csv_file, low_memory=False)  # we set low_memory=False to supress the dtype warning
     with zip_file.open( test_csv_path ) as csv_file:
          test_df = pd.read_csv( csv_file, low_memory=False)
#print(train_df.tail())
# getting hdf data files

df=train_df

df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)

# Data ratio -> positive:negative = 1:25
df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*20, :]])
print("filtered>", df.shape, df.target.sum(), df["patient_id"].unique().shape)

df = df.reset_index(drop=True)
print("df.shape, # of positive cases, # of patients")
print("original>", df.shape, df.target.sum(), df["patient_id"].unique().shape)
from zipfile import ZipFile
from h5py import File




with ZipFile( DATA_DIR, mode='r') as zip_file:
     with zip_file.open( train_images_hdf5_path) as hdf_file:
         hdf1 = File(hdf_file,  mode='r')
'''
         with open("data.pkl", "wb") as pkl_file:
             h5pickle.dump(hdf, pkl_file)
         # Unpickle the h5py object
         with open("data.pkl", "rb") as f:
             hdf = h5pickle.load(f)
'''
#////////////////////////////////////////////////////////////////////////////////////////////
import zipfile
import tempfile
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
import os

# Function to extract HDF5 file from ZIP archive
def extract_hdf5_from_zip(zip_path, hdf5_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Create a temporary file to store the extracted HDF5 file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5') as temp_file:
            temp_file.write(zip_ref.read(hdf5_filename))
            temp_file_path = temp_file.name
    return temp_file_path

# Extract HDF5 file
hdf= extract_hdf5_from_zip(DATA_DIR, train_images_hdf5_path)
print(hdf)
#//////////////////////////////////////////////////////////////////////////////
df=train_df
# Feature engineering
df['lesion_size_ratio'] = df['tbp_lv_minorAxisMM'] / df['clin_size_long_diam_mm']
df['color_uniformity'] = df['tbp_lv_color_std_mean'] / df['tbp_lv_radial_color_std_max']
df['3d_position_distance'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)

num_feat:List[str] = ['age_approx', 'lesion_size_ratio',
                                'color_uniformity',   'tbp_lv_Lext',
                                'tbp_lv_eccentricity', '3d_position_distance']

def clean_data(df:pd.DataFrame,
               scaler:bool=True)->pd.DataFrame:

                # Replace infinite values with NaN
                df[num_feat] = df[num_feat].replace([np.inf, -np.inf], np.nan)

                # Handle missing values (if any) by filling them with the mean of the column
                df[num_feat] = df[num_feat].fillna(df[num_feat].mean())

                # Scale numerical features
                if scaler:
                       from sklearn.preprocessing import StandardScaler
                       scaler = StandardScaler()
                       df[num_feat] = scaler.fit_transform(df[num_feat])

                       print("Feature engineering and scaling complete.")

                return df




#df=clean_data(train_df)
#numerical_features = train_df.select_dtypes(include=['number']).columns
#print(df.head())





# Define image transformer
data_transforms = {
    'train': A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758],
            max_pixel_value=255.0),
        ToTensorV2(),
    ]),

    'valid': A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758],
            max_pixel_value=255.0),
        ToTensorV2(),
    ])
}

class ISICDataset(Dataset):
    def __init__(self, dataframe, feat, hdf, transforms=None, is_training=True):
        self.df = dataframe
        self.file_path = File(hdf, mode="r")
        self.transforms = transforms
        self.is_training = is_training
        self.df_positive = self.df[self.df['target'] == 1].reset_index()
        self.df_negative = self.df[self.df['target'] == 0].reset_index()
        self.isic_ids_positive = self.df_positive['isic_id'].values
        self.isic_ids_negative = self.df_negative['isic_id'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.metadata = self.df[feat].values

    def __len__(self):
        return len(self.df_positive) * 2 if self.is_training else len(self.df)

    def __getitem__(self, idx):
        if self.is_training:
            is_positive = random.random() >= 0.5
            df_subset = self.df_positive if is_positive else self.df_negative
            isic_ids = self.isic_ids_positive if is_positive else self.isic_ids_negative
            targets = self.targets_positive if is_positive else self.targets_negative
            idx = idx % len(df_subset)
            target = targets[idx]
            isic_id = isic_ids[idx]
        else:
            row = self.df.iloc[idx]
            target = row['target']
            isic_id = row['isic_id']

        metadata = self.metadata[idx]

        image = np.array(Image.open(BytesIO(self.file_path[isic_id][()])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)["image"]

        return {'image': image,
                'target': target,
                'metadata': metadata}


#print(hdf)
