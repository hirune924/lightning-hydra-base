from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import cv2
from hydra.utils import to_absolute_path
import os
import albumentations as A
import pandas as pd
from utils.utils import load_obj
import torch

def get_dataset(cfg: DictConfig) -> dict:
    df = pd.read_csv(to_absolute_path(os.path.join(cfg.data_dir, "labels.csv")))
    test_df = pd.read_csv(to_absolute_path(os.path.join(cfg.data_dir, "sample_submission.csv")))

    df['label'] = df.groupby(['breed']).ngroup()
    breed2id_dict = df[['breed','label']].sort_values(by='label').set_index('breed').to_dict()['label'] 
    id2breed_dict = df[['breed','label']].sort_values(by='label').set_index('label').to_dict()['breed']  

    kf = load_obj(cfg.cv_split.class_name)(**cfg.cv_split.params)
    for fold, (train_index, val_index) in enumerate(kf.split(df.values, df["breed"])):
        df.loc[val_index, "fold"] = int(fold)
    df["fold"] = df["fold"].astype(int)

    train_df = df[df["fold"] != cfg.target_fold]
    valid_df = df[df["fold"] == cfg.target_fold]

    train_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.train]
    train_augs = A.Compose(train_augs_list)

    valid_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.train]
    valid_augs = A.Compose(valid_augs_list)

    train_dataset = DogBreedDataset(train_df, cfg.data_dir, transform=train_augs)
    valid_dataset = DogBreedDataset(valid_df, cfg.data_dir, transform=valid_augs)

    return {"train": train_dataset, "valid": valid_dataset}
    

class DogBreedDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = to_absolute_path(os.path.join(self.data_dir, "train/" + self.data.loc[idx, "id"] + "." + "jpg"))
        # [TODO] 画像読み込みをpytorch nativeにしたい
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = torch.from_numpy(image["image"].transpose(2, 0, 1))
        label = self.data.loc[idx, "label"]
        return image, label