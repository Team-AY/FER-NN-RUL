# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
import pandas as pd
import random

from utils import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        print("1")
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        
        print("2")
        df = pd.read_csv(args.label_path, sep=' ', header=None)

        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('Training')]
        else:
            dataset = df[df[name_c].str.startswith('Test')]

        # notice the raf-db label starts from 1 while label of other dataset starts from 0
        print("3")
        print(dataset)
        self.label = dataset.iloc[:, label_c].values
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [filp_image, add_g]
        self.file_paths = []
        print(images_names)
        for f in images_names:
            f = f.split(".")[0]
            f += '.jpg'
            file_name = os.path.join(self.raf_path, 'images', f)
            self.file_paths.append(file_name)
            

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = image[:, :, ::-1]

        if self.phase == 'train':

            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx