import torch 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import os 
import matplotlib.pyplot as plt
import numpy as np
import random 

class CityScapeDataset(Dataset):
    def __init__(self, data_root, label_root, city_names=None, split='train', transform=None, rate=0.8, seed=None):
        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform 
        
        
        self.path = os.path.join(self.data_root, 'train')
        self.label_path = os.path.join(self.label_root, 'train')
               
        self.img_fns = []
        self.mask_fns = []

        if seed != None:
            self.rnd = random.Random(seed)
        else:
            self.rnd = random.Random()

        if city_names is not None:
            for c in city_names:
                if c in os.listdir(self.path):
                    if 'foggy' in self.path:
                        self.img_fns += [os.path.join(self.path, c, f) for f in sorted(os.listdir(os.path.join(self.path, c))) if f.endswith('.png') and 'beta_0.02' in f]
                    else:
                        self.img_fns += [os.path.join(self.path, c, f) for f in sorted(os.listdir(os.path.join(self.path, c))) if f.endswith('.png')]

                    self.mask_fns += [os.path.join(self.label_path, c, f) for f in sorted(os.listdir(os.path.join(self.label_path, c))) if f.endswith('.png') and 'labelIds' in f]
        else:
            for c in os.listdir(self.path):
                if 'foggy' in self.path:
                    self.img_fns += [os.path.join(self.path, c, f) for f in sorted(os.listdir(os.path.join(self.path, c))) if f.endswith('.png') and 'beta_0.02' in f]
                else:
                    self.img_fns += [os.path.join(self.path, c, f) for f in sorted(os.listdir(os.path.join(self.path, c))) if f.endswith('.png')]

                self.mask_fns += [os.path.join(self.label_path, c, f) for f in sorted(os.listdir(os.path.join(self.label_path, c))) if f.endswith('.png') and 'labelIds' in f]

        c = list(zip(self.img_fns, self.mask_fns))
        self.rnd.shuffle(c)
        self.img_fns, self.mask_fns = zip(*c)

        if split == 'train':
            self.img_fns = self.img_fns[:int(rate * len(self.img_fns))]
            self.mask_fns = self.mask_fns[:int(rate * len(self.mask_fns))]
        elif split == 'test':
            self.img_fns = self.img_fns[int(rate * len(self.img_fns)):]
            self.mask_fns = self.mask_fns[int(rate * len(self.mask_fns)):]

        
    def __len__(self):
        return len(self.img_fns)
    
    def __getitem__(self, idx):
                
        image = Image.open(self.img_fns[idx])
        mask = Image.open(self.mask_fns[idx])

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0 
        mask = torch.from_numpy(np.array(mask)[:, :, np.newaxis]).permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask).long()
                
        return image, mask
    
class CityScapeDatasetRain(Dataset):
    def __init__(self, data_root, label_root, city_names=None, split='train', transform=None, rate=0.8, seed=None):
        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform 
        
        self.path = os.path.join(self.data_root, split)
        self.label_path = os.path.join(self.label_root, split)
               
        self.img_fns = []
        self.mask_fns = []

        if seed != None:
            self.rnd = random.Random(seed)
        else:
            self.rnd = random.Random()

        if split=='test':
            self.path = os.path.join(self.data_root, 'val')
            self.label_path = os.path.join(self.label_root, 'val')

        if city_names is not None:
            for c in city_names:
                if c in os.listdir(self.path):
                    for f in sorted(os.listdir(os.path.join(self.path, c))):
                        data_id = int(f.split('_')[1])
                        data_id2 = int(f.split('_')[2])
                        label_path_id = os.path.join(self.label_path, c, f'{c}_{data_id:06d}_{data_id2:06d}_gtFine_labelIds.png')
                        self.mask_fns.append(label_path_id)

                        if f.endswith('.png'):
                            self.img_fns.append(os.path.join(self.path, c, f))
                    
        else:
            for c in os.listdir(self.path):
                for f in sorted(os.listdir(os.path.join(self.path, c))):
                    data_id = int(f.split('_')[1])
                    data_id2 = int(f.split('_')[2])
                    label_path_id = os.path.join(self.label_path, c, f'{c}_{data_id:06d}_{data_id2:06d}_gtFine_labelIds.png')
                    self.mask_fns.append(label_path_id)

                    if f.endswith('.png'):
                        self.img_fns.append(os.path.join(self.path, c, f))
        
        c = list(zip(self.img_fns, self.mask_fns))
        self.rnd.shuffle(c)
        self.img_fns, self.mask_fns = zip(*c)

        if split == 'train':
            self.img_fns = self.img_fns[:int(rate * len(self.img_fns))]
            self.mask_fns = self.mask_fns[:int(rate * len(self.mask_fns))]
        elif split == 'test':
            self.img_fns = self.img_fns[int(rate * len(self.img_fns)):]
            self.mask_fns = self.mask_fns[int(rate * len(self.mask_fns)):]
            

        print(len(self.img_fns), len(self.mask_fns))

        
    def __len__(self):
        return len(self.img_fns)
    
    def __getitem__(self, idx):
                
        image = Image.open(self.img_fns[idx])
        mask = Image.open(self.mask_fns[idx])

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0 
        mask = torch.from_numpy(np.array(mask)[:, :, np.newaxis]).permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask).long()
                
        return image, mask
    
class CityScapeDatasetWithinCitySplit(Dataset):
    def __init__(self, data_root, label_root, city_names=None, split='train', transform=None, rate=0.8, seed=None):
        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform 

        # splits = os.listdir(data_root)

        splits = ['train']
        self.img_fns = []
        self.mask_fns = []

        if seed != None:
            self.rnd = random.Random(seed)
        else:
            self.rnd = random.Random()

        for s in splits:
            self.path = os.path.join(self.data_root, s)
            self.label_path = os.path.join(self.label_root, s)
            
            if city_names is not None:
                for c in city_names:
                    if c in os.listdir(self.path):
                        self.img_fns += [os.path.join(self.path, c, f) for f in sorted(os.listdir(os.path.join(self.path, c))) if f.endswith('.png')]
                        self.mask_fns += [os.path.join(self.label_path, c, f) for f in sorted(os.listdir(os.path.join(self.label_path, c))) if f.endswith('.png') and 'labelIds' in f]

            else:
                for c in os.listdir(self.path):
                    self.img_fns += [os.path.join(self.path, c, f) for f in sorted(os.listdir(os.path.join(self.path, c))) if f.endswith('.png')]
                    self.mask_fns += [os.path.join(self.label_path, c, f) for f in sorted(os.listdir(os.path.join(self.label_path, c))) if f.endswith('.png') and 'labelIds' in f]

        
        c = list(zip(self.img_fns, self.mask_fns))
        self.rnd.shuffle(c)
        self.img_fns, self.mask_fns = zip(*c)


        if split == 'train':
            self.img_fns = self.img_fns[:int(rate*len(self.img_fns))]
            self.mask_fns = self.mask_fns[:int(rate*len(self.mask_fns))]
        elif split == 'test':
            self.img_fns = self.img_fns[int(rate*len(self.img_fns)):]
            self.mask_fns = self.mask_fns[int(rate*len(self.mask_fns)):]

        
        
    def __len__(self):
        return len(self.img_fns)
    
    def __getitem__(self, idx):
                
        image = Image.open(self.img_fns[idx])
        mask = Image.open(self.mask_fns[idx])

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0 
        mask = torch.from_numpy(np.array(mask)[:, :, np.newaxis]).permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask).long()
            
        return image, mask


def get_city_task_splits(data_root, task_num):

    # splits = os.listdir(data_root)
    splits = ['train']
    cities = []


    for s in splits:
        cities += [c for c in os.listdir(os.path.join(data_root, s)) if c not in cities]

    random.shuffle(cities)
    
    city_per_task = len(cities) // task_num
    city_splits = []
    for i in range(task_num):
        city_splits.append(cities[i*city_per_task:min((i+1)*city_per_task, len(cities))])
    
    return city_splits


if __name__ == '__main__':
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((256, 512))
    ])
    
    home_path = os.path.expanduser('~')
    data_root = os.path.join(home_path, 'data', 'leftImg8bit')
    label_root = os.path.join(home_path, 'data', 'gtFine')

    get_city_task_splits(data_root, 6)

    # city_names = ['aachen', 'bochum', 'bremen', 'cologne']

    # train_dataset = CityScapeDataset(data_root=data_root, label_root=label_root, transform=transform, city_names=city_names, split='train')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # for i, data in enumerate(train_loader):
    #     image, mask = data
    #     print(image.shape, mask.shape, mask.max(), mask.min())

    #     print(mask.unique())

    #     #create subplots and show case each image label pair in two rows
    #     f, axarr = plt.subplots(2, 4, figsize=(30, 15))
    #     for j in range(4):
    #         axarr[0, j].imshow(image[j].permute(1, 2, 0))
    #         axarr[1, j].imshow(mask[j].permute(1, 2, 0)[:, :, 0])
        
    #     plt.savefig('datalaoder_test.png')
    #     break

