import os
import torch
import tifffile as tiff
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Custom transform for min-max normalization
class Normalize(object):
    def __call__(self, image):
        min_val = torch.min(image)
        max_val = torch.max(image)
        image = (image - min_val) / (max_val - min_val)
        return image

# Define the transform pipeline with ToTensor and MinMaxNormalize
transform = transforms.Compose([
    transforms.ToTensor(),
    #Normalize(),
])

class DataGen(Dataset):
    def __init__(self, data_path, label_path, transform=transform):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.images = os.listdir(data_path)
        self.labels = os.listdir(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.images[idx])
        label_name = os.path.join(self.label_path, self.labels[idx])
        image = Image.open(img_name)
        label = tiff.imread(label_name)
        label = label.squeeze(0)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            label = label.squeeze(0)
            label[label > 0] = 1
            label[label < 0] = 0
        return image.cuda(), label.cuda(), self.images[idx], self.labels[idx]


class DataGenAll(Dataset):
    def __init__(self, data_path, label_path, test_ids, transform=transform, mode = 'train'):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        if mode =='train':
            self.images = [img for img in os.listdir(data_path) if img[:4] not in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:4] not in test_ids]# Print the lengths for debugging
            #print(f"Number of images: {len(self.images)}")
            #print(f"Number of labels: {len(self.labels)}")
        else:
            self.images = [img for img in os.listdir(data_path) if img[:4] in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:4] in test_ids]# Print the lengths for debugging
            #print(f"Number of images: {len(self.images)}")
            #print(f"Number of labels: {len(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(idx)
        img_name = os.path.join(self.data_path, self.images[idx])
        #print(img_name)
        label_name = os.path.join(self.label_path, self.labels[idx])
        image = Image.open(img_name)
        label = tiff.imread(label_name)
        label = label.squeeze(0)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            label = label.squeeze(0)
            label[label>0] = 1  # Assuming the class values are 255, 128, 0 initially
            #label[label == 128] = 2
            label[label<0] = 0
            #print(label.max(), image.max())
            #print(image.shape, label.shape)
        return image.cuda(), label.cuda(), self.images[idx], self.labels[idx]

