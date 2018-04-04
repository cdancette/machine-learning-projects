from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import transforms
import os

def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

transform_driving_image = transforms.Compose([
    transforms.CenterCrop(72),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class CustomDataset:
    def __init__(self, dataset_path):
        self.images = os.path.join(dataset_path, "images")
        with open(os.path.join(dataset_path, "labels.txt"), 'r') as f:
            lines = [l.strip().split() for l in f.readlines()]
            lines = [[f, int(label)] for (f, label) in lines]
            self.labels = lines
        self.transform = transform_driving_image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name, label = self.labels[index]
        return self.transform(load_image(os.path.join(self.images, image_name))), torch.LongTensor([label])


def get_dataloader(dataset_path, batch_size):
    dataset = CustomDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test():
    loader = CustomDataset("dataset2")
    print(len(loader))
    print(loader[0])

if __name__=='__main__':
    test()