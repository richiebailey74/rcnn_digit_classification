import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import Dataset
import os
import json
from PIL import Image


class SVHNDataset(Dataset):
    def __init__(self, json_file, root_dir, img_size=(224, 224), transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            img_size (tuple): Desired output size of the image as (height, width).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        self.root_dir = root_dir
        self.transform = self.transform = Compose([
            Resize(img_size),  # Resize images to 800x800
            ToTensor()           # Then convert them to tensor
        ]) if transform is None else transform  # Default to resizing if no transform provided

        # Preload data
        self.images = []
        self.boxes = []
        self.labels = []

        for ann in annotations:
            img_path = os.path.join(self.root_dir, ann['image_path'])
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            image = self.transform(image) if self.transform else image
            try:
                self.images.append(ToTensor()(image))
            except:
                self.images.append(image)

            boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
            labels = torch.tensor(ann['labels'], dtype=torch.int64)

            # Adjust boxes for the resized image
            scale_x = img_size[0] / original_size[0]
            scale_y = img_size[1] / original_size[1]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            self.boxes.append(boxes)
            self.labels.append(labels)

        # Pad boxes and labels to ensure uniformity
        self.boxes = pad_sequence(self.boxes, batch_first=True, padding_value=-1)
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.boxes[idx], self.labels[idx]
