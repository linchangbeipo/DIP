import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

def normalize(image, mean, std):
    mean = torch.tensor(mean, dtype=image.dtype)
    std = torch.tensor(std, dtype=image.dtype)
    image = (image - mean[:, None, None]) / std[:, None, None]
    
    return image

def random_crop(im1, im2, crop_size):
    height, width = im1.shape[:2]
    crop_height, crop_width = crop_size

    if height < crop_height or width < crop_width:
        raise ValueError("Crop size is larger than the image size.")

    x = np.random.randint(0, width - crop_width + 1)
    y = np.random.randint(0, height - crop_height + 1)

    return im1[y:y+crop_height, x:x+crop_width], im2[y:y+crop_height, x:x+crop_width]

class PairedDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        self.file = list_file
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        image_rgb = img_color_semantic[:, :256]
        image_semantic = img_color_semantic[:, 256:]
        if (np.random.rand() < 0.5):
            image_rgb = cv2.flip(image_rgb, 1)
            image_semantic = cv2.flip(image_semantic, 1)
        image_rgb, image_semantic = random_crop(image_rgb, image_semantic, (224, 224))

        # Convert the image to a PyTorch tensor
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float()/255.0 * 2.0-1.0
        image_semantic = torch.from_numpy(image_semantic).permute(2, 0, 1).float()/255.0 * 2.0-1.0
        if (self.file == 'train.txt'):
            image_rgb = normalize(image_rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image_semantic = normalize(image_semantic, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image_rgb, image_semantic