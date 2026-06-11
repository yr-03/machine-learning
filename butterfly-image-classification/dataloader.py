import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, cache_to_ram=True):
        """
        Args:
            dataframe: dataframe csv file with annotations
            img_dir: Directory with all the images
            transform (function): Optional transform to be applied on a sample
            cache_to_ram (bool): If True, pre-loads and pre-transforms all images into memory
        """
        label_encoder = LabelEncoder()
        self.annotations = dataframe.reset_index(drop=True)
        self.annotations['encoded_label'] = label_encoder.fit_transform(self.annotations["label"])
        self.img_dir = img_dir
        self.transform = transform
        self.mappings = {index: label for index, label in enumerate(label_encoder.classes_)}
        
        self.cache_to_ram = cache_to_ram
        self.cached_images = []
        self.cached_labels = []

        # If caching is enabled, process the entire dataset right now
        if self.cache_to_ram:
            print(f"Pre-loading and transforming {len(self.annotations)} images into RAM...")
            
            for idx in tqdm(range(len(self.annotations))):
                # 1. Resolve image path and load
                img_id = self.annotations.iloc[idx, 0]
                img_path = os.path.join(self.img_dir, img_id)
                image = Image.open(img_path).convert("RGB")

                # 2. Extract and cast the label tensor
                label = int(self.annotations.iloc[idx, 2])
                label_tensor = torch.tensor(label, dtype=torch.long)

                # 3. Apply the transforms right now, once and for all
                if self.transform:
                    image = self.transform(image)

                # 4. Store the processed tensor directly in system memory
                self.cached_images.append(image)
                self.cached_labels.append(label_tensor)
                
            print("Dataset caching complete! Your data pipeline is now completely running in RAM.")

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        if self.cache_to_ram:
            # CPU completely skips disk I/O, PIL loading, and resizing math.
            # It just pulls the pre-built tensor directly from RAM instantly.
            return self.cached_images[index], self.cached_labels[index]
        else:
            # Fallback slow path if you ever turn off caching
            img_id = self.annotations.iloc[index, 0]
            img_path = os.path.join(self.img_dir, img_id)
            image = Image.open(img_path).convert("RGB")

            label = int(self.annotations.iloc[index, 2])
            label = torch.tensor(label, dtype=torch.long)

            if self.transform:
                image = self.transform(image)

            return image, label
    
    def get_mappings(self):
        return self.mappings