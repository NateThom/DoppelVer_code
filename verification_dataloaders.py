from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms.functional as TF
import pandas as pd

# TODO: learn tensorflow and huggingface dataloaders and then add them

class Dataloaders:
    # options for api_name are 1) pytorch, 2) tensorflow, 3) huggingface, api_number should be 1-3
    def __init__(self, api_number, path_to_datasets, dataset, protocol_name, split, transform):
        path_to_dataset = f"{path_to_datasets}{dataset}/"
        path_to_protocol = f"{path_to_datasets}{protocol_name}"
        
        options = [i+1 for i in range(3)]
        if api_number not in options:
            print(f"bad api number, given number:{api_number}, accepted numbers {options}")
        self.dataloader =   self.pytorch_verification_dataloader(path_to_dataset, path_to_protocol, split, transform
                                                               ) if api_number == 1 else \
                            self.tensorflow_verification_dataloader(path_to_dataset, path_to_protocol, split, transform
                                                               ) if api_number == 2 else \
                            self.huggingface_verification_dataloader(path_to_dataset, path_to_protocol, split, transform)

    class pytorch_verification_dataloader(Dataset):
        def __init__(
        self,
        path_to_classes,
        path_to_protocol,
        split="training",
        transform=None,
        ):
        
            self.split = split
            self.path_to_classes = path_to_classes

            self.protocol = pd.read_csv(path_to_protocol)
            self.protocol = self.protocol[self.protocol['SPLIT'] == self.split]

            # If there are any transform functions to be called, store them
            self.transform = transform

        def __len__(self):
            return self.protocol.shape[0]

        def __getitem__(self, idx):

            sample = self.protocol.iloc[idx]

            path1 = f"{self.path_to_classes}{sample['INDIVIDUAL_1']}/{sample['IMAGE_1']}"
            path2 = f"{self.path_to_classes}{sample['INDIVIDUAL_2']}/{sample['IMAGE_2']}"
            image1 = torchvision.io.read_image(path1)
            image2 = torchvision.io.read_image(path2)

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            
            image1 = TF.convert_image_dtype(image1, torch.float32)
            image2 = TF.convert_image_dtype(image2, torch.float32)

            # return pathA, pathP, pathN
            return image1, image2, sample['LABEL']
        
    class tensorflow_verification_dataloader(Dataset):
        def __init__(
        self,
        path_to_classes,
        path_to_protocol,
        split="training",
        transform=None,
        ):
        
            self.split = split
            self.path_to_classes = path_to_classes

            self.protocol = pd.read_csv(path_to_protocol)
            self.protocol = self.protocol[self.protocol['SPLIT'] == self.split]

            # If there are any transform functions to be called, store them
            self.transform = transform
    
    class huggingface_verification_dataloader(Dataset):
        def __init__(
        self,
        path_to_classes,
        path_to_protocol,
        split="training",
        transform=None,
        ):
        
            self.split = split
            self.path_to_classes = path_to_classes

            self.protocol = pd.read_csv(path_to_protocol)
            self.protocol = self.protocol[self.protocol['SPLIT'] == self.split]

            # If there are any transform functions to be called, store them
            self.transform = transform