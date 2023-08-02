import os
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat

from tqdm import tqdm
import pandas as pd
from bing_image_downloader import downloader
import cv2
import shutil


class DataCollection:
    def __init__(self, save_location, path_to_names_list, number_of_images_to_save):
        self.save_location = Path(save_location)
        self.path_to_names_list = path_to_names_list
        self.number_of_images_to_save = number_of_images_to_save
        self.list_of_names = pd.read_csv(self.path_to_names_list)['NAME'].values.tolist()
        
        if self.save_location.exists() and self.save_location.is_dir():
            shutil.rmtree(self.save_location)
        self.save_location.mkdir(parents=True, exist_ok=True)

    def download_images(self, name, limit, adult_filter_off, filter, force_replace, timeout):
        downloader.download(
            name,
            limit=limit,
            output_dir=self.save_location,
            adult_filter_off=adult_filter_off,
            filter=filter,
            force_replace=force_replace,
            timeout=timeout
        )

    def find_on_internet(self):
        func_inputs = zip(
            self.list_of_names,
            [self.number_of_images_to_save] * len(self.list_of_names),
            [False] * len(self.list_of_names),
            ["photo"] * len(self.list_of_names),
            [False] * len(self.list_of_names),
            [1] * len(self.list_of_names)
        )
        with Pool() as p:
            p.starmap(self.download_images, func_inputs)

    def rename_images(self, name):
        images_dir = self.save_location / name
        imgs = os.listdir(images_dir)
        for id_img, img in enumerate(imgs):
            new_img_name = str(id_img).zfill(len(str(self.number_of_images_to_save-1))) + ".jpg"
            image = cv2.imread(str(images_dir / img))
            cv2.imwrite(str(images_dir / new_img_name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            os.remove(str(images_dir / img))

    # for each img in the dataset, rename image to be XXX.jpg, where XXX is the id of the image padded with 0's to the len of the max image count
    def rename_all_images(self):
        with Pool() as p:
            p.map(self.rename_images, self.save_location.iterdir())
            
    # make sure each individual has the correct number of image, print msg otherwise
    def check_image_count(self):
        correct_count = {}
        incorrect_count = {}
        for name in self.save_location.iterdir():
            actual_num_images = len(os.listdir(name))
            if actual_num_images == self.number_of_images_to_save:
                correct_count[name.name] = actual_num_images
            else:
                incorrect_count[name.name] = actual_num_images

        return correct_count, incorrect_count
    
if __name__ == "__main__":
    base_dir = "/home/nthom/Documents/DoppelVer/" # Change this to the location of the repository root
    downloads_dir = os.path.join(base_dir, "Downloads/") # Change this to the location to download the images
    demographics_file = os.path.join(base_dir, "IndividualGenderCountryOrigin.csv") # Provide the path to the "IndividualGenderCountryOrigin.csv"
    number_of_images_to_save = 5  # Change this to the desired number of images per person

    data_collection = DataCollection(downloads_dir, demographics_file, number_of_images_to_save)
    data_collection.find_on_internet()
    data_collection.rename_all_images()
    correct_count, incorrect_count = data_collection.check_image_count()
    print("Correct image count:", correct_count)
    print("Incorrect image count:", incorrect_count)