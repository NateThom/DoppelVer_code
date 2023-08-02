import os
from tqdm import tqdm
import pandas as pd
from bing_image_downloader import downloader
from PIL import Image
import shutil


class DataCollection:

    # On init, from list of all identites, find n images of that identity, and rename each found image to be 0-N.jpg, finally check if each individual has the right number of images saved
    def __init__(self, save_location, path_to_names_list, number_of_images_to_save):
        if os.path.isdir(save_location):
            shutil.rmtree(save_location)
        os.mkdir(save_location)
        self.save_location = save_location
        self.path_to_names_list = path_to_names_list
        self.number_of_images_to_save = number_of_images_to_save
        self.list_of_names = pd.read_csv(self.path_to_names_list)['NAME'].values.tolist()
        self.findOnInternet()
        self.rename_images()
        self.check_image_count()


    # for each individual in the list of all individuals, collect n images of that person
    def findOnInternet(self):
        for name in tqdm(self.list_of_names, leave=True):
            downloader.download(name, limit=self.number_of_images_to_save, output_dir=self.save_location, adult_filter_off=True, force_replace=False, timeout=60)

    # for each img in the dataset, rename image to be XXX.jpg, where XXX is the id of the image padded with 0's to the len of the max image count
    def rename_images(self):
        for name in os.listdir(self.save_location):
            imgs = os.listdir(self.save_location+name)
            for id_img, img in enumerate(imgs):
                new_img_name = str(id_img).zfill(len(str(self.number_of_images_to_save-1))) + ".jpg"
                im = Image.open(f"{self.save_location}{name}/{img}")
                im = im.convert("RGB")
                im.save(f"{self.save_location}{name}/{new_img_name}")
                os.remove(f"{self.save_location}{name}/{img}")
                
    # make sure each individual has the correct number of image, print msg otherwise
    def check_image_count(self):
        for name in os.listdir(self.save_location):
            actual_num_images = len(os.listdir(self.save_location+name))
            if not actual_num_images == self.number_of_images_to_save:
                print(f"{name} has {actual_num_images} images, when it needs {self.number_of_images_to_save} images")