from pathlib import Path
import shutil
import csv
from PIL import Image
import fastdup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

class remove_dups:
    def __init__(self, path_to_doppel_datasets):
        # TODO: Much code goes here
        self.threshold = 0.90
        self.model_name = "vanilla_classification"

        self.path_to_doppel_dataset = path_to_doppel_datasets
        self.original_path = Path(f"{self.path_to_doppel_dataset}CCA/")
        self.destination_path = Path(f"{self.path_to_doppel_dataset}CCA_noDuplicates_noOutliers/")
        self.duplicates_path = Path(f"{self.path_to_doppel_dataset}CCA_duplicates_outliers/")

        # creates duplicates and noDuplicates folders
        self.create_folders()
        # creates list where each sample is an image, and its path, for all images in the dataset
        self.classification_df = self.list_filename_class()
        # finds the distance between all images
        self.similarity_df, self.outliers_df = self.run_fastdup()

        # finds the images to delete
        both_list_to_del = self.delete_imgs_same_and_diff_class()

        self.list_to_del = both_list_to_del[0]
        self.outliers_list = both_list_to_del[1]
        self.list_to_del_w_extra_info = both_list_to_del[2]

        # adds all duplicate images to CCA_duplicates, removes all duplicates from CCA_noDuplicates
        self.remove_duplicate_images()

    # clears the save locations if they already exists and creates them
    def create_folders(self):
        if self.destination_path.exists():
            shutil.rmtree(str(self.destination_path))
        self.destination_path.mkdir()

        if self.duplicates_path.exists():
            shutil.rmtree(str(self.duplicates_path))
        self.duplicates_path.mkdir()

    # creates a list of all images and their filepaths
    def list_filename_class(self):
        classification_list = []
        for id in tqdm(self.original_path.iterdir()):
            current_id_string = str(id).split("/")[-1]
            if not self.destination_path.joinpath(current_id_string).exists():
                self.destination_path.joinpath(current_id_string).mkdir()
            for image in id.iterdir():
                file_destination = self.destination_path.joinpath(current_id_string, image.name)
                file_destination.write_bytes(image.read_bytes())
                classification_list.append([str(file_destination), current_id_string])

        return pd.DataFrame(classification_list, columns=["filename", "label"])

    # Run fastdup on the whole dataset, returning similar dataframes and outlier images dataframe
    def run_fastdup(self):

        output_path = f"{self.path_to_doppel_dataset}fastdup_findAndRemoveDups_{self.model_name}_{self.threshold}/"
        fd = fastdup.create(work_dir=output_path, input_dir=self.destination_path)
        if not Path(output_path).exists():
            if self.model_name.split("_")[0] == "vanilla":
                fd.run(annotations=self.classification_df, threshold=self.threshold)
            else:
                fd.run(annotations=self.classification_df, model_path=self.model_name.split("_")[0], d=390, threshold=self.threshold)

        similarity_df = fd.vis.similarity_gallery()    # create a gallery of similar images
        outliers_df = fd.outliers()

        return similarity_df, outliers_df

    # creates lists of images to remove based off the similarity df and outliers df
    def delete_imgs_same_and_diff_class(self):
        list_outliers = []
        list_to_del = []
        list_to_del_w_extra_info = []
        from_images = []

        id_count_dict = {}


        for row_index, row in enumerate(self.similarity_df.iterrows()):
            row_distances = row[1]["distance"]
            row_label = row[1]["label"]
            row_label_2 = row[1]["label2"]
            row_from = row[1]["from"]
            row_tos = row[1]["to"]

            if row_from not in from_images:
                from_images.append(row_from)

            for distance_index, distance in enumerate(row_distances):
                if (distance>=.92) and (row_label[distance_index]==row_label_2[distance_index]) and (row_tos[distance_index] not in from_images) and (row_tos[distance_index] not in list_to_del):
                # if (distance>=1.0) and (row_label[distance_index]!=row_label_2[distance_index]) and (row_tos[distance_index] not in from_images) and (row_tos[distance_index] not in list_to_del):
                    if not row_label[distance_index] in id_count_dict.keys():
                        id_count_dict[row_label[distance_index]] = 1
                    else:
                        id_count_dict[row_label[distance_index]] += 1

                    list_to_del_w_extra_info.append([row_from, row_tos[distance_index], row_label[distance_index], row_label_2[distance_index], distance])
                    list_to_del.append(row_tos[distance_index])

        for row_index, row in enumerate(self.outliers_df.iterrows()):
            row_distances = row[1]["distance"]
            row_path = row[1]["filename_outlier"]
            if (row_distances<=.7):
                list_outliers.append([row_distances, row_path])

        return list_to_del, list_outliers, list_to_del_w_extra_info

    # displays images
    def display_duplicate_images(self):
        for img1, img2, label, label2, distance in self.list_to_del_w_extra_info:
            if distance >=.95:
                continue

            img_A = mpimg.imread(img1)
            img_B = mpimg.imread(img2)

            # display images
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(img_A)
            ax[1].imshow(img_B)
            ax[0].set_title(f"{label}\n{img1.split('/')[-2:]}")
            ax[1].set_title(f"{label2}\n{img2.split('/')[-2:]}")
            fig.suptitle(distance)
            plt.show()

    # adds duplicate and outlier images to the duplicates_outliers folder, removes them from noDuplicates_noOutliers
    def remove_duplicate_images(self):
        # for img_from, img_to, label in list_to_del:

        for row in self.outliers_list:
            curr_file_source_path = row[1]
            label = curr_file_source_path.split("/")[-2]
            img_name = curr_file_source_path.split("/")[-1]
            curr_label_path = self.duplicates_path.joinpath(label)
            if not curr_label_path.exists():
                curr_label_path.mkdir()

            curr_file_source_path = Path(curr_file_source_path)
            curr_file_destination_path = curr_label_path.joinpath(img_name)
            
            curr_file_destination_path.write_bytes(curr_file_source_path.read_bytes())
            # print(curr_file_source_path)
            curr_file_source_path.unlink()
        for img_to in self.list_to_del:
            label = img_to.split("/")[-2]
            curr_label_path = self.duplicates_path.joinpath(label)
            if not curr_label_path.exists():
                curr_label_path.mkdir()
            
            img_name = img_to.split("/")[-1]
            curr_file_source_path = Path(img_to)
            curr_file_destination_path = curr_label_path.joinpath(img_name)
            
            curr_file_destination_path.write_bytes(curr_file_source_path.read_bytes())
            # print(curr_file_source_path)
            curr_file_source_path.unlink()