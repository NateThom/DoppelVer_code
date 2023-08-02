import dataCollection
import original_to_CCA
import remove_duplicate_images

# Will create the dataset based on list of names, create head crops of each individual in each image, and remove duplicate images
# Once this is completed, the user will need to manually remove all remaining images within the dataset
# Once the dataset is fully cleaned run protocol_creation.py
# To use this dataset, and protocols, call the premade dataloaders or create your own!

datasets_save_location = "/home/adebolt/Documents/DoppelgangerVerification/"
# run this, once this successfully finishes run ViSE_protocol_creation.ipynb
download_save_location = f"{datasets_save_location}Downloads/"
list_of_names_save_location = f"{datasets_save_location}IndividualGenderCountryOrigin.csv"
number_of_images_to_download = 10
dataCollection.DataCollection(download_save_location, list_of_names_save_location, number_of_images_to_download)

cca_save_location = f"{datasets_save_location}CCA/"
original_to_CCA.CCA_all_images(download_save_location, cca_save_location)

remove_duplicate_images.remove_dups(datasets_save_location)
