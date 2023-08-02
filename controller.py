import os

from dataCollection import DataCollection
import original_to_CCA
import remove_duplicate_images

def main():
    """
    - Collects data and downloads images
    - Converts images to CCA format
    - Removes duplicate images
    - User needs to manually remove remaining images within the dataset
    - Once the dataset is fully cleaned, run protocol_creation.py
    - To use this dataset and protocols, call the premade dataloaders or create your own!
    """
    
    # Define file locations
    # base_dir = "/path/to/doppelver/root/"
    base_dir = "/home/nthom/Documents/DoppelVer/" # Change this to the location of the repository root
    downloads_dir = os.path.join(base_dir, "Downloads/") # Change this to the location to download the images
    demographics_file = os.path.join(base_dir, "IndividualGenderCountryOrigin.csv") # Provide the path to the "IndividualGenderCountryOrigin.csv"
    cca_dir = os.path.join(base_dir, "CCA/")

    # Step 1: Collect data and download images
    number_of_images_to_save = 2  # Change this to the desired number of images per person
    try:
        data_collection = DataCollection(downloads_dir, demographics_file, number_of_images_to_save)
        data_collection.find_on_internet()
        data_collection.rename_all_images()
        correct_count, incorrect_count = data_collection.check_image_count()
        print("Correct image count:", correct_count)
        print("Incorrect image count:", incorrect_count)
        print("Image downloading completed successfully!")
    except Exception as e:
        print(f"An error occurred during image downloading: {e}")
        exit()

    # Step 2: Convert images to CCA format
    try:
        original_to_CCA.CCA_all_images(downloads_dir, cca_dir)
        print("CCA of all downloaded images completed successfully!")
    except Exception as e:
        print(f"An error occurred during the CCA operation: {e}")
        exit()

    # Step 3: Remove duplicate images
    try:
        remove_duplicate_images.RemoveDups(base_dir)
        print("Removal of duplicates completed successfully!")
    except Exception as e:
        print(f"An error occurred during the removal of duplicates: {e}")
        exit()

if __name__ == "__main__":
    main()
