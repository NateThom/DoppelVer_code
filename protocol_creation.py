import os
import random
import pandas as pd
import csv
from tqdm import tqdm
from pathlib import Path
import numpy as np
import shutil
import fastdup

# Function to load the dataset by creating a dictionary of images for each individual
def load_dataset(path):
    dataset = {}
    # make list of images for each individual in the dataset
    for name in os.listdir(path):
        dataset[name] = os.listdir(path+name)
    return dataset

# Function to print samples to a file in CSV format
def print_samples_to(location, protocol_name, data, count):
    with open(f"{location}{protocol_name}", "a+") as fp:
        # move to end of file
        fp.seek(0, 2)
        if os.stat(f"{location}{protocol_name}").st_size == 0:
            fp.write("ID,INDIVIDUAL_1,IMAGE_1,INDIVIDUAL_2,IMAGE_2,LABEL\n")

        for sample in data:
            sample = sample[0] + "," + sample[1] + "," + sample[2] + "," + sample[3] + "," + str(sample[4])
            count += 1
            fp.write(f"{str(count)},{sample}\n")
        fp.close()
        
    return count

# Recursive function to group names into different splits
def namesPerSplits(name, split, namesPerSplit, numSamplesPerName, pairs, totalSamplesUsed):
    if name not in namesPerSplit["combined"]:
            
            namesPerSplit[split].append(name)
            namesPerSplit["combined"].append(name)
            totalSamplesUsed += numSamplesPerName[name]

            for namePair in pairs[name]:
                namesPerSplit, c = namesPerSplits(namePair, split, namesPerSplit, numSamplesPerName, pairs, 0)
                totalSamplesUsed += c
    return namesPerSplit, totalSamplesUsed

# Class for creating the doppelganger protocols
class protocol_creation:
    def __init__(self, path_to_datasets, dataset):
        self.path_to_pairs_list = f"{path_to_datasets}doppelgangerPairs.csv"
        temp = False
        attempt_counter = 0
        while not temp:
            if os.path.isfile(path_to_datasets+"DoppelgangerProtocol.csv"):
                os.remove(path_to_datasets+"DoppelgangerProtocol.csv")
            if os.path.isfile(path_to_datasets+"ViSEProtocol.csv"):
                os.remove(path_to_datasets+"ViSEProtocol.csv")
            print(attempt_counter)
            doppel_protocol = create_doppelganger_protocol(path_to_datasets, dataset, self.path_to_pairs_list)
            # fix file names before moving on
            doppel_protocol.rename_files()

            ViSE_protocol = create_ViSE_protocol(path_to_datasets, dataset)
            temp = ViSE_protocol.successful_creation
            attempt_counter += 1

# Class for creating the doppelganger protocol
class create_doppelganger_protocol:
    def __init__(self, path_to_datasets, dataset, path_to_pairs_list):
        self.path_to_datasets = path_to_datasets
        self.path_to_dataset = path_to_datasets + dataset + "/"
        self.dataset = load_dataset(self.path_to_dataset)
        self.pairs = pd.read_csv(path_to_pairs_list)
        self.genderCo = self.genderCOList()
        self.genderList = {"F": [], "M": []}
        self.names = list(self.genderCo.keys())
        for name in self.genderCo:
            self.genderList[self.genderCo[name][0]].append(name)

        # generates initial doppel protocol
        self.doppel_protocol_10_fold_cross_splits()
        names_per_split = self.generateCrossValidationSplit()
        self.addSplitsCrossValidation(names_per_split)

    # Function to read the gender and country origin data from a CSV file
    def genderCOList(self):
        with open(self.path_to_datasets + "IndividualGenderCountryOrigin.csv", "r") as fp:
            lines = fp.read().splitlines()
            lines = [line.split(",") for line in lines]
            lines = {line[0]: [line[1], line[2]] for line in lines[1:]}
        return lines
    
    # Function to create the initial doppelganger protocol with 10-fold cross-validation
    def doppel_protocol_10_fold_cross_splits(self):
        print("STARTING CREATION OF DOPPELGANGER PROTOCOL-----------------------------------")
        sampleGenders = {"total": 0, "F": 0, "M": 0}
        totalLabel = 0
        count = 0
        
        # Goal, in pairs, for each key, each other key should be added, then the original key removed from the list
        # This forces each pair of names to only appear once
        pairs = self.pairs.values.tolist()
        temp_pairs = {}
        for pair in pairs:
            if pair[0] not in temp_pairs and pair[1] not in temp_pairs:
                temp_pairs[pair[0]] = {pair[0]: list(self.dataset[pair[0]]), pair[1]: list(self.dataset[pair[1]])}
            elif pair[0] in temp_pairs and pair[1] not in temp_pairs and pair[1] not in temp_pairs[pair[0]]:
                temp_pairs[pair[0]][pair[1]] = list(self.dataset[pair[1]])
            elif pair[1] in temp_pairs and pair[0] not in temp_pairs and pair[0] not in temp_pairs[pair[1]]:
                temp_pairs[pair[1]][pair[0]] = list(self.dataset[pair[0]])
            elif (pair[0] in temp_pairs and pair[1] in temp_pairs) and (pair[1] not in temp_pairs[pair[0]] and pair[0] not in temp_pairs[pair[1]]):
                temp_pairs[pair[0]][pair[1]] = list(self.dataset[pair[1]])

        pairs = temp_pairs

        # Check for identities without any samples and remove them
        pop = []
        for pair in pairs:
            for name in pairs[pair]:
                if len(pairs[pair][name]) == 0:
                    pop.append([pair, name])
        for temp_pairs in pop:
            pairs[temp_pairs[0]].pop(temp_pairs[1])

        samplesToWrite = []
        for name1 in tqdm(pairs):
            used_photo_pairs = []
            for name2 in pairs[name1]:
                current_pair_image_count = 0
                current_pair_attempts = 0
                while current_pair_image_count < 250:
                    current_pair_attempts += 1
                    label = 1 if name1 == name2 else 0

                    photo1 = random.sample(list(self.dataset[name1]), 1)[0]
                    photo2 = random.sample(list(self.dataset[name2]), 1)[0]

                    if photo1+photo2 not in used_photo_pairs:
                        current_pair_image_count += 1
                    else:
                        continue

                    used_photo_pairs.append(name1+photo1+name2+photo2)
                    used_photo_pairs.append(name2+photo2+name1+photo1)

                    totalLabel += label
                    samplesToWrite.append([name1, photo1, name2, photo2, label])
                    sampleGenders["M"] += (1 if name1 in self.genderList["M"] else 0) + (1 if name2 in self.genderList["M"] else 0)
                    sampleGenders["F"] += (1 if name1 in self.genderList["F"] else 0) + (1 if name2 in self.genderList["F"] else 0)
                    sampleGenders["total"] += 2

        count = print_samples_to(self.path_to_datasets, "DoppelgangerProtocol.csv", samplesToWrite, count)

        print(f"{count} Samples Created in DoppelgangerProtocol.csv")
        print("Doppelganger Zero class percent: ", totalLabel / count)
        print("Doppelganger Female Dist(percent of total): ", sampleGenders["F"] / sampleGenders["total"])
        return samplesToWrite
    
    # Function to generate cross-validation splits for the doppelganger protocol
    def generateCrossValidationSplit(self):
        print("STARTING CREATION OF TTV SPLITS----------------------------------")

        dgProtocol = []
        # convert pairs.csv to a dictionary where the keys are all names in the dataset and their pairs are all pairs to that name
        pairs = list(csv.reader(open(f"{self.path_to_datasets}doppelgangerPairs.csv")))[1:]
        p = {}
        for pair in pairs:
            if pair[0] not in p and pair[1] not in p:
                p[pair[0]] = [pair[0], pair[1]]
                p[pair[1]] = [pair[0], pair[1]]
            elif pair[0] in p and pair[1] not in p:
                p[pair[1]] = [pair[0], pair[1]]
                if pair[1] not in p[pair[0]]:
                    p[pair[0]].append(pair[1])
            elif pair[0] not in p and pair[1] in p:
                p[pair[0]] = [pair[0], pair[1]]
                if pair[0] not in p[pair[1]]:
                    p[pair[1]].append(pair[0])
            else:
                if pair[0] not in p[pair[1]]:
                    p[pair[1]].append(pair[0])
                elif pair[1] not in p[pair[0]]:
                    p[pair[0]].append(pair[1])
        pairs = p

        # read the whole doppel protocol and make it a list of samples
        with open(f'{self.path_to_datasets}DoppelgangerProtocol.csv') as csv_file:
            csvReader = csv.DictReader(csv_file)
            for row in csvReader:
                dgProtocol.append({key: row[key] for key in row})

        # make needed variables for later parts
        numSamplesPerName = {name:0 for name in self.names}
        totalSamplesUsed = 0

        # count number of samples per name, if label is 1(name1 == name2) only count that one time
        totalNumSamples = 0
        for sample in dgProtocol:
            if not sample["INDIVIDUAL_1"] == sample["INDIVIDUAL_2"]:
                totalNumSamples += 2
                numSamplesPerName[sample["INDIVIDUAL_1"]] += 1
                numSamplesPerName[sample["INDIVIDUAL_2"]] += 1
            else:
                totalNumSamples += 1
                numSamplesPerName[sample["INDIVIDUAL_1"]] += 1
        
        # sort in decending order
        sortedNames = sorted(numSamplesPerName.items(), key=lambda x:x[1], reverse=True)
        random.shuffle(sortedNames)
        namesPerSplit = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], "combined":[]}

        # add names to the train test val splits starting at the largest and working my way down
        for name in tqdm(sortedNames):
            name = name[0]
            if name not in namesPerSplit["combined"]:
                p = totalSamplesUsed / totalNumSamples
                # first 10% split 0, and so on
                split = int(p*10) if p < 1 else 9

                namesPerSplit, totalSamplesUsed = namesPerSplits(name, split, namesPerSplit, numSamplesPerName, pairs, totalSamplesUsed)

        return namesPerSplit
    
    # Function to add the generated splits to the doppelganger protocol
    def addSplitsCrossValidation(self, namesPerSplit):
        protocol_name = 'DoppelgangerProtocol'
        print(f"ADDING SPLITS TO {protocol_name}---------------------------------")
        with open(f'{self.path_to_datasets}{protocol_name}.csv', 'r') as icsv:
            with open(f'{self.path_to_datasets}{protocol_name}Splits.csv', 'w+') as ocsv:
                writer = csv.writer(ocsv, lineterminator="\n")
                reader = csv.reader(icsv)
                
                all = []
                row = next(reader)
                row.append('SPLIT')
                all.append(row)

                for row in tqdm(reader):
                    if (row[1] in namesPerSplit[0] and row[3] in namesPerSplit[0]):
                        split  = 0
                    elif (row[1] in namesPerSplit[1] and row[3] in namesPerSplit[1]):
                        split = 1
                    elif (row[1] in namesPerSplit[2] and row[3] in namesPerSplit[2]):
                        split = 2
                    elif (row[1] in namesPerSplit[3] and row[3] in namesPerSplit[3]):
                        split = 3
                    elif (row[1] in namesPerSplit[4] and row[3] in namesPerSplit[4]):
                        split = 4
                    elif (row[1] in namesPerSplit[5] and row[3] in namesPerSplit[5]):
                        split = 5
                    elif (row[1] in namesPerSplit[6] and row[3] in namesPerSplit[6]):
                        split = 6
                    elif (row[1] in namesPerSplit[7] and row[3] in namesPerSplit[7]):
                        split = 7
                    elif (row[1] in namesPerSplit[8] and row[3] in namesPerSplit[8]):
                        split = 8
                    elif (row[1] in namesPerSplit[9] and row[3] in namesPerSplit[9]):
                        split = 9
                    else:
                        split = ""
                    if split == "":
                        print(
                            protocol_name, 
                            row, 
                            [
                                row[1] in namesPerSplit[0], 
                                row[1] in namesPerSplit[1], 
                                row[1] in namesPerSplit[2],
                                row[1] in namesPerSplit[3],
                                row[1] in namesPerSplit[4],
                                row[1] in namesPerSplit[5],
                                row[1] in namesPerSplit[6],
                                row[1] in namesPerSplit[7],
                                row[1] in namesPerSplit[8],
                                row[1] in namesPerSplit[9],
                            ], 
                            [
                                row[3] in namesPerSplit[0], 
                                row[3] in namesPerSplit[1], 
                                row[3] in namesPerSplit[2], 
                                row[3] in namesPerSplit[3], 
                                row[3] in namesPerSplit[4], 
                                row[3] in namesPerSplit[5], 
                                row[3] in namesPerSplit[6], 
                                row[3] in namesPerSplit[7], 
                                row[3] in namesPerSplit[8], 
                                row[3] in namesPerSplit[9], 
                            ]
                        )
                    row.append(split)
                    all.append(row)

                writer.writerows(all)

    # Function to rename files and prepare for ViSE protocol creation
    def rename_files(self):
        if os.path.isfile(self.path_to_datasets+"DoppelgangerProtocolSplits.csv") and \
            os.path.isfile(self.path_to_datasets+"DoppelgangerProtocol.csv"):
            
            os.remove(self.path_to_datasets+"DoppelgangerProtocol.csv")
            os.rename(self.path_to_datasets+"DoppelgangerProtocolSplits.csv", self.path_to_datasets+"DoppelgangerProtocol.csv")
        else:
            print("error with doppel_proto")

# Class for creating the ViSE protocol
class create_ViSE_protocol:
    def __init__(self, path_to_datasets, dataset):
        self.wanted_num_samples = 3500
        self.path_to_protocols = path_to_datasets
        self.path_to_dataset = self.path_to_protocols + dataset + "/"
        self.path_to_splits_fastdup = f"{path_to_datasets}FastDupSplits/"
        if os.path.isdir(self.path_to_splits_fastdup):
            shutil.rmtree(self.path_to_splits_fastdup)
        os.mkdir(self.path_to_splits_fastdup)
        self.successful_creation = self.ViSE_controller()

    # Function to control the ViSE protocol creation process
    def ViSE_controller(self):
        original_path = Path(self.path_to_dataset)

        doppelganger_protocol_df = pd.read_csv(f"{self.path_to_protocols}DoppelgangerProtocol.csv")

        sample_list = []
        id_count_dict = {}
        count = 0

        for split in doppelganger_protocol_df["SPLIT"].unique():

            df_curr_split = doppelganger_protocol_df[doppelganger_protocol_df["SPLIT"]==split]

            unique_ids = df_curr_split["INDIVIDUAL_1"].unique()
            unique_ids = np.concatenate((unique_ids, df_curr_split["INDIVIDUAL_2"].unique()))
            unique_ids = np.unique(unique_ids)

            classification_df = self.get_classification_df(original_path=original_path, current_split_filenames=unique_ids)

            split_path = Path(f"{self.path_to_splits_fastdup}/CCA_noDuplicates_{split}/")
            if split_path.exists():
                shutil.rmtree(str(split_path))
                split_path.mkdir()
            else:
                split_path.mkdir()

            for _, row in classification_df.iterrows():
                source_filename = row["filename"]
                destination_filename = split_path.joinpath(source_filename.split("/")[-2] + "/" + source_filename.split("/")[-1])

                if not split_path.joinpath(source_filename.split("/")[-2]).exists():
                    split_path.joinpath(source_filename.split("/")[-2]).mkdir()
                
                source_filename = Path(source_filename)
                destination_filename = Path(destination_filename)

                destination_filename.write_bytes(source_filename.read_bytes())

            classification_df = self.get_classification_df(original_path=split_path, current_split_filenames=unique_ids)

            model_name = f"vanilla_classification_{split}"
            threshold = 0.8
            fd = self.create_fastdup(model_name=model_name, threshold=threshold, original_path=split_path, classification_df=classification_df)
            similarity_df = fd.vis.similarity_gallery()    # create a gallery of similar images

            unpacked_similarity_df = self.unpack_similarity_df(similarity_df=similarity_df)

            sample_count = 0
            for _, row in unpacked_similarity_df.iterrows():
                if sample_count == self.wanted_num_samples:
                    break
                
                count += 1
                neg_id1 = row["from_label"]
                neg_id2 = row["to_label"]
                neg_image1 = row["from_path"].split("/")[-1]
                neg_image2 = row["to_path"].split("/")[-1]
                sample_list.append([count, neg_id1, neg_image1, neg_id2, neg_image2, 0, split])
                sample_count += 1
                
                if neg_id1 in id_count_dict.keys() and neg_id2 in id_count_dict.keys():
                    if id_count_dict[neg_id1] <= id_count_dict[neg_id2]:
                        selected_id = neg_id1
                        id_count_dict[neg_id1] += 1
                    else:
                        selected_id = neg_id2
                        id_count_dict[neg_id2] += 1
                elif not neg_id1 in id_count_dict.keys():
                    selected_id = neg_id1
                    id_count_dict[neg_id1] = 1
                else:
                    seleced_id = neg_id2
                    id_count_dict[neg_id2] = 1

                temp_pos_df = classification_df[classification_df["label"] == selected_id]
                temp_pos_df = temp_pos_df.sample(2, replace=False).reset_index()
                pos_id1 = temp_pos_df.loc[0]["label"]
                pos_id2 = temp_pos_df.loc[1]["label"]
                pos_image1 = temp_pos_df.loc[0]["filename"].split("/")[-1]
                pos_image2 = temp_pos_df.loc[1]["filename"].split("/")[-1]
                sample_list.append([count, pos_id1, pos_image1, pos_id2, pos_image2, 1, split])
                sample_count += 1

        ViSE_protocol_df = pd.DataFrame(sample_list, columns=["ID", "INDIVIDUAL_1", "IMAGE_1", "INDIVIDUAL_2", "IMAGE_2", "LABEL", "SPLIT"])

        if self.wanted_num_samples * 10 == len(ViSE_protocol_df.index):
            ViSE_protocol_df.to_csv(f"{self.path_to_protocols}ViSEProtocol.csv", index=None)
            return True
        else:
            print(f"Wanted number of samples {self.wanted_num_samples * 10}, actual number of samples {len(ViSE_protocol_df.index)}")
            return False


    # creates a list of all images and their filepath
    def get_classification_df(self, original_path, current_split_filenames):

        classification_list = []
        for id in original_path.iterdir():
            if str(id).split("/")[-1] in current_split_filenames:
                current_id_string = str(id).split("/")[-1]
                for image in id.iterdir():
                    classification_list.append([str(image), current_id_string])

            else:
                continue

        classification_df = pd.DataFrame(classification_list, columns=["filename", "label"])

        return classification_df
    
    # runs fastdup on all images in the dataset
    def create_fastdup(self, model_name, threshold, original_path, classification_df):
        output_path = f"{self.path_to_splits_fastdup}/fastdup_findDifficultVerifPairs_{model_name}_{threshold}/"
        fd = fastdup.create(work_dir=output_path, input_dir=original_path)
        if not Path(output_path).exists():
            if model_name.split("_")[0] == "vanilla":
                fd.run(annotations=classification_df, threshold=threshold)
            else:
                fd.run(annotations=classification_df, model_path=model_name.split("_")[0], d=390, threshold=threshold)
        
        return fd
    
    # removes duplicate pairs
    def remove_duplicates(self, df, column1, column2):
        # Create a new column 'sorted_AB' that contains sorted values of columns A and B
        df['sorted_AB'] = df[[column1, column2]].apply(sorted, axis=1)
        # Find duplicate rows based on the sorted values in 'sorted_AB'
        duplicate_rows = df.duplicated(subset='sorted_AB', keep='first')
        # Drop the first occurrence of the duplicates from the dataframe
        df = df[~duplicate_rows]
        del duplicate_rows
        # Remove the 'sorted_AB' column
        df = df.drop('sorted_AB', axis=1)
        df = df.sort_values(by="distance", axis=0, ascending=False)

        return df
    
    # unpacks the similarity df from fastdup for use in the ViSE protocol
    def unpack_similarity_df(self, similarity_df):
        unpacked_similarity_list = []
        for df_index, row in enumerate(similarity_df.iterrows()):
            row_distances = row[1]["distance"]
            row_label = row[1]["label"]
            row_label_2 = row[1]["label2"]
            row_from = row[1]["from"]
            row_tos = row[1]["to"]

            for row_index in range(len(row_distances)):
                unpacked_similarity_list.append([row_from, row_tos[row_index], row_label[row_index], row_label_2[row_index], row_distances[row_index]])

        unpacked_similarity_df = pd.DataFrame(unpacked_similarity_list, columns=["from_path", "to_path", "from_label", "to_label", "distance"])

        diffLabel_unpacked_similarity_df = unpacked_similarity_df[unpacked_similarity_df["from_label"] != unpacked_similarity_df["to_label"]]
        print("********************")
        print(f"Num rows w/out same class combinations: {len(diffLabel_unpacked_similarity_df.index)}")
        print("********************")

        noDup_diffLabel_unpacked_similarity_df = self.remove_duplicates(diffLabel_unpacked_similarity_df, "from_path", "to_path")

        return noDup_diffLabel_unpacked_similarity_df
    

datasets_save_location = "/home/adebolt/Desktop/Datasets/DoppelgangerVerification/"
dataset_folder = "identities"
protocol_creation(datasets_save_location, dataset_folder)