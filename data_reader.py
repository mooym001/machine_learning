#! /usr/bin/env python3
"""
Author: Paul Mooijman
Date: 2021-04-12
Reads in a multitude of LC-MS datafiles and produces an table with the detected hormones in all samples.
"""


#imports
from sys import argv
from os import listdir
import numpy as np
import pandas as pd

# functions
def read_animal_experiment_file(file_name):
    """
    Reads the animal_experiment_v2.csv file containing the information of hormone treated and untreated
    cows

    :input:
    file_name:  str, name of file to process
    :output:
    samples: dict, contains the sample names (minus the 'QEx_'-prefix) and the corresponding
    treatment label (0= (assumed) untreated, 1= hormone treated)
    """

    with open(file_name) as open_file:
        samples={}
        for line in open_file:
            if not line.strip():
                continue
            else:
                #split the line based on commas
                parts = line.strip(). split(',')
                # get sample name and only keep interesting part
                sample = parts[0]
                sample_parts = sample.split('_')
                sample_name = sample_parts[1] + '_' + sample_parts[2]
                # get the treatment of sample
                treatment = int(parts[7])
                # add sample_name with treatment to library
                samples[sample_name] = treatment
    return(samples)


def read_treatment_file(file_name):
    """
    Reads a treatment file containing the information of the LC-MS run analysis
    and returns the samples-of-interest as a list

    :input:
    file_name:  str, name of file to process
    :output:
    samples: list, contains the sample names (minus the 'QEx_'-prefix)
    derived from a bovine urine sample
    """

    with open(file_name) as open_file:
        samples=[]
        for line in open_file:
            if not line.strip():
                continue
            elif line.startswith('Bracket Type=4'):
                continue
            elif line.startswith('File Name'):
                continue
            else:
                #split the line based on commas
                parts = line.strip(). split(',')
                # get sample name and only keep interesting part
                sample = parts[0]
                sample_parts = sample.split('_')
                sample_name = sample_parts[1] + '_' + sample_parts[2]
                # get the origin of the sample
                sample_origin = parts[3]
                # only keep samples with origin 'bovine' or 'bovine_dierproef'
                if sample_origin == 'bovine' or sample_origin == 'bovine_dierproef' or sample_origin =='bovine_neg_hormones':
                    samples.append(sample_name)
    return(samples)

def get_info_from_all_treatment_files(subdir_name):
    """
        Reads all treatment files from the given subdirectory,
        and returns a list of all  samples-of-interest.

        :input:
        subdir_name:  str, name of the subdirectory containing the treatment files
        :output:
        total_list: list, contains all sample names (minus the 'QEx_'-prefix) of interest
        from the files in the treatment subdir
    """
    total_list = []

    # get all entries from the subdirectory
    entries = listdir(subdir_name)
    if 'geen_urine.csv' in entries:
        entries.remove('geen_urine.csv')
    # append all samples-of-interest to total_list,
    # using the read_treatment_file function
    for entry in entries:
        if entry.endswith('.csv'):
            total_list.extend(read_treatment_file(subdir_name + '\\' + entry))

    return (total_list)


def read_MINE_file(file_name):
    """
    Reads a MINE LC-MS analysis file containing the analysis results for a sample,
    and returns a list with the molecules, as well as a dictionary with the
    detected molecules with their intensities for that sample

    :input:
    file_name:  str, name of file to process
    :output:
    molecules: lst, contains the molecule names
    samples: dict, contains the molecules(key) with the summed intentities(val)
    """

    with open(file_name) as open_file:
        hit_IDs=[]
        molecules=[]
        samples = {}
        for line in open_file:
            if not line.strip():
                continue
            elif line.startswith('Hit_ID'):
                continue
            else:
                #split the line based on commas
                parts = line.strip(). split(',')
                hit_ID = int(parts[0])
                retention = float(parts[3])
                intensity = float(parts[5])
                molecule = parts[9]
                if hit_ID in hit_IDs:
                    continue
                else:
                    hit_IDs.append(hit_ID)
                    if molecule not in molecules:
                        molecules.append(molecule)
                        samples[molecule] = intensity
                    else:
                        samples[molecule] += intensity
        return(molecules, samples)

def read_all_MINE_files(file_list, MINE_CSVS_DIR):
    """
    takes the MINE samplefile-list and retrieves the molecule and Intensity
    information for all these samples.
    Returns a list containing all found molecules, as well as a
    list of sample names linked to a dict containing molecules(key),
    with the summed intensities (val)
    input:
    file_list: lst, contains all
    output:
    molecule_list: lst, contains the molecules detected in all samples
    sample_list: ls, contains sample names linked to a dict with molecules(keys),
    summed intensities(vals)
    """
    all_molecules=[]
    all_sample_names = []
    all_sample_info=[]
    for sample_name in file_list:
        sample_path= MINE_CSVS_DIR + 'MINE_C18__combined_database_R_Qex_' + sample_name + '.csv'
        molecules, samples = read_MINE_file(sample_path)
        all_molecules.extend(molecules)
        sample_info = [sample_name, samples]
        all_sample_names.append(sample_name)
        all_sample_info.append(sample_info)
    # remove duplicates
    all_molecules = list(set(all_molecules))
    return (all_molecules, all_sample_names, all_sample_info)

def make_clas_label_column(sample_names, class_label):
    """
    Creates a list with the class-scoring (1= hormone treated, 0 = untreated) for all samples of interest.
    input:
    sample_names: lst, contains names of all samples-of-interest
    class_labels: lst, contains class labels extracted from the animal_experiment file
    output:
    class_label: lst, contains class labels for all samples-of-interest
    """
    label = [0]*len(sample_names)
    for key, val in class_label.items():
        if val == 1:
            label[sample_names.index(key)] = 1
    return (label)


def make_dataframe(row_names, column_names, sample_info, class_labels):
    """
    Creates a Pandas Dataframe from the sample_info.
    Sample cells that do not contain information are set at a random value
    between 9999.5 and 10000.5
    input:
    row_names: lst, contains the sample names
    column_names: lst, contains the column names
    sample_info: lst, contains detected molecules with intensity for each sample
    output:
    df: pandas.dataframe, contains the molecule formulas (columns)
    with summed intensties (cells) for each sample (rows)
    """
    df = pd.DataFrame(np.random.rand(len(row_names), len(column_names))*10000 + 5000.0,\
                      index = row_names, columns = column_names)
    for sample in sample_info:
        sample_name= sample[0]
        for key, val in sample[1].items():
            df[key][sample_name] = val
    # Add the class scoring to the dataframe
    labels = make_clas_label_column(row_names, class_labels)
    df['class_label'] = labels
    return df

def run_main():
    MAIN_DIR = "D:/thesis/INF_data_080320/"
    DESCRIPTION_CSVS_DIR = MAIN_DIR + 'description_csvs/'
    MINE_CSVS_DIR = MAIN_DIR + 'Mine_csvs/'
    ANIMAL_EXPERIMENT_LCCATION = MAIN_DIR + 'animal_experiments_v2.csv'
    OUPUTFILE_LOCATION = MAIN_DIR + 'Sample_info.csv'

    # collect information from the input files
    print('Running!')
    samples_of_interest = get_info_from_all_treatment_files(DESCRIPTION_CSVS_DIR)
    molecules, sample_names, sample_info = read_all_MINE_files(samples_of_interest, MINE_CSVS_DIR)
    class_labels = read_animal_experiment_file(ANIMAL_EXPERIMENT_LCCATION)

    # create the dataframe
    df = make_dataframe(sample_names, sorted(molecules), sample_info, class_labels)
    print('Showing first 50 columns of the dataset')
    print(df.head(50))
    print('The datafile contains %d samples'%(len(df)))
    print('%d samples are of class 1' %(len(df[df.class_label == 1])))
    print('%d samples are of class 0' %(len(df[df.class_label == 0])))
    # save the file
    df.to_csv(OUPUTFILE_LOCATION)
    print('info is saved to: ' + str(OUPUTFILE_LOCATION))
    return df


# main
if __name__ == "__main__":
    run_main()



