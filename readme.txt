Author: Paul Mooijman
date: 2022-05-20

These scripts were written as part of my thesis within the Information Technology chairgroup, 
in cooperation with the Wageningen Food Safety Research institute.
 
Goal was the identification of a machine learning algorithm that could correctly identify cases of 
illegal growth stimulating hormone usage within the cattle industry by analysing an LC-MS dataset of bovine urine samples,
keeping in mind that the dataset contained a multitude of "missing data points" that were the result of compound concentrations 
falling below the detection threshold of the LC-MS machine and the fact that we were dealing with an imbalanced dataset.



The python script data_reader.py was created to read in the original datafiles, and produced the dataset for the machine learning algorithms.
Missing datapoints are replaced here with random values around the detection threshold of the LC-MS machine. 

The file Sample_info.csv is the (adapted) result of this script.
Due to confidentiality issues the original names of the samples have been changed.

The python script data_analysis.py was the main script written to test the diffrrent machine learning algorithms.
The script includes several data transformation functions, resampling methods, machine learning algorithms, 
and a prediction function that uses the machine learning algorithms that were found to work best on our dataset.
The script is written in such a way that analyses can be repeated with other settings by switching on/off the desired functions 
in the run_main function.

Finally the python script data_statistics_analysis.py was written to perform a statistical comparison between tested algorithm outcomes,
and determine if the outcomes of the tests differ significantly.
 

