#! /usr/bin/env python3
"""
Author: Paul Mooijman
date: 2021-05-13
This program was mainly written to check if the results of the different classifier algorithms,
and their tested variables have a similar distribution, or are significantly different.
As an extra, the Shapiro-Wilk test was added to check if the tested combinations in a test have a normal distribution.

As input file, the program uses an F1-scores_...-file, produced by data_analysis.py (stored in ../Tables/).
Usage: python data_statistics_analysis.py <statistical test> <full path + F1-scores_filename.csv>

Built-in statistical tests are:
    shapiro         : Shapiro-Wilk test             (to check for a normal distribution among your test combinations)
    ttest_ind       : Student's t-test              (for normally distributed, independent samples)
    ttest_rel       : Student's t-test              (for normally distributed, paired samples)
    mannwhitneyu    : Mann-Whitney's U test         (for non-normally distributed, independent samples)
    wilcoxon        : Wilcoxon's signed-rank test   (for non-normally distributed, paired samples)

To check if the test combinations in your dataset have a normal distribution:
                                    -> use the Shapiro-Wilk test

                                    H0 assumption: The distribution of the test results is normally distributed

To check if the distribution between samples is similar or significantly different:
    if the samples in your dataset have a normal distribution:
        if samples are independent  -> use Student's t-test for independent samples
        if samples are related      -> use Student's t-test for paired samples
    if the samples in your dataset do not have a normal distribution:
        if samples are independent  -> use Mann-Whitney's U test
        if samples are related      -> use Wilcoxon's signed-rank test

                                    H0 assumption: the two compared tests have the same distribution.
"""

from itertools import combinations
from matplotlib import pyplot
from numpy.random import rand
from pandas import read_csv
import os.path
from scipy.stats import \
    mannwhitneyu,\
    shapiro,\
    ttest_ind,\
    ttest_rel,\
    wilcoxon
from statistics import mean, stdev
from sys import argv

# functions
def load_dataset(filepath):
    """
    Loads in the datafile, and returns the data as a pandas dataframe, and (feature) labels (=hormones)
    input:
    TABLE_LOCATION: str, subdirectory that contains the datafiles to process
    filename: str,  filename that contains the data ( expected: comma-separated csv-file)
    output:
    df: pandas dataframe, contains the sample data
    labels: np array, contains the names of the tests (row names of the file)
    X:          np array, contains the measure data for all tests
    """
    #df = read_csv(filepath, sep=',', skiprows=(0,1), header=None)
    df = read_csv(filepath, sep=',', header=None)
    labels = df.iloc[:,0]
    df = df.set_index(labels).transpose()
    df = df.drop(0)

    return (df, labels)

def stat_test(test, df, labels, file_path, input_file):
    """
    Performs the desired statistical test on the data in the given file
    input:
    test:   str, desired statitical test
        options: ['mannwhitneyu', 'student_t_ind', 'student_t_rel', 'wilcoxon']
    df: pandas.df, contains the sample data
    labels: np array, contains the names of the tests (row names of the file)
    file_path: str, full subdirectory path to the input file
    input_file: str,  filename that contains the data ( expected: comma-separated csv-file)
    output:
    print-to-screen of results
    out_file: csv-file, contains the statistical test results
        format: prefix_ + filename (where prefix is name of stat-test)
    """
    # H0 assumption: both samples have the same distribution
    # H1 assumption: the samples' distributions differ significantly

    test_type = {
        'mannwhitneyu'  : [mannwhitneyu, 'Mann-Whitney\'s U test', 'Mann_Whitney_'],\
        'student_t_ind' : [ttest_ind, 'Student\'s t-test for independent samples', 'Student_t_ind_'],\
        'student_t_rel' : [ttest_rel, 'Student\'s t-test for paired samples', 'Student_t_rel_'],\
        'wilcoxon'      : [wilcoxon, 'Wilcoxon\'s signed-rank test', 'Wilcoxon_test_']
    }

    if test == 'shapiro':
        shapiro_wilk_test(test, df, labels, file_path, input_file)

    elif test in ['mannwhitneyu', 'student_t_ind', 'student_t_rel', 'wilcoxon']:
        # print to screen and save to file
        print('#' * 50 + '\n%s\n' %test_type[test][1])
        out_file = file_path + test_type[test][2] + input_file
        with open(out_file, 'w') as out_fn:
            out_fn.write(str(test_type[test][1]) + '\n')
            out_fn.write("Test set 1,Test set 2,Distributions are,H0 is,Criterium,Statistic,Comments\n")

            # iterate over all combinations of two methods
            for combi in combinations(labels, 2):
                # skip the comparison if a column only contains zeros
                if (df[combi[0]] == 0).all() or (df[combi[1]] == 0).all():
                    print('%-24s vs %-24s: Test not valid due to all-zero values in at least one of the test sets' % (combi[0], combi[1]))
                    out_fn.write('%s,%s,,,,,Test not valid due to all-zero values in at least one of the test sets\n' % (combi[0], combi[1]))
                    pass
                # skip the comparison if two column are the same
                elif df[combi[0]].equals(df[combi[1]]):
                    print('%-24s vs %-24s: Test not valid due to identical values in both test sets' % (combi[0], combi[1]))
                    out_fn.write('%s,%s,,,,,Test not valid due to identical values in both test sets\n' % (
                    combi[0], combi[1]))
                    pass
                # perform statistical test on the combination
                else:
                    # compare samples
                    s_test = test_type[test][0]
                    stat, p = s_test(df[combi[0]], df[combi[1]], alternative='less')

                    # interpret
                    alpha = 0.05
                    if p > alpha:
                        print('%-24s vs %-24s: Same distribution (failed to reject H0): p(%.3f) > alpha (%.3f), Stat = %.3f'
                              %(combi[0], combi[1], p, alpha, stat))
                        out_fn.write('%s,%s,Same,failed to reject,p(%.3f) >   alpha(%.3f),%.3f\n'
                              %(combi[0], combi[1], p, alpha, stat))
                    else:
                        print('%-24s vs %-24s: Different distribution (H0 rejected): p(%.3f) <= alpha (%.3f), Stat = %.3f'
                              % (combi[0], combi[1], p, alpha, stat))
                        out_fn.write('%s,%s,Different,rejected,p(%.3f) <= alpha(%.3f),%.3f\n'
                                     % (combi[0], combi[1], p, alpha, stat))
                    for i in [0,1]:
                        print('%s : mean = %.2f, stdev= %.2f\n' %( combi[i], mean(df[combi[i]]), stdev(df[combi[i]])))

        out_fn.close()
        print('\nThe %s results were stored in file %s\n' %(test_type[test][1], out_file))
    else:
        print('Sorry! The statistical test %s is not included in this program' %test)

def shapiro_wilk_test(test, df, labels, file_path, input_file):
    """
    Performs the Shapiro-Wilk test on each row (=test) in the the input_file,
    to see if the test results have a normal/Gaussian distribution.
    H0 assumption: The distribution of the test results is normally distributed.
    input:
    test:   str, desired statitical test
        option: ['shapiro']
    df: pandas.df, contains the sample data
    labels: np array, contains the names of the tests (row names of the file)
    file_path: str, full subdirectory path to the input file
    input_file: str,  filename that contains the data ( expected: comma-separated csv-file)
    output:
    print-to-screen of results
    out_file: csv-file, contains the statistical test results
        format: prefix_ + filename (where prefix is name of stat-test)
    """
    test_type = {'shapiro'       : [shapiro, 'Shapiro-Wilk test', 'Shapiro_']}
    # print to screen and save to file
    print('#' * 50 + '\n%s\n' % test_type[test][1])
    out_file = file_path + test_type[test][2] + input_file
    with open(out_file, 'w') as out_fn:
        out_fn.write(str(test_type[test][1]) + '\n')
        out_fn.write("Test,Distributions looks,H0 is,Criterium,Statistic,Comments\n")
        for combi in labels:
            # skip the comparison if a column only contains zeros
            if (df[combi] == 0).all():
                print('%-24s: Test not valid due to all-zero values in the test' %combi)
                out_fn.write('%s,,,,,Test not valid due to all-zero values in the test.\n' %combi)
                pass

            # perform statistical test on the test results
            else:
                # compare samples
                stat, p = shapiro(df[combi])

                # interpret
                alpha = 0.05
                if p > alpha:
                    print('%-24s: Sample looks Gaussian (failed to reject H0): p(%.3f) > alpha (%.3f), Stat = %.3f'
                          % (combi, p, alpha, stat))
                    out_fn.write('%s,Gaussian,failed to reject,p(%.3f) >   alpha(%.3f),%.3f\n'
                          % (combi, p, alpha, stat))
                else:
                    print('%-24s: Sample looks non-Gaussian (H0 rejected): p(%.3f) <= alpha (%.3f), Stat = %.3f'
                          % (combi, p, alpha, stat))
                    out_fn.write('%s,Non-Gaussian,rejected,p(%.3f) <= alpha(%.3f),%.3f\n'
                          % (combi, p, alpha, stat))
    out_fn.close()
    print('\nThe %s results were stored in file %s\n' % (test_type[test][1], out_file))

def show_histogram(df,labels):
    for label in labels:
        x = df[label]
        # create histogram plot
        pyplot.hist(x, bins=100)  # define the number of bins to get a finer graph
        # show line plot
        pyplot.show()


def run_main():
    test = argv[1]
    filepath = argv[2]
    dir_name, input_file = os.path.split(filepath)
    dir_name = str(dir_name) + '\\'
    df, labels = load_dataset(filepath)
    stat_test(test, df, labels, dir_name, input_file)
    #show_histogram(df, labels)
# main
if __name__ == "__main__":
    run_main()