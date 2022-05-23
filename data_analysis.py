#! /usr/bin/env python3
"""
Author: Paul Mooijman
date: 2021-04-13
Performs a data analysis using several machine learning algorithms
on an imbalanced dataset
"""


# imports
from collections import\
    defaultdict,\
    Counter

import pandas as pd
from imbalanced_ensemble.ensemble import AdaCostClassifier
from imblearn.ensemble import \
    BalancedRandomForestClassifier,\
    EasyEnsembleClassifier,\
    RUSBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import \
    ADASYN,\
    BorderlineSMOTE,\
    SMOTE,\
    SVMSMOTE,\
    RandomOverSampler
from imblearn.under_sampling import \
    CondensedNearestNeighbour,\
    EditedNearestNeighbours,\
    NearMiss,\
    NeighbourhoodCleaningRule,\
    OneSidedSelection,\
    RandomUnderSampler,\
    RepeatedEditedNearestNeighbours,\
    TomekLinks
from imblearn.pipeline import Pipeline
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from operator import itemgetter
import os
from pandas import \
    read_csv,\
    set_option,\
    DataFrame
from scipy.cluster import hierarchy
from scipy.stats import \
    boxcox,\
    normaltest,\
    spearmanr, \
    shapiro
from sklearn.base import \
    BaseEstimator,\
    TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import \
    LinearDiscriminantAnalysis,\
    QuadraticDiscriminantAnalysis
from sklearn.ensemble import\
    AdaBoostClassifier,\
    BaggingClassifier,\
    ExtraTreesClassifier,\
    GradientBoostingClassifier,\
    RandomForestClassifier,\
    IsolationForest
from sklearn.feature_selection import \
    chi2,\
    f_classif,\
    mutual_info_classif,\
    RFE,\
    RFECV,\
    SelectFromModel,\
    SelectKBest,\
    VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import \
    RBF,\
    DotProduct,\
    Matern,\
    RationalQuadratic,\
    WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import \
    accuracy_score,\
    auc,\
    average_precision_score,\
    brier_score_loss,\
    classification_report,\
    confusion_matrix,\
    f1_score,\
    make_scorer,\
    precision_recall_curve,\
    precision_score,\
    recall_score,\
    roc_auc_score
from sklearn.model_selection import \
    cross_val_predict,\
    cross_val_score, \
    cross_validate,\
    GridSearchCV,\
    StratifiedKFold, \
    RepeatedStratifiedKFold, \
    train_test_split
from sklearn.naive_bayes import \
    GaussianNB, \
    MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import \
    FunctionTransformer,\
    MinMaxScaler,\
    normalize,\
    Normalizer,\
    StandardScaler,\
    PowerTransformer,\
    LabelEncoder
from sklearn.svm import\
    SVC,\
    OneClassSVM
from sklearn.tree import\
    DecisionTreeClassifier,\
    DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
from time import asctime
from xgboost import XGBClassifier

# classes
class NormalizeTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y = None):
        return self

    def transform(self, x, y=None):
        x = pd.DataFrame(x)
        columns = x.columns
        norm = Normalizer()
        x[columns] = norm.fit_transform(x[columns])
        return x


class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y = None):
        return self

    def transform(self, x, y=None):
        x = pd.DataFrame(x)
        for column in x.columns:
            x[column] = np.log(x[column])
        return x


class ReduceVIF(BaseEstimator, TransformerMixin):
    """
    Feature reduction method based on the variance inflation factor.
    Taken from https://stats.stackexchange.com/a/253620/53565 and modified.
    """
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = SimpleImputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    def calculate_vif(X, thresh=5.0):

        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print('Dropping %-13s with vif = %.3f' % (X.columns[maxloc] ,max_vif))
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X



# Functions

# Data import and export functions
def load_dataset(filename):
    """
    Loads in, and returns the datafile
    input:
    filename: str,  path + input filename ( expected: csv-format)
    output:
    df: pandas dataframe, contains the sample data
    """
    print('#' * 50 + '\nLoading the file %s' %filename)
    df = read_csv(filename, header=0)
    return df

def train_test_splitting(df):
    """
    Splits the dataset into a 90% training set and a 10% test set,
    then returns and saves the train and test set to files (if not saved already).
    The test set is meant for evaluate the classification model on samples
    that has not seen before during training and validation.
    input:
    df: pandas dataframe, contains the complete dataset
    output:
    train_set: pandas dataframe, contains the training dataset
    test_set: pandas dataframe, contains the validation dataset.
    """
    out_fn = MAIN_DIR + 'train_dataset.csv'
    out_fn2= MAIN_DIR + 'test_dataset.csv'
    if os.path.exists(out_fn):
        print("Loading the pre-produced train and test dataset")
        train_set = read_csv(out_fn, header=0)
        test_set = read_csv(out_fn2, header=0)
        print("Information of the train dataset")
        show_basic_dataset_info(train_set)
        print("Information of the test dataset")
        show_basic_dataset_info(test_set)
        return train_set, test_set
    else:
        print("Producing the train and test dataset")
        sample_names, x, y = get_x_y(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle= True, stratify=y)    # stratify=y -> here y are the y-values not y(es)
        train_set = pd.concat([sample_names, x_train, y_train], axis=1).dropna()
        test_set = pd.concat([sample_names, x_test, y_test], axis=1).dropna()
        train_set.to_csv(out_fn, index=False)
        test_set.to_csv(out_fn2, index=False)
        print("Information of the train dataset")
        show_basic_dataset_info(train_set)
        print("Information of the test dataset")
        show_basic_dataset_info(test_set)
        return train_set, test_set

def get_x_y(df):
    """
    Splits the dataframe into the x-values (=features) and y-values (=class labels)
    :param df: pandas datafram, contains the dataset
    output:
    sample_names: pandas dataframe, contains the sample names
    x: pandas dataframe, contains the x-values (=feature values)
    y: pandas dataframe, contains the y-values (= class labels)
    """
    sample_names = df[df.columns[0]]
    x = df[df.columns[1:-1]]
    y = df[df.columns[-1]].astype('int')
    return sample_names, x, y


# Special measure definition functions
def pr_auc_score(y_true, y_pred):
    """
    Generates the Area Under the Curve for the precision and recall.
    """
    return average_precision_score(y_true, y_pred)


def brier_skill_score(y_true, y_pred):
    """
    Generates the Brier Skill Score to determine a Classifier's performance.
    input:
    y_true: pandas df, contains the real class labels of the samples
    y_pred: pandas df, contains the predicted class labels of hte samples
    output:
    brier skill score: float, BrierSkillScore of the classifier.
    """
    # calculate reference
    weight = np.count_nonzero(y_true == 1) / np.count_nonzero(y_true == 0)
    probabilities = [weight for _ in range(len(y_true))]
    brier_ref = brier_score_loss(y_true, probabilities)
    # calculate BrierSkillScore
    bs = brier_score_loss(y_true, y_pred)
    return 1.0 - (bs / brier_ref)


def TP_score(y_true, y_pred):
    """
    Generates the confusion matrix's True Positive score for a classifier
    input:
    y_true: pandas df, contains the real class labels of the samples
    y_pred: pandas df, contains the predicted class labels of hte samples
    output:
    TP: np.array, contains the values for the False Negative
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm[1,1]


def FN_score(y_true, y_pred):
    """
    Generates the confusion matrix's False Negative scores for a classifier
    input:
    y_true: pandas df, contains the real class labels of the samples
    y_pred: pandas df, contains the predicted class labels of hte samples
    output:
    FN: np.array, contains the values for the False Negative
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 0]


def FP_score(y_true, y_pred):
    """
    Generates the confusion matrix's False Positive scores for a classifier
    input:
    y_true: pandas df, contains the real class labels of the samples
    y_pred: pandas df, contains the predicted class labels of hte samples
    output:
    FP: np.array, contains the values for the False Positives
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 1]


def TN_score(y_true, y_pred):
    """
    Generates the confusion matrix's True Negative scores for a classifier
    input:
    y_true: pandas df, contains the real class labels of the samples
    y_pred: pandas df, contains the predicted class labels of hte samples
    output:
    TN: np.array, contains the values for the True Negatives
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0]


#statistical tests
def shapiro_wilk_test(data):
    """
    Performs the Shapiro-Wilk test on each column in the features-data (= X),
    to see if the data in that feature has a normal/Gaussian distribution
    input:
    data: pandas array, contains the feature data (X-values of the dataset)
    :return:
    print to screen if the data in each column looks Gaussian distributed
    """
    print('#' * 50 + '\nRunning the Shapiro-Wilk test')
    print('(check if the features are Gaussian distributed)')

    for i in range(data.shape[1]):
        print('column %.3d:' % (i+1))
        # normality test
        stat, p = shapiro(data[:, i])

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')


def dagostino_pearson_test(data):
    """
    Performs the d'Agostino-Pearson test on each column in the features-data (= X),
    to see if the data in that feature has a normal/Gaussian distribution
    input:
    data: pandas array, contains the feature data (X-values of the dataset)
    :return:
    print to screen if the data in each column looks Gaussian distributed
    """
    print('#' * 50 + '\nRunning the d\'Agostino-Pearson-test')
    print('(check if the features are Gaussian distributed)')

    for i in range(data.shape[1]):
        print('column %.3d:' % (i + 1))
        # normality test
        stat, p = normaltest(data[:, i])

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')


# visalisation methods
def show_basic_dataset_info(df):
    # produces some general information about the class and features
    print('#' * 50)
    shape = df.shape
    print("Total number of sample:   %5d" % shape[0])
    class_count = df.groupby('class_label').size()
    print("Total samples of class 0: %5d (%6.2f perc)" % (class_count[0], (class_count[0] / shape[0] * 100)))
    print("Total samples of class 1: %5d (%6.2f perc)" % (class_count[1], (class_count[1] / shape[0] * 100)))
    print("Total number of features: %5d" % (shape[1]-2))
    print('#'*50)

    print('statistics of the dataset:')
    print(df.describe())


def make_distribution_plots(data, feature_names, start, end):
    """
    Creates a series of histograms for the desired features in your dataset
    Use start and end to define a range of images to produce
    Images will be shown but not saved.
    input:
    data: pandas df, contains the feature data (X-values of the dataset)
    feature_names: lst, contains the feature names from the dataset
    start: int, first features of the dataset to produce an image for
    end: int, last feature of rthe dataset to produce an image for
    :return:
    print to screen if the data in each column looks Gaussian distributed
    """
    for i in range(start, end):
        plt.hist(data[i], bins=50)
        plt.title(feature_names[i])
        plt.xlabel('Log Hormone Intensities')
        plt.ylabel('Frequency')
        plt.show()


def boxcox_transf(data, lmbda):
    """
    Performs a boxcox transformation on the data,
    to try to make it more Gaussian-like
    input:
    data: np array, contains the feature values (= X-values)
    lmbda: hyperparameter for the boxcox function
            options:
             -1,0: reciprocal transformation
             -0.5: reciprocal square root
             0.0: log transformation
             0.5: square root transformation
             1.0: no transformation
    :return:
    data_mod: np array, contains the transformed feature values (X-values).
    """
    data_mod = data.copy()
    for i in range(data.shape[1]):
        data_mod[:, i] = boxcox(data[:, i].astype(float), lmbda)
    return data_mod


def pairwise_pearson(df):
    """
    performs a pairwise pearson correlation on the dataset
    input:
    df: pandas df, contains the data to analyse
    output:
    """
    set_option('display.width', 100)
    set_option('precision', 3)
    correlations = df.corr(method='pearson')
    print(correlations)


def correlation_plot(df, method):
    """
    creates a correlation heatmap of the data
    input:
    df: pandas dataframe, contains the data
    method: str, correlation method to be used. options:['pearson', 'spearman', 'kendall']
    shows:
    correlation heatmap image.
    """
    names = list(df)
    names.remove('Unnamed: 0')
    correlations = df.corr(method=method)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


def create_box_whisker_plot(results, names, location, file_to_save):
    """
    Creates and saves a box-whisker plot
    input:
    results: lst, contains the data to plot
    names: lst, contains the model names
    file_to_save: str, name of the boxplot image file
    output: box-whisker plot
    """
    plt.boxplot(results, labels=names, showmeans=True)
    plt.xlabel('Method')
    plt.ylabel('F1-score')
    plt.ylim((0, 1.0))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(IMG_LOCATION + location + file_to_save + ".png")
    plt.close()
    print('The plot-image is saved to file %s' % IMG_LOCATION + location + file_to_save + ".png")



# Feature modification methods
def normalize_transform(df):
    """
    Normalizes the feature values in the dataframe with maintenance of the feature names
    input:
    df: pandas dataframe, contains the feature data
    output:
    df: pandas dataframe, contains the normalizes feature data
    """
    column_names = df.columns
    df = pd.DataFrame(normalize(df), columns=column_names)
    return df


def logtransform(df):
    """
    Log transforms the feature values in the dataframe, with maintenance of the feature names
    input:
    df: pandas dataframe, contains the feature data
    output:
    df: pandas dataframe, contains the log transformed feature data
    """
    column_names = df.columns
    logtransformer = FunctionTransformer(np.log, validate=True)
    df = pd.DataFrame(logtransformer.transform(df), columns=column_names)
    return df


def recursive_feature_eliminator(df, model, out_fn_prefix, norm=False, logtr=False):
    """
    Prunes the least informative features from the dataset,
    only keeping the features showing the best correlation with the class labels.
    input:
    df: contains the dataset
    model: str, name of the classifier to use for feature importance determination
    norm: bool, True if you wish to normalize the dataset
    logtr: bool, True if you wish to log transform the dataset
    out_fn_prefix: str, prefix for the desired filename to store the results into
    output:
    df: pandas datafram, contains the feature filtered dataset
    saved csv-file with the feature eliminated dataset

    """
    print('#' * 50 + '\nCalling Recursive Feature Eliminator')

    if norm == True and logtr == True:
        out_fn = MAIN_DIR + out_fn_prefix + '_norm_logtr.csv'
        info = 'with normalization + logtransformation'
    elif norm == True and logtr == False:
        out_fn = MAIN_DIR + out_fn_prefix + '_norm.csv'
        info = 'with normalization'
    elif norm == False and logtr == True:
        out_fn = MAIN_DIR + out_fn_prefix + '_logtr.csv'
        info = 'with logtransformation'
    else:
        out_fn = MAIN_DIR + out_fn_prefix + '.csv'
        info = 'without normalization or logtransformation'
    if os.path.exists(out_fn):
        print('Attention!\nFile %s already exists.\nNot producing it again.' % out_fn)
        print('If you do wish a new file to be produced, first rename or discard the old one.')
    else:
        print('Producing file %s.\nThis may take a while! Have some coffee.' % out_fn)
        print('Used model: %s' % model)
        print(info)

        sample_names, x, y = get_x_y(df)
        feature_names = x.columns

        # perform normalization and/or log transformation, if requested
        if norm == True:
            x = normalize_transform(x)
        if logtr == True:
            x = logtransform(x)
        # feature extraction
        rfe = RFECV(
            estimator=model,
            step=1,
            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1),
            min_features_to_select=1
            )
        fit = rfe.fit(x, y)
        print("Optimal number of features: %d" % fit.n_features_)
        kept = [x for x, y in zip(feature_names, fit.support_) if y == True]
        print("Selected features:\n%s\n" % kept)
        discarded = [x for x, y in zip(feature_names, fit.support_) if y == False]
        print("Number of discarded features: %d" % len(discarded))
        print("Discarded features:\n%s\n" % discarded)
        # join everything together into a new dataframe
        x = pd.DataFrame(rfe.fit_transform(x,y))
        df = pd.concat([sample_names, x, y], axis=1)

        # save new dataframe to file
        df.to_csv(out_fn, index=False)
        print('File %s has been saved' % out_fn)


def low_variance_feature_removal(df, threshold=0.05):
    """
    FORGET IT!!! THIS DOES NOT WORK ON OUR DATASET.
    This function attempts to remove features from the dataset (X) that are identical in more than the set fraction of the samples.
    input:
    X: np.array, contains the feature data

    """
    print('#' * 50 + '\nRunning Low variance feature removal')
    selector = VarianceThreshold(threshold=threshold)
    x = df[df.columns[1:-1]]
    feature_names = x.columns.tolist()
    selector.fit(x)
    print("Optimal number of features: %d" % len(selector.get_support()))
    kept = [a for a, b in zip(feature_names, selector.get_support()) if a == True]
    print("Selected features: \n %s \n" % kept)
    discarded = [a for a, b in zip(feature_names, selector.get_support()) if b == False]
    print("Number of discarded features: %d" % len(discarded))
    print("Discarded features: \n %s \n" % discarded)
    return df[df.columns[selector.get_support(indices=True)]]


def highly_correlated_feature_removal(df, cutoff=0.90):
    """
    This function removes highly correlated features from a dataset, using the correlation matrix data
    this script was based on the example found on the site:
    https://www.dezyre.com/recipes/drop-out-highly-correlated-features-in-python
    input:
    df: pandas dataframe, contains the dataset
    cutoff: float, cutoff-value: The minimal percentage for which a features should be correlated to other features
    in order for it to be dropped.
    output:
    df1: pandas dataframe, contains the dataset with the highly correlated features removed.
    """
    print('#' * 50 + '\nRemoving highly correlated features from dataframe')
    # produce a correlation matrix
    corr_matrix = df.corr().abs()
    print('correlation matrix: ')
    print(corr_matrix)

    # selecting the upper triangle of the correlation matrix without the diagonal 1.0 values
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # find the columns that need to be dropped due to high correlation to other columns
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > cutoff)]

    print('A total of %d features (=columns) were more than %.1f %% correlated to other features,'\
          %(len(to_drop), (cutoff*100)))
    print('and will therefore be removed.')
    print('Removed features: ' + str(to_drop))

    # drop the column with high correlation
    df = df.drop(columns=to_drop, axis=1)
    return df


def multicollinear_feature_removal(df):
    """
    Removes features that are highly collinear with other features within the dataset
    This function and its class are based on info from:
    https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class
    input:
    df: pandas df: contains the dataset
    :return:
    df: np.array, contains the cleaned up data
    """
    print('#' * 50 + '\nCalling multicollinear feature removal')
    print('Features with a variance inflation factor (vif) > 5.0 will be dropped')

    out_fn = MAIN_DIR + 'sample_file2.csv'
    if os.path.exists(out_fn):
        df = read_csv(out_fn, header=0)
        return df
    else:
        print('Beware! This step takes quite some time to run.')
        # get x and y from the dataframe
        sample_names, x, y = get_x_y(df)


        steps = [('t1', NormalizeTransform()), ('t2', LogTransform()), ('t3', ReduceVIF())]
        model = Pipeline(steps)
        x = model.fit_transform(x, y)
        x = pd.DataFrame(x)
        df = pd.concat([sample_names, x, y], axis=1)
        df.to_csv(out_fn, index=False)
        return df

def selectkbest_features(df, k=10):
    """
    This function removes all but the <k> best features from the feature data <X>,
    based on the univariate statistical test.
    X: np.array, contains the feature data
    y: np.array, contains the class values
    feature_names: lst, contains the feature names from the dataset
    k: int, number of features to keep (default = 10)
    output:
    X_new: np.array, contains the <k> best features
    kept: list, contains the names of the <k> best features
    """
    sample_names, x, y = get_x_y(df)
    feature_names = x.columns.tolist()

    print('#' * 50 + '\nRunning SelectKBest Feature selection')
    model = SelectKBest(score_func=chi2, k=k)
    fit = model.fit(X, y)
    kept = [a for a, b in zip(feature_names, fit.get_support()) if b == True]
    print("Number of selected features: \n %s \n" % len(kept))
    print("Selected features: \n %s \n" % kept)
    discarded = [a for a, b in zip(feature_names, fit.get_support()) if b == False]
    print("Number of discarded features: %d" % len(discarded))
    print("Discarded features: \n %s \n" % discarded)
    X_new = fit.transform(X)
    return X_new


def adaboost_feature_selector(df, out_fn_prefix):
    print('#' * 50 + '\nRunning Adaboost Feature Selector')
    out_fn = MAIN_DIR + out_fn_prefix + '.csv'

    sample_names, x, y = get_x_y(df)
    feature_names = x.columns.to_list
    model = AdaBoostClassifier(n_estimators=1000)
    fit = model.fit(x,y)
    model =SelectFromModel(fit, prefit=True)
    x_new = pd.DataFrame(model.transform(x), columns= [x.columns[i] for i in range(len(x.columns)) if model.get_support()[i]])
    print('Selected features: %s' %list(x_new.columns))
    df = pd.concat([sample_names, x_new, y], axis=1)
    df.to_csv(out_fn, index=False)
    return df


def principal_component_analysis(df):
    print('#' * 50 + '\nRunning Principal Component Analysis')
    sample_names, x, y = get_x_y(df)
    feature_names = x.columns.tolist()
    # perform StandardScaling on the feature data
    mod = StandardScaler()
    x = mod.fit_transform(x)
    # perform pca analysis on the features
    model = PCA(n_components=3)
    fit = model.fit(x)

    print('PCA1: %f' % fit.explained_variance_ratio_[0])
    print('PCA2: %f' % fit.explained_variance_ratio_[1])
    print('PCA3: %f' % fit.explained_variance_ratio_[2])


    principal_components = model.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2', 'principal component 3'])
    final_df = pd.concat([principal_df, y], axis=1)

    # 3d image visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    class_labels = [0, 1]
    colors = ['r', 'g']
    for class_label, color in zip(class_labels, colors):
        indices_to_keep = final_df['class_label'] == class_label
        ax.scatter(final_df.loc[indices_to_keep, 'principal component 1'],
                   final_df.loc[indices_to_keep, 'principal component 2'],
                   final_df.loc[indices_to_keep, 'principal component 3'],
                   c=color,
                   s=50)
    ax.legend(class_labels)
    ax.grid()

    # If you wish to add the features as vectors to the plot, activate this part:
    """
    enlargement_factor = 50
    coeff = np.transpose(model.components_[0:3, :])
    for i in range(len(feature_names)):
        ax.quiver(0, 0, 0, coeff[i, 0] * enlargement_factor, coeff[i, 1]* enlargement_factor, coeff[i, 2] * enlargement_factor, color = 'b', arrow_length_ratio=0.01)
        ax.text(coeff[i, 0]* 1.15 * enlargement_factor, coeff[i, 1]* 1.15 * enlargement_factor, coeff[i, 2]* 1.15 * enlargement_factor, feature_names[i], color = 'b', ha = 'center', va = 'center')
    """

    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(30, 40)
    plt.show()
    return(final_df)

##########################
# outlier removal function

def automatic_outlier_removal(df, out_fn, class1_contamination=0.056, removal_threshold_class_0=3, removal_threshold_class_1=8):
    """
    This function removes samples from the majority class that are considered outliers
    by two OneClass classifiers: OneClassSVM and IsolationForest.
    A repeated stratified kfold crossvalidation strategy is used to determine if a sample is considered
    to be an outlier more than once.
    Samples that are scored as outliers equal or more times as <removal_threshold> will be dropped from the dataset.
    input:
    df: pandas dataframe, contains hte dataset
    class1_contamination: float, represents the fraction of class 1 samples in your dataset. default set at 0.056
    removal_threshold: integer,  the minimal number of times a samples needs to be scored as an outlier before it is removed.
    output:
    df: pandas dataframe, the dataset without the outlier samples.
    """
    print('#' * 50 + '\nRunning Automatic Outlier Removal')
    out_fn = MAIN_DIR + out_fn
    if os.path.exists(out_fn):
        print('Attention!\nFile %s already exists.\nNot producing it again.' % out_fn)
        print('If you do wish a new file to be produced, first rename or discard the old one.')
        df = load_dataset(out_fn)

    else:
        sample_names, x, y = get_x_y(df)
        questionables_class0 = {}
        questionables_class1 = {}
        rskf= RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=3)
        for train_index, test_index in rskf.split(x, y):
            train_x, test_x = x.iloc[train_index], x.iloc[test_index]
            train_y, test_y = y.iloc[train_index], y.iloc[test_index]

            # make predictions with OneClassSVM
            model = OneClassSVM(gamma='scale', nu=class1_contamination)

            train_x0 = train_x[train_y == 0]    # class 0 samples
            train_x1 = train_x[train_y == 1]    # class 1 samples
            # detect class 0 outliers in the test set
            model.fit(train_x0)
            y_hat0_1 = model.predict(test_x)    # make predictions for class 0 samples
            # detect class 1 outliers in the test set
            model.fit(train_x1)
            y_hat1_1 = model.predict(test_x)    # make predictions for class 1 samples
            # outliers are labeled as -1, inliers as 1

            # make predictions with IsolationForest
            model2 = IsolationForest(contamination=class1_contamination)
            model2.fit(train_x0)
            y_hat0_2 = model2.predict(test_x)   # make predictions for class 0 samples
            model2.fit(train_x1)
            y_hat1_2 = model2.predict(test_x)   # make predictions for class 1 samples


            # Only store those samples that are predicted as outliers by both models

            for i, y_true, y2, y3 in zip(test_y.index, test_y, y_hat0_1, y_hat0_2):
                if y_true == 0 and y2 == -1 and y3 == -1:
                    if i not in questionables_class0:
                        questionables_class0[i] = 1
                    else:
                        questionables_class0[i] += 1
            for i, y_true, y2, y3 in zip(test_y.index, test_y, y_hat1_1, y_hat1_2):
                if y_true == 1 and y2 == -1 and y3 == -1:
                    if i not in questionables_class1:
                        questionables_class1[i] = 1
                    else:
                        questionables_class1[i] += 1

        print('Class 0 samples that are questionable:')
        print('sample     outlier score')
        for key, value in questionables_class0.items():
            print(sample_names[key], value)
        print('number of questionable samples: %d' % len(questionables_class0))

        print('Class 1 samples that are questionable:')
        print('sample     outlier score')
        for key, value in questionables_class1.items():
            print(sample_names[key], value)
        print('number of questionable samples: %d' % len(questionables_class1))
        # Remove all samples that are scored as outlier equal or more than the desired threshold

        to_remove =[]
        for key, value in questionables_class0.items():
            if value >= removal_threshold_class_0:
                to_remove.append(int(key))
        for key, value in questionables_class1.items():
            if value >= removal_threshold_class_1:
                to_remove.append(int(key))
        print('Samples that will be removed:\n%s' %sample_names[to_remove])
        print('Nr of samples to be removed: %d' %len(to_remove))
        print(df.shape)
        df = df.drop(to_remove)
        print(df.shape)
        df.to_csv(out_fn, index=False)
        print('File %s has been saved to disc.' % out_fn)
    return df

# Defining the Classification models to use
def basic_models():
    """
    Defines several basic classifier models without any optimalization settings
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
    """
    file_to_save = 'Comparison_Basic_Classifiers_Performance_on_Dataset'
    description = 'Comparison of several basic classifier methods on the dataset'
    models, names = list(), list()
    max_iter = 1000
    # LR
    models.append(LogisticRegression(solver='liblinear', max_iter=max_iter))
    names.append('LR')

    # LDA
    models.append(LinearDiscriminantAnalysis())
    names.append('LDA')

    # KNN
    models.append(KNeighborsClassifier())
    names.append('KNN')

    # DecisionTree
    models.append(DecisionTreeClassifier())
    names.append('DecisionTree')

    # QDA
    models.append(QuadraticDiscriminantAnalysis())
    names.append('QDA')

    # GNB
    models.append(GaussianNB())
    names.append('GaussianNB')

    # MNB   -> Works only rarely
    #models.append(MultinomialNB())
    #names.append('MNB')

    # GPC
    models.append(GaussianProcessClassifier())
    names.append('GaussianProc')

    # SVM
    models.append(SVC(gamma='scale'))
    names.append('SVM')

    return models, names, file_to_save, description


def LogRegression_normalization_standardization_models():
    """
    Define different combination of standardization, normalization and PowerTransformation methods
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
    """
    file_to_save = 'Comparison_Unbalanced_Normalization_and_standardization_on_LogRegression'
    description = 'LogRegression performance tested with standardization, normalization and PowerTransformation methods'
    models, names = list(), list()
    max_iter = 500

    # LR Unbalanced
    models.append(LogisticRegression(solver='liblinear', max_iter=max_iter))
    names.append('LR')

    # LR + Normalization
    steps = [('t', Normalizer()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+Normalization')

    # LR + MinMaxScaling
    steps = [('t', MinMaxScaler()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+MinMaxScaling')

    # LR + StandardScaling
    steps = [('t', StandardScaler()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+StdScaling')

    # LR + PowerTransformation
    steps = [('t', PowerTransformer(method='box-cox')), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+PowerTr')

    # LR + LogTransformation
    steps = [('t', LogTransform()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+LogTr')

    # LR  + MinMaxScaling + Normalization
    steps = [('t1', MinMaxScaler()), ('t2', Normalizer()), ('m', LogisticRegression(solver='liblinear',
                                                                                    max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+MinMaxScaling+Norm')

    # LR + MinMaxScaling + StandardScaling
    steps = [('t1', MinMaxScaler()), ('t2', StandardScaler()), ('m', LogisticRegression(solver='liblinear',
                                                                                        max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+MinMax+StdScaling')

    # LR + MinMaxScaling + PowerTransformation
    steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()), ('m', LogisticRegression(solver='liblinear',
                                                                                          max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+MinMax+PowerTr')

    # LR + MinMaxScaling + LogTransformation
    # TRIED IT, BUT DID NOT WORK!!! -> Div by zero error

    # LR  + Normalization + MinMaxScaling
    steps = [('t1', Normalizer()), ('t2', MinMaxScaler()), ('m', LogisticRegression(solver='liblinear',
                                                                                    max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+Norm+MinMaxScaling')

    # LR  + Normalization + StandardScaling
    steps = [('t1', Normalizer()), ('t2', StandardScaler()), ('m', LogisticRegression(solver='liblinear',
                                                                                      max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+Norm+StandardScaling')

    # LR  + Normalization + PowerTransformation
    steps = [('t1', Normalizer()), ('t2', PowerTransformer()), ('m', LogisticRegression(solver='liblinear',
                                                                                        max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+Norm+PowerTr')

    # LR  + Normalization + LogTransformation
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', LogisticRegression(solver='liblinear',
                                                                                    max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+Norm+LogTr')

    # LR  + StandardScaling + Normalization
    steps = [('t1', StandardScaler()), ('t2', Normalizer()), ('m', LogisticRegression(solver='liblinear',
                                                                                      max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+StandardScaling+Norm')

    # LR + StandardScaler + PowerTransformation
    steps = [('t1', StandardScaler()), ('t2', PowerTransformer()), ('m', LogisticRegression(solver='liblinear',
                                                                                            max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+StdScaling+PowerTr')

    # LR + LogTransformation + MinMaxScaling
    steps = [('t1', LogTransform()), ('t2', MinMaxScaler()), ('m', LogisticRegression(solver='liblinear',
                                                                                      max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+LogTr+MinMax')

    # LR  + LogTransformation + Normalization
    steps = [('t1', LogTransform()), ('t2', Normalizer()), ('m', LogisticRegression(solver='liblinear',
                                                                                    max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+LogTr+Norm')

    # LR  + LogTransformation + StandardScaler
    steps = [('t1', LogTransform()), ('t2', StandardScaler()), ('m', LogisticRegression(solver='liblinear',
                                                                                        max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+LogTr+StdScaling')

    # LR + LogTransformation + PowerTransform
    steps = [('t1', LogTransform()), ('t2', PowerTransformer()),
             ('m', LogisticRegression(solver='liblinear',
                                      max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+logTr+PowerTr')

    # LR + PowerTransformation + MinMaxScaling
    # TRIED IT, BUT DID NOT WORK!!! -> Div by zero error

    # LR  + PowerTransformation + Normalization
    # TRIED IT, BUT DID NOT WORK!!! -> Div by zero error

    # LR + MinMaxScaling + Normalization + PowerTransformation
    steps = [('t1', MinMaxScaler()), ('t2', Normalizer()), ('t3', PowerTransformer()),
             ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+MinMax+Norm+PowerTr')

    # LR + MinMaxScaling + Normalization + LogTransformation
    # TRIED IT, BUT DID NOT WORK!!! -> Div by zero error

    # LR + MinMaxScaling + LogTransformation + Normalization
    # TRIED IT, BUT DID NOT WORK!!! -> Div by zero error

    # LR + StandardScaler + Normalization + PowerTransformation
    steps = [('t1', StandardScaler()), ('t2', Normalizer()), ('t3', PowerTransformer()),
             ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+StdScaling+Norm+PowerTr')

    # LR + StandardScaler + Normalization + LogTransformation
    # TRIED IT, BUT DID NOT WORK!!! NaN errors

    # LR + LogTransformation + StandardScaler + Normalization
    steps = [('t1', LogTransform()), ('t2', StandardScaler()), ('t3', Normalizer()),
             ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+logTr+StdScaling+Norm')

    # LR + LogTransformation + Normalization + StandardScaler
    steps = [('t1', LogTransform()), ('t2', Normalizer()), ('t3', StandardScaler()),
             ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+logTr+Norm+StdScaling')

    # LR + LogTransformation + Normalization + PowerTransformation
    steps = [('t1', LogTransform()), ('t2', Normalizer()), ('t3', PowerTransformer()),
             ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('+logTr+Norm+PowerTr')

    return models, names, file_to_save, description


def basic_models_with_normalization_and_logtransf():
    """
    Defines the basic classifier models with normalization and log-transformation
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
    """
    file_to_save = 'Comparison_Basic_Classifiers_Performance_with_Normalization_LogTransformation'
    description = 'Comparison of several basic classifier methods with Normalization and LogTransformation'
    models, names = list(), list()
    max_iter = 500

    # LR  + Normalization + PowerTransformation
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('LR')

    # LDA
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', LinearDiscriminantAnalysis())]
    models.append(Pipeline(steps=steps))
    names.append('LDA')

    # KNN
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', KNeighborsClassifier())]
    models.append(Pipeline(steps=steps))
    names.append('KNN')

    # DecisionTree
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', DecisionTreeClassifier())]
    models.append(Pipeline(steps=steps))
    names.append('DecisionTree')

    # QuadraticDiscriminantAnalysis
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', QuadraticDiscriminantAnalysis())]
    models.append(Pipeline(steps=steps))
    names.append('QDA')

    # GaussianNaiveBayes
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', GaussianNB())]
    models.append(Pipeline(steps=steps))
    names.append('GaussianNB')

    # Multinomial Naieve Bayes -> Didn't work!!!
    # steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', MultinomialNB())]
    # models.append(Pipeline(steps=steps))
    # names.append('MNB')

    # GPC
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', GaussianProcessClassifier())]
    models.append(Pipeline(steps=steps))
    names.append('GaussianProc')

    # SVM
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', SVC(gamma='scale'))]
    models.append(Pipeline(steps=steps))
    names.append('SVM')

    return models, names, file_to_save, description


def basic_models_with_logtransf():
    """
    Defines the basic classifier models with LogTransformation
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
    """
    file_to_save = 'Comparison_Basic_Classifiers_Performance_with_LogTransformation'
    description = 'Comparison of several basic classifier methods with LogTransformation'
    models, names = list(), list()
    max_iter = 500

    # LR  + LogTransformation
    steps = [('t', LogTransform()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    models.append(Pipeline(steps=steps))
    names.append('LR')

    # LDA
    steps = [('t', LogTransform()), ('m', LinearDiscriminantAnalysis())]
    models.append(Pipeline(steps=steps))
    names.append('LDA')

    # KNN
    steps = [('t', LogTransform()), ('m', KNeighborsClassifier())]
    models.append(Pipeline(steps=steps))
    names.append('KNN')

    # DecisionTree
    steps = [('t', LogTransform()), ('m', DecisionTreeClassifier())]
    models.append(Pipeline(steps=steps))
    names.append('DecisionTree')

    # QuadraticDiscriminantAnalysis
    steps = [('t', LogTransform()), ('m', QuadraticDiscriminantAnalysis())]
    models.append(Pipeline(steps=steps))
    names.append('QDA')

    # GaussianNaiveBayes
    steps = [('t', LogTransform()), ('m', GaussianNB())]
    models.append(Pipeline(steps=steps))
    names.append('GaussianNB')

    # Multinomial Naieve Bayes -> Didn't work!!!
    #steps = [('t', LogTransform()), ('m', MultinomialNB())]
    #models.append(Pipeline(steps=steps))
    #names.append('MNB')

    # GPC
    steps = [('t', LogTransform()), ('m', GaussianProcessClassifier())]
    models.append(Pipeline(steps=steps))
    names.append('GaussianProc')

    # SVM
    steps = [('t', LogTransform()), ('m', SVC(gamma='scale'))]
    models.append(Pipeline(steps=steps))
    names.append('SVM')

    return models, names, file_to_save, description


def oversampling_models():
    """
    Defines the default oversampling models (that increase the samples in the minority class).
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
    """
    file_to_save = 'Comparison_OverSampling_methods_with_LogReg_dataset'
    description = 'Comparison of oversampling methods on the dataset using the LogisticRegression Classifier'
    models, names = list(), list()
    max_iter = 500

    # Random Oversampler
    sampling = RandomUnderSampler(sampling_strategy='majority')
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('ROS')

    # SMOTE
    sampling = SMOTE()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('SMOTE')

    # Borderline SMOTE
    sampling = BorderlineSMOTE()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BorderSMOTE')

    # BorderLine SMOTE with Support Vector Machines
    sampling = SVMSMOTE()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('SVMSMOTE')

    # ADASYN
    sampling = ADASYN()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('ADASYN')

    return models, names, file_to_save, description


def undersampling_models():
    """
    Defines the default undersampling models (that decrease the samples in the majority class).
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
    """
    file_to_save = 'Comparison_UnderSampling_methods_with_LogReg_dataset'
    description = 'Comparison of undersampling methods on the dataset using the LogisticRegression Classifier'
    models, names = list(), list()
    max_iter = 500

    # Random Undersampler
    sampling = RandomUnderSampler(sampling_strategy='majority')
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('RUS')

    # Near Miss version 3 (only keeps majority samples closest to all minority samples)
    sampling = NearMiss(version=3, n_neighbors=3)
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('NearMiss')

    # Condensed Nearest Neighbours
    sampling = CondensedNearestNeighbour(n_neighbors=1)
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('CNN')

    # Tomek Links
    sampling = TomekLinks()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('Tomek')

    # Edited Nearest Neighbours
    sampling = EditedNearestNeighbours(n_neighbors=3)
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('ENN')

    # RENN
    sampling = RepeatedEditedNearestNeighbours()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('RENN')

    # OSS
    sampling = OneSidedSelection()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('OneSided')

    # NCR
    sampling = NeighbourhoodCleaningRule()
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('NeighClean')

    return models, names, file_to_save, description


def ensemble_models():
    """
    Defines several ensemble models.
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
   """
    file_to_save = 'Comparison_Several_Ensember_methods_on_dataset'
    description = 'Performance comparison of several ensemble methods on the dataset'
    models, names = list(), list()

    # Bagging
    models.append(BaggingClassifier(n_estimators=1000))
    names.append('Bagging')

    # RandomForest
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RandomForest')

    # ExtraTrees
    models.append(ExtraTreesClassifier(n_estimators=1000))
    names.append('ExtraTrees')

    # RusBoost
    models.append(RUSBoostClassifier(n_estimators=1000))
    names.append('RusBoost')

    # Balanced Bagging
    models.append(BalancedRandomForestClassifier(n_estimators=1000))
    names.append('BalancedBagging')

    # GradientBoosting
    models.append(GradientBoostingClassifier(n_estimators=1000))
    names.append('GradientBoosting')

    # EasyEnsemble
    models.append(EasyEnsembleClassifier(n_estimators=1000))
    names.append('EasyEnsemble')

    # XGBoost
    models.append(XGBClassifier(n_estimators=1000, use_label_encoder=False, verbosity=0))
    names.append('XGBoost')

    # Adaboost
    models.append(AdaBoostClassifier(n_estimators=1000))
    names.append('AdaBoost')

    return models, names, file_to_save, description


def ensemble_models_with_logtransformation():
    """
    Defines several ensemble models with log transformation of the dataset.
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
   """
    file_to_save = 'Comparison_Several_Ensember_methods_on_LogTransformed_dataset'
    description = 'Performance comparison of several ensemble methods on the LogTransformed dataset'
    models, names = list(), list()

    # Bagging
    steps = [('t', LogTransform()), ('m', BaggingClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('Bagging')

    # RF
    steps = [('t', LogTransform()), ('m', RandomForestClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('RandomForest')

    # ExtraTrees
    steps = [('t', LogTransform()), ('m', ExtraTreesClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('ExtraTrees')

    # RusBoost
    steps = [('t', LogTransform()), ('m', RUSBoostClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('RusBoost')

    # Balanced Bagging
    steps = [('t', LogTransform()), ('m', BalancedRandomForestClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('BalancedBagging')

    # GradientBoosting
    steps = [('t', LogTransform()), ('m', GradientBoostingClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('GradientBoosting')

    # EasyEnsemble
    steps = [('t', LogTransform()), ('m', EasyEnsembleClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('EasyEnsemble')

    # XGBoost
    steps = [('t', LogTransform()),
             ('m', XGBClassifier(n_estimators=1000, use_label_encoder=False, verbosity=0))]
    models.append(Pipeline(steps=steps))
    names.append('XGBoost')

    # Adaboost
    steps = [('t', LogTransform()), ('m', AdaBoostClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('AdaBoost')

    return models, names, file_to_save, description


def ensemble_models_with_normalization_logtransformation():
    """
    Defines several ensemble models with normalization + log transformation of the dataset.
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
   """
    file_to_save = 'Comparison_Several_Ensember_methods_on_norm_LogTransformed_dataset'
    description = 'Performance comparison of several ensemble methods on the normalized plus LogTransformed dataset'
    models, names = list(), list()

    # Bagging
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', BaggingClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('Bagging')

    # RF
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', RandomForestClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('RandomForest')

    # ExtraTrees
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', ExtraTreesClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('ExtraTrees')

    # RusBoost
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', RUSBoostClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('RusBoost')

    # Balanced Bagging
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', BalancedRandomForestClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('BalancedBagging')

    # GradientBoosting
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', GradientBoostingClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('GradientBoosting')

    # EasyEnsemble
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', EasyEnsembleClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('EasyEnsemble')

    # XGBoost
    steps = [('t1', Normalizer()), ('t2', LogTransform()),
             ('m', XGBClassifier(n_estimators=1000, use_label_encoder=False, verbosity=0))]
    models.append(Pipeline(steps=steps))
    names.append('XGBoost')

    # Adaboost
    steps = [('t1', Normalizer()), ('t2', LogTransform()), ('m', AdaBoostClassifier(n_estimators=1000))]
    models.append(Pipeline(steps=steps))
    names.append('AdaBoost')

    return models, names, file_to_save, description


def BorderSMOTE_SelectKBest_with_logtransformation():
    """
    Defines several BorderSmote plus SelectKBest feature selection variants on the logtransformed dataset.
    (Attempt to create an alternative feature selection method based on SelectKbest. Was no improvement)
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    description: str, description of the goal of this function to print to screen
   """
    file_to_save = 'Comparison_BorderSMOTE_SelectKbest-Variants_on_LogTr_dataset'
    description = 'Performance comparison of BorderSMOTE with several SelectKBest variants on the LogTransformed dataset'
    models, names = list(), list()
    max_iter = 500

    # BorderSMOTE
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BorderSMOTE')

    # BorderSMOTE+SelectKBest chi2 k=10
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=10)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k10')

    # BorderSMOTE+SelectKBest f_classif k=10
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=10)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k10')

    # BorderSMOTE+SelectKBest mic k=10
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()),
             ('s2', SelectKBest(score_func=mutual_info_classif, k=10)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k10')

    # BorderSMOTE+SelectKBest chi2 k=20
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=20)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k20')

    # BorderSMOTE+SelectKBest f_classif k=20
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=20)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k20')

    # BorderSMOTE+SelectKBest mic k=20
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=mutual_info_classif, k=20)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k20')

    # BorderSMOTE+SelectKBest chi2 k=50
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=50)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k50')

    # BorderSMOTE+SelectKBest f_classif k=50
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=50)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k50')

    # BorderSMOTE+SelectKBest mic k=50
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=mutual_info_classif, k=50)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k50')

    # BorderSMOTE+SelectKBest chi2 k=100
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=100)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k100')

    # BorderSMOTE+SelectKBest f_classif k=100
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=100)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k100')

    # BorderSMOTE+SelectKBest mic k=100
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()),
             ('s2', SelectKBest(score_func=mutual_info_classif, k=100)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k100')

    # BorderSMOTE+SelectKBest chi2 k=200
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=200)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k200')

    # BorderSMOTE+SelectKBest f_classif k=200
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=200)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k200')

    # BorderSMOTE+SelectKBest mic k=200
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()),
             ('s2', SelectKBest(score_func=mutual_info_classif, k=200)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k200')

    # BorderSMOTE+SelectKBest chi2 k=250
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=250)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k250')

    # BorderSMOTE+SelectKBest f_classif k=250
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=250)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k250')

    # BorderSMOTE+SelectKBest mic k=250
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()),
             ('s2', SelectKBest(score_func=mutual_info_classif, k=250)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k250')

    # BorderSMOTE+SelectKBest chi2 k=260
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=chi2, k=260)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_chi_k260')

    # BorderSMOTE+SelectKBest f_classif k=260
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()), ('s2', SelectKBest(score_func=f_classif, k=260)),
             ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_f_class_k260')

    # BorderSMOTE+SelectKBest mic k=260
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('t', LogTransform()), ('s1', BorderlineSMOTE()),
             ('s2', SelectKBest(score_func=mutual_info_classif, k=260)), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('BSMOTE+SKBest_mic_k260')

    return models, names, file_to_save, description


def combination_models_LR():
    """
    Defines several over/undersampling combination tested with LogisticRegression
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    """
    file_to_save = 'Comparison_Several_Over_Undersampling_methods_LR'
    description = 'LogisticRegression performance tested with several over and undersampling strategies'
    models, names = list(), list()
    max_iter = 1000

    # classification model:
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)

    # Basic Logistic Regression
    models.append(model)
    names.append('LR_basic')

    # Oversamplers
    # SMOTE
    steps = [('e', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE')

    # BorderSMOTE
    steps = [('e', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE')

    # ADASYN
    steps = [('e', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN')


    # undersamplers:
    # Tomek
    steps = [('e', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+Tomek')

    # Edited Nearest Neighbours
    sampling = EditedNearestNeighbours(n_neighbors=3)
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    steps = [('e', sampling), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ENN')

    # OSS
    steps = [('e', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+OneSided')

    # NeighClean
    steps = [('e', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+NeighClean')

    # combis Over and Undersampling
    # SMOTE + Tomek
    steps = [('e1', SMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+Tomek')

    # SMOTE + ENN
    steps = [('e1', SMOTE()), ('e2', EditedNearestNeighbours(sampling_strategy='majority')), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+ENN')

    # SMOTE + OneSided
    steps = [('e1', SMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+OneSided')

    # SMOTE + Neighclean
    steps = [('e1', SMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+NeighClean')

    # BorderSMOTE + Tomek
    steps = [('e1', BorderlineSMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+Tomek')

    # BorderSMOTE + ENN
    steps = [('e1', BorderlineSMOTE()), ('e2', EditedNearestNeighbours(sampling_strategy='majority')), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+ENN')

    # BorderSMOTE + OneSided
    steps = [('e1', BorderlineSMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+OneSided')

    # BorderSMOTE + Neighclean
    steps = [('e1', BorderlineSMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+NeighClean')

    # ADASYN + Tomek
    steps = [('e1', ADASYN()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+Tomek')

    # ADASYN + ENN
    steps = [('e1', ADASYN()), ('e2', EditedNearestNeighbours(sampling_strategy='majority')), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+ENN')

    # ADASYN + OneSided
    steps = [('e1', ADASYN()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+OneSided')

    # ADASYN + NeighClean
    steps = [('e1', ADASYN()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+NeighClean')


    # Reversed order

    # Tomek + SMOTE
    steps = [('e1', TomekLinks()), ('e2', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+Tomek+SMOTE')

    # ENN + SMOTE
    steps = [('e1', EditedNearestNeighbours(sampling_strategy='majority')), ('e2', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ENN+SMOTE')

    # OneSided + SMOTE
    steps = [('e1', OneSidedSelection()), ('e2', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+OneSided+ SMOTE')

    # Neighclean + SMOTE
    steps = [('e1', NeighbourhoodCleaningRule()), ('e2', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+NeighClean+SMOTE')

    # Tomek + BorderSMOTE
    steps = [('e1', TomekLinks()), ('e2', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+Tomek+BSMOTE')

    # ENN + BorderSMOTE
    steps = [('e1', EditedNearestNeighbours(sampling_strategy='majority')), ('e2', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ENN+BSMOTE')

    # OneSided + BorderSMOTE
    steps = [('e1', OneSidedSelection()), ('e2', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+OneSided+BSMOTE')

    # Neighclean + BorderSMOTE
    steps = [('e1', NeighbourhoodCleaningRule()), ('e2', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+NeighClean+BSMOTE')

    # Tomek + ADASYN
    steps = [('e1', TomekLinks()), ('e2', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+Tomek+ADASYN')

    # ENN + ADASYN
    steps = [('e1', EditedNearestNeighbours(sampling_strategy='majority')), ('e2', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ENN+ADASYN')

    # OneSided + ADASYN
    steps = [('e1', OneSidedSelection()), ('e2', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+OneSided+ADASYN')

    # NeighClean + ADASYN
    steps = [('e1', NeighbourhoodCleaningRule()), ('e2', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+NeighClean+ADASYN')

    return models, names, file_to_save, description


def combination_models_LDA():
    """
    Defines several over/undersamplers tested with LinearDiscriminantAnalysis
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    """
    file_to_save = 'Comparison_Several_Over_Undersampling_methods_LDA_svd'
    description = 'LDA performance tested with several over and undersampling strategies'
    models, names = list(), list()
    max_iter = 1000

    # classification model:
    model = LinearDiscriminantAnalysis(solver='svd')

    # Basic Logistic Regression
    models.append(model)
    names.append('basic_classifier')

    # Oversamplers
    # SMOTE
    steps = [('e', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE')

    # BorderSMOTE
    steps = [('e', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE')

    # ADASYN
    steps = [('e', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN')

    # combis Over and Undersampling
    # SMOTE + Tomek
    steps = [('e1', SMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+Tomek')

    # SMOTE + OneSided
    steps = [('e1', SMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+OneSided')

    # SMOTE + Neighclean
    steps = [('e1', SMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+NeighClean')

    # BorderSMOTE + Tomek
    steps = [('e1', BorderlineSMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+Tomek')

    # BorderSMOTE + OneSided
    steps = [('e1', BorderlineSMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+OneSided')

    # BorderSMOTE + Neighclean
    steps = [('e1', BorderlineSMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+NeighClean')

    # ADASYN + Tomek
    steps = [('e1', ADASYN()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+Tomek')

    # ADASYN + OneSided
    steps = [('e1', ADASYN()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+OneSided')

    # ADASYN + NeighClean
    steps = [('e1', ADASYN()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+NeighClean')

    return models, names, file_to_save, description


def combination_models_GaussianProc():
    """
    Defines several over/undersamplers tested with GaussianProcessClassifier
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    """
    file_to_save = 'Comparison_Several_Over_Undersampling_methods_GaussianProc_DotProduct'
    description = 'GaussianProc performance tested with several over and undersampling strategies'
    models, names = list(), list()
    max_iter = 1000

    # classification model:
    model = GaussianProcessClassifier(kernel=DotProduct(sigma_0=1))

    # Basic Logistic Regression
    models.append(model)
    names.append('basic_classifier')

    # Oversamplers
    # SMOTE
    steps = [('e', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE')

    # BorderSMOTE
    steps = [('e', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BorderSMOTE')

    # ADASYN
    steps = [('e', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN')

    # combis Over and Undersampling
    # SMOTE + Tomek
    steps = [('e1', SMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+Tomek')

    # SMOTE + OneSided
    steps = [('e1', SMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+OneSided')

    # SMOTE + Neighclean
    steps = [('e1', SMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+NeighClean')

    # BorderSMOTE + Tomek
    steps = [('e1', BorderlineSMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+Tomek')

    # BorderSMOTE + OneSided
    steps = [('e1', BorderlineSMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+OneSided')

    # BorderSMOTE + Neighclean
    steps = [('e1', BorderlineSMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BSMOTE+NeighClean')

    # ADASYN + Tomek
    steps = [('e1', ADASYN()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+Tomek')

    # ADASYN + OneSided
    steps = [('e1', ADASYN()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+OneSided')

    # ADASYN + NeighClean
    steps = [('e1', ADASYN()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+NeighClean')

    return models, names, file_to_save, description


def combination_models_ADABOOST():
    """
    Defines several over/undersamplers tested with Adaboost.
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    """
    #file_to_save = 'Comparison_Several_Over_Undersampling_methods_ADABOOST_LR1_1'
    file_to_save = 'Comparison_Several_Over_Undersampling_methods_ADABOOST_LogRegr'
    description = 'AdaBoost performance tested with several over and undersampling strategies'
    models, names = list(), list()

    # classification model:
    #model = AdaBoostClassifier(n_estimators=1000)
    #model = AdaBoostClassifier(learning_rate=1.1, n_estimators=1000)
    model = AdaBoostClassifier(base_estimator=LogisticRegression(solver='liblinear'), learning_rate=1.4, n_estimators=50)
    #model = AdaCostClassifier(learning_rate=1.1, n_estimators=2000)

    # Basic Logistic Regression
    models.append(model)
    names.append('basic_classifier')

    # Oversamplers
    # SMOTE
    steps = [('e', SMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE')

    # BorderSMOTE
    steps = [('e', BorderlineSMOTE()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BorderSMOTE')

    # ADASYN
    steps = [('e', ADASYN()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN')

    # combis Over and Undersampling
    # SMOTE + Tomek
    steps = [('e1', SMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+Tomek')

    # SMOTE + OneSided
    steps = [('e1', SMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+OneSided')

    # SMOTE + Neighclean
    steps = [('e1', SMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+SMOTE+NeighClean')

    # BorderSMOTE + Tomek
    steps = [('e1', BorderlineSMOTE()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BorderSMOTE+Tomek')

    # BorderSMOTE + OneSided
    steps = [('e1', BorderlineSMOTE()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BorderSMOTE+OneSided')

    # BorderSMOTE + Neighclean
    steps = [('e1', BorderlineSMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+BorderSMOTE+NeighClean')

    # ADASYN + Tomek
    steps = [('e1', ADASYN()), ('e2', TomekLinks()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+Tomek')

    # ADASYN + OneSided
    steps = [('e1', ADASYN()), ('e2', OneSidedSelection()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+OneSided')

    # ADASYN + NeighClean
    steps = [('e1', ADASYN()), ('e2', NeighbourhoodCleaningRule()), ('m', model)]
    models.append(Pipeline(steps=steps))
    names.append('+ADASYN+NeighClean')

    return models, names, file_to_save, description



def cost_sensitive_models():
    """
    Defines several classifiers with cost sensitive learning.
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    file_to_save: str, filename for the plot saving.
    """
    file_to_save = 'Comparison_Cost_Sensitive_methods'
    description = 'Test of classifiers with built-in cost-sensitive learning'
    models, names = list(), list()
    max_iter = 1000

    # Basic LogisticRegression
    models.append(LogisticRegression(solver='liblinear', max_iter=max_iter))
    names.append('LR_basic')

    # Basic Support Vector Machines
    models.append(SVC(gamma='scale'))
    names.append('SVM_basic')

    # Basic Random Forrest
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RF_basic')

    # Basic DecisionTreeClassifier
    models.append(DecisionTreeClassifier())
    names.append('DecisionTree_basic')

    # Basic ExtraTrees
    models.append(ExtraTreesClassifier(n_estimators=1000))
    names.append('ExtraTrees_basic')

    # Basic XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    models.append(model)
    names.append('XGBoost_basic')


    # LogisticRegression with cost sensitive learning
    models.append(LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=max_iter))
    names.append ('LR_CSL')

    # Support Vector Machines with csl
    models.append( SVC(gamma='scale', class_weight='balanced'))
    names.append('SVM_CSL')

    #Random Forrest with cost sensitive learning
    models.append(RandomForestClassifier(class_weight='balanced', n_estimators=1000))
    names.append('RF_CSL')

    # DecisionTreeClassifier + csl
    models.append(DecisionTreeClassifier(class_weight='balanced'))
    names.append('DecisionTree_CSL')

    # ExtraTrees with cost sensitive learning
    models.append(ExtraTreesClassifier(class_weight='balanced', n_estimators=1000))
    names.append('ExtraTrees_CSL')

    # XGBoost with csl
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=81.5)
    models.append(model)
    names.append('XGBoost_CSL')

    return models, names, file_to_save, description


####################################
# Grid search functions

def xgboost_grid_search(df):
    """
    This function determines the optimal scale_pos_weight hyperparameter value
    needed for the cost-senstive learning of the XGBoost classifier.
    :param df: pandas dataframe, contains the dataset
    :return:
    print of the Estimated class imbalance.
    print of the best scale_pos_weight value with its F1-score <- use this one!
    print of an overview of all tested scale_pos_weight values with their F1-scores
    """
    sample_names, x, y = get_x_y(df)

    counter = Counter(y)
    estimate = counter[0]/ counter[1]
    print('Estimated scale_pos_weight value: %.3f' %estimate)

    weights = [1, 10, 18.24, 25, 50, 75, 80, 81, 81.1, 81.2, 81.3, 81.4, 81.5, 81.6, 81.76, 99, 100]
    param_grid = dict(scale_pos_weight=weights)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
    grid_result = grid.fit(x, y)
    print('Best: %f using %s' %(grid_result.best_score_,grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stdevs = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param, in zip(means, stdevs, params):
        print('%f (%f) with %r' %(mean,stdev,param))


def adaboost_grid_search(df):
    """
    This function determines the optimal hyperparameters for Adaboost.
    The function is highly based on: https://machinelearningmastery.com/adaboost-ensemble-in-python/
    Results are printed to screen.
    input:
    df: pandas dataframe, contains the dataset
    output:
    screen print:
    * Best found hyperparameters
    * Overview of achieved F1 scores with all tested hyperparameter combis
    """
    print('#' * 50 + '\nRunning Adaboost hyperparameter grid search')
    print('Warning! This can take a looong time to complete')
    sample_names, x, y = get_x_y(df)

    # Tried different base estimators (default is decisiontree)
    #model = AdaBoostClassifier(base_estimator=LogisticRegression())
    #model = AdaBoostClassifier(base_estimator=RUSBoostClassifier())
    model = AdaBoostClassifier()
    print('start time: %s'% asctime())
    # define parameters to vary
    grid = dict()
    grid['n_estimators'] = [1100]
    #grid['learning_rate'] = [i for i in np.arange (0.8, 1.5, 0.1)]
    grid['learning_rate'] = [1.1]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator= model, param_grid=grid, n_jobs=1, cv=cv, scoring='f1')
    grid_result = grid_search.fit(x,y)
    # summarize the best score and configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print('end time: %s'% asctime())

def lda_grid_search(df):
    """ This function tries some extra hyperparameters for optimal performance of LinearDiscriminantAnalysis.
    The function is highly based on: https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
    Results are printed to screen.
    input:
    df: pandas dataframe, contains the dataset
    output:
    screen print:
    * Best found hyperparameters
    * Overview of achieved scores with all tested hyperparameter combis
    """
    print('#' * 50 + '\nRunning LDA hyperparameter grid search')
    sample_names, x, y = get_x_y(df)
    model = LinearDiscriminantAnalysis()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    # define grid
    grid = dict()
    grid['solver'] = ['svd', 'lsqr', 'eigen']
    # define search
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform search
    results = search.fit(x,y)
    # summary
    print('Mean accuracy: %.3f' % results.best_score_)
    print('Config: %s' %results.best_params_)


def gaussianprocessclassifier_grid_search(df):
    """ This function tries some extra hyperparameters for optimal performance of GaussianProcessClassifier.
    The function is highly based on: https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/
    Results are printed to screen.
    input:
    df: pandas dataframe, contains the dataset
    output:
    screen print:
    * Best found hyperparameters
    * Overview of achieved scores with all tested hyperparameter combis
    """
    print('#' * 50 + '\nRunning GaussianProcessClassifier hyperparameter grid search')
    sample_names, x, y = get_x_y(df)
    model = GaussianProcessClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    # define grid
    grid = dict()
    grid['kernel'] = [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(), 1 * WhiteKernel()]
    # define search
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x, y)
    # summarize best
    print('Best Mean Accuracy: %.3f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.3f with: %r" % (mean, param))


#####################
# Prediction functions
def prediction_models_LR():
    """
    Defines several LogisticRegression prediction models to be used for the test-dataset predictions
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    """
    models, names = list(), list()
    max_iter = 1000

    # basic Logistic Regression
    model = LogisticRegression(solver='liblinear', max_iter=max_iter)
    models.append(model)
    names.append('LR')

    # +SMOTE
    steps = [('e1', SMOTE()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+SMOTE')

    # +SMOTE+NeighborCleaningRule
    steps = [('e1', SMOTE()), ('e2', NeighbourhoodCleaningRule()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+SMOTE+NCR')

    # BorderSMOTE
    steps = [('e1', BorderlineSMOTE()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+BSMOTE')

    # +BSMOTE+Tomek
    steps = [('e1', BorderlineSMOTE()), ('e2', TomekLinks()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+BSMOTE+Tomek')

    # ADASYN
    steps = [('e1', ADASYN()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+ADASYN')

    # ADASYN + Tomek
    steps = [('e1', ADASYN()), ('e2', TomekLinks()), ('m', LogisticRegression(solver='liblinear', max_iter=max_iter))]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+ADASYN+Tomek')

    # Logistic regression with Cost-senstive learning
    model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=max_iter)
    models.append(model)
    names.append('LR(CSL)')

    return models, names


def prediction_models_GPC():
    """
    Defines several GaussianProcessClassifier prediction models to be used for the test-dataset predictions
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    """
    classifier_model = GaussianProcessClassifier(kernel=DotProduct(sigma_0=1))
    models, names = list(), list()

    # GaussianProcessClassifier(kernel=DotProduct)
    models.append(classifier_model)
    names.append('GPC(DotProduct)')

    # +SMOTE
    steps = [('e1', SMOTE()),
             ('m', classifier_model)]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+SMOTE')

    # +Borderline-SMOTE
    steps = [('e1', BorderlineSMOTE()),
             ('m', classifier_model)]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+BSMOTE')

    # +ADASYN
    steps = [('e1', ADASYN()), ('m', classifier_model)]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+ADASYN')

    # +SMOTE + NCR
    steps = [('e1', SMOTE()), ('e2', NeighbourhoodCleaningRule()),
             ('m', classifier_model)]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+SMOTE+NCR')

    # +Borderline-SMOTE + Tomek
    steps = [('e1', BorderlineSMOTE()), ('e2', TomekLinks()),
             ('m', classifier_model)]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+BSMOTE+Tomek')

    # +ADASYN
    steps = [('e1', ADASYN()), ('e2', TomekLinks()), ('m', classifier_model)]
    model = Pipeline(steps=steps)
    models.append(model)
    names.append('+ADASYN+Tomek')

    # LDA(solver='svd')
    classifier_model = LinearDiscriminantAnalysis(solver='svd')
    models.append(classifier_model)
    names.append('LDA(svd)')

    return models, names


def prediction_models_LDA():
    """
    Defines several prediction models to be used for the test-dataset predictions
    output:
    models: lst, contains model settings
    names:  lst, contains model names
    """

    models, names = list(), list()

    # basic Logistic Regression
    models.append(LogisticRegression(solver='liblinear', max_iter=1000))
    names.append('LR')

    # GaussianProcessClassifier(kernel=DotProduct)
    models.append(GaussianProcessClassifier(kernel=DotProduct(sigma_0=1)))
    names.append('GPC(DotProduct)')

    # LDA(solver='svd')
    models.append(LinearDiscriminantAnalysis(solver='svd'))
    names.append('LDA(svd)')

    return models, names


######################
# Model test functions

def evaluate_model(df, model, scoring, n_splits=10, n_repeats=10):
    """
    Tests a given model several times on the X and y values,
    by repeatedly (n_repeats) splitting the dataset in (n_splits) folds
    for training and testing, and returns the mean and stdev of the
    the given metric.
    input:
    X: np array, contains the non-collinear features
    y, np array, contains the class labels
    model: lst, contains the models with desired settings
    metric: str, desired metric for the model
    n_splits=10: int, number of folds for X_train and X_test division (default set at 10)
    n_repeats=3: int, number of times the fold splitting should be repeated (default set at 10)
    output:
    scores: np.array, contains the metric scores of the model
    conf_matrix: np.array, contains the confusion matrix
    """
    # get x and y from dataframe
    sample_names, x, y = get_x_y(df)


    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    # random_state guarantees that the splits are reproducible, so that model results are comparable.

    # evaluate the model on the dataset
    scores = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    return scores


def test_models(df, models, names, scoring, location, file_to_save, description):
    """
    Tests all the models described in names and models, with the given scoring metrics,
    by calling the evaluate_model function for each model.
    Prints the performance of each model to screen with the mean and standard dev of the metrics.
    Input:
    X: np array, contains the non-collinear features
    y, np array, contains the class labels
    models: lst, contains the models with desired settings
    names: lst, contains the model names
    scoring: dict, contains the scoring metrics to determine the classifier's performance
    location: str, subdirectory where the results of this test should be saved
    file_to_save: str, name of the file to store the test results into
    description: str, contains a description of the model-collection's purpose for he user
    output:
    print of each model's performance with the given metric
    visualisation of all model performances in the form of a box-whisker plot.
    out_file: csv-file, contains the model's performances
    """
    # Print information to screen and save to file
    out_file = TABLE_LOCATION + location + file_to_save + ".csv"
    out_file2 = TABLE_LOCATION + location + 'F1_scores_' + file_to_save + ".csv"
    with open(out_file, 'w') as out_fn:
        with open(out_file2, 'w') as out_fn2:
            print('#' * 50 + '\n' + str(description))
            out_fn.write(description + "\n")
            out_fn2.write(description + "\n")
            out_fn2.write(',F1-scores' + "\n")
            print('Metric:                      F1          Precision   Recall      PR_AUC      ROC_AUC      BrierSkill    TP   FN   FP   TN')
            out_fn.write(",F1,,Precision,,Recall,,PR_AUC,,ROC_AUC,,BrierSkill,,TP,FN,FP,TN\n")
            print('Method                       mean  std   mean  std   mean  std   mean  std   mean  std    mean  std')
            out_fn.write("Method,mean,std,mean,std,mean,std,mean,std,mean,std,mean,std\n")
            results = []

            # evaluate each model
            for i in range(len(models)):
                # evaluate the model and store results
                scores = evaluate_model(df, models[i], scoring, 10, 10)
                results.append(scores['test_f1'])
                # store and print
                out_fn2.write('%s,' % names[i])
                out_fn2.write(','.join(str(i) for i in scores['test_f1']) + '\n')

                # summarize and store the scoring results
                print('%-28s %5.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %6.3f %.3f %4.1f %4.1f %4.1f %4.1f' %
                (names[i],
                np.mean(scores['test_f1']), np.std(scores['test_f1']),
                np.mean(scores['test_precision']), np.std(scores['test_precision']),
                np.mean(scores['test_recall']), np.std(scores['test_recall']),
                np.mean(scores['test_pr_auc']), np.std(scores['test_pr_auc']),
                np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc']),
                np.mean(scores['test_brier_skill_score']), np.std(scores['test_brier_skill_score']),
                np.mean(scores['test_TP']), np.mean(scores['test_FN']),
                np.mean(scores['test_FP']), np.mean(scores['test_TN'])))

                out_fn.write('%-22s,%5.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f\n' %
                (names[i],
                np.mean(scores['test_f1']), np.std(scores['test_f1']),
                np.mean(scores['test_precision']), np.std(scores['test_precision']),
                np.mean(scores['test_recall']), np.std(scores['test_recall']),
                np.mean(scores['test_pr_auc']), np.std(scores['test_pr_auc']),
                np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc']),
                np.mean(scores['test_brier_skill_score']), np.std(scores['test_brier_skill_score']),
                np.mean(scores['test_TP']), np.mean(scores['test_FN']),
                np.mean(scores['test_FP']), np.mean(scores['test_TN'])))

        out_fn2.close()
    out_fn.close()
    print('The measure results were saved to file ' + str(out_file))
    print('The F1-scores for all tests were saved to file ' + str(out_file2))
    # Create a box whisker-plot
    create_box_whisker_plot(results, names, location, file_to_save)


def test_model_collections(df, scoring, model_collections_to_test, location):
    """
    calls the test_models-function on each collection of models from <model_collections_to_test>,
    using <X>, and <y> as data-input and the desired scoring measures from <scoring>.
    Results are stored in <location>.
    input:
    X: np array, contains the non-collinear features
    y: np array, contains the class labels
    scoring: dict, contains the desired scoring measures
    model_collections_to_test: lst, contains the model collections to perform
    location: str, subdirectory in <IMG_location> and <TABLE_LOCATION> to store the results into.
    """
    # create the directory for storing experiment specific produced figures
    if not os.path.exists(IMG_LOCATION + location):
        os.makedirs(IMG_LOCATION + location)
    # create the directory for storing experiment specific result tables
    if not os.path.exists(TABLE_LOCATION + location):
        os.makedirs(TABLE_LOCATION + location)
    for model_to_test in model_collections_to_test:
        models, names, file_to_save, description = model_to_test
        test_models(df, models, names, scoring, location, file_to_save, description)


def make_predictions(df, test_set, location, file_to_save, model_function):
    """
    Predicts the class labels for the test dataset samples using the given classifier.
    Also produces the confusion matrix values (TP, FN, FP, TN) and the accuracy of the classifier
    input:
    classifier: str, desired classifier to use
    df: pandas dataframe, contains the training dataset
    test_set: pandas dataframe, contains the test dataset
    location: str, subdirectory name to save results to,
    file_to_save: str, name of file to save results into
    model_function: str, name of model-function to use
    output:
    df: pandas dataframe, contains the predicted class labels, confusion matrix values, and accuracy
    """
    print('#' * 50 + '\nMaking predictions on the test set')

    # Get the x (=features) and y (=labels) values from the test and train dataset
    _, x_train, y_train = get_x_y(df)
    feature_names = list(x_train)

    test_sample_names, x_test, y_test = get_x_y(test_set)


    # log transform the x_test values
    x_test = logtransform(x_test)

    # Only keep the relevant features (same as in train set)
    x_test = x_test.filter(items=feature_names)

    # get the models from the prediction_models function
    models, names = model_function
    results={}
    test_sample_names = test_sample_names.tolist()
    test_sample_names[0:0] = ["TP", "FN", "FP", "TN"]
    results['sample'] = test_sample_names
    for i in range(len(models)):
        # predict the class labels for the test-set samples
        models[i].fit(x_train, y_train)
        y_pred = list(models[i].predict(x_test))
        TP = TP_score(y_test, y_pred)
        FN = FN_score(y_test, y_pred)
        FP = FP_score(y_test, y_pred)
        TN = TN_score(y_test, y_pred)

        y_pred[0:0] = [TP, FN, FP, TN] # insert TP, FN, FP, TN at the first position
        results[names[i]] = y_pred
    y_test = y_test.tolist()
    counter = Counter(y_test)
    estimate = counter[0] / counter[1]
    y_test[0:0] = [counter[1], 0, 0, counter[0]]
    results['y_true'] = y_test
    df = pd.DataFrame.from_dict(results)
    print(df)
    out_fn = TABLE_LOCATION + location + file_to_save
    df.to_csv(out_fn, index=False)
    print('The prediction results have been saved to the file %s' %out_fn)



def run_main():

    # create the directory for storing produced figures
    if not os.path.exists(IMG_LOCATION):
        os.makedirs(IMG_LOCATION)
    # create the directory for storing the result tables
    if not os.path.exists(TABLE_LOCATION):
        os.makedirs(TABLE_LOCATION)

    # load the dataset
    df = load_dataset(FILENAME)

    # split into train and test set
    df, test_set = train_test_splitting(df)


    #########################################
    # Feature selection

    # remove the features that are more than 95% correlated to other features
    #df = highly_correlated_feature_removal(df) -> NO IMPROVEMENT

    # Remove multicollinear features
    #df = multicollinear_feature_removal(df) -> FAILED

    # Test the LowVariance feature removal function
    # This should remove all features that have a low variance between samples.
    #low_variance_feature_removal(df, X, feature_names, 0.0)     # -> DIDN'T WORK: NO FEATURES WERE REMOVED

    #########################################
    # check if the data in the feature columns (=hormones) have a normal distribution

    # print('Shapiro-Wilk test on the raw dataset:')
    # shapiro_wilk_test(X)

    # print('d\'Agostino-Pearson test on the raw dataset:')
    # dagostino_pearson_test(X)

    #########################################
    # check if the data in the feature columns (=hormones) have a normal distribution after log transformation
    # X_log = np.log(X.astype('float64'))
    # print('Shapiro Wilk test on the log-transformed dataset:')
    # shapiro_wilk_test(X_log)

    #########################################
    # make some distribution plots for a few of the features
    # make_distribution_plots(X, feature_names, 56, 60)
    # make_distribution_plots(X_log, feature_names, 56, 60)

    # create a pairwise pearson plot
    #pairwise_pearson(df)

    # make histograms
    #fully_normalized_df.hist()
    #pyplot.show()

    # make density plots
    # df.plot (kind= 'density', subplots=True, layout =(34,8), sharex=False)
    # pyplot.show()

    # creat a Spearman correlation heatmap
    #correlation_plot(df, 'spearman')


    #########################################

    # Determining the Baseline performance of the classifiers

    # Define the scoring metrics to use for the classifier's performance comparison
    scoring = {
        'f1': make_scorer(f1_score, pos_label=1, average='binary'),
        'precision': make_scorer(precision_score, pos_label=1, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, pos_label=1, average='binary'),
        'pr_auc': make_scorer(pr_auc_score, greater_is_better=True),
        'roc_auc': make_scorer(roc_auc_score, average='weighted'),
        'brier_skill_score': make_scorer(brier_skill_score, greater_is_better=True),
        'TP': make_scorer(TP_score),
        'FN': make_scorer(FN_score),
        'FP': make_scorer(FP_score),
        'TN': make_scorer(TN_score)
    }

    # Select the collections of classification models to run
    # Several collections of models have been defined above
    # (e.q.naive_models(), normalized_models_unbalanced(), etc)
    # By adding or removing these  collections to the model_collections_to_test-list,
    # the desired analyses-to-perform can be selected.

    model_collections_to_test = [
        basic_models(),
        LogRegression_normalization_standardization_models(),
        basic_models_with_normalization_and_logtransf(),
        basic_models_with_logtransf(),
        #oversampling_models(),
        #undersampling_models(),
        ensemble_models(),
        ensemble_models_with_logtransformation(),
        ensemble_models_with_normalization_logtransformation(),
        #combination_models_LR()
    ]

    # Define the subdirectory name in IMG_LOCATION and TABLE_LOCATION to store the result into
    location = 'analysis_results_on_raw_dataset/'

    # Perform the comparison of the desired model collections, defined in test_model_collections
    #test_model_collections(df, scoring, model_collections_to_test, location)

    #########################################
    # Perform Recursive Feature Elimination,
    # with the desired classifier model for feature importance determination
    # This discards features that show no correlation with the classes

    out_fn_prefix = 'Recursive_Feature_Eliminated_LDA_dataset'
    model = LinearDiscriminantAnalysis()
    recursive_feature_eliminator(df, model, out_fn_prefix, norm=False, logtr=True)
    recursive_feature_eliminator(df, model, out_fn_prefix, norm=True, logtr=True)

    out_fn_prefix = 'Recursive_Feature_Eliminated_LR_dataset'
    model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=500)
    recursive_feature_eliminator(df, model, out_fn_prefix, norm=False, logtr=True)
    recursive_feature_eliminator(df, model, out_fn_prefix, norm=True, logtr=True)


    ##########################################################
    # Determine all classifier's performances after Recursive Feature Elimination 
    # Select the collections of classification models to run
    model_collections_to_test = [
        #basic_models(),
        #oversampling_models(),
        #undersampling_models(),
        #ensemble_models(),
        combination_models_LR(),
        combination_models_LDA(),
        combination_models_GaussianProc(),
        #combination_models_ADABOOST(),
        #cost_sensitive_models()
    ]

    # perform the comparison of the log-transformed feature-filtered dataset produced with LR
    #location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LR/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # perform the comparison of the normalized + log-transformed feature-filtered dataset produced with LR
    #location = 'analysis_results_on_normalized_logtransformed_feature_filtered_dataset_LR/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_norm_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # perform the comparison of the log-transformed feature-filtered dataset produced with LDA
    #location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LDA/'
    #filename = 'Recursive_Feature_Eliminated_LDA_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # perform the comparison of the normalized + log transformed feature-filtered dataset produced with LDA
    #location = 'analysis_results_on_normalized_logtransformed_feature_filtered_dataset_LDA/'
    #filename = 'Recursive_Feature_Eliminated_LDA_dataset_norm_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # perform the comparison of the log-transformed feature-filtered dataset produced with LR and Automatic Outlier Removal
    #location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LR_AOR/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #file_to_save = 'Recursive_Feature_Eliminated_LR_dataset_logtr_AOR.csv'
    #df = automatic_outlier_removal(df, file_to_save)
    #test_model_collections(df, scoring, model_collections_to_test, location)


    # Perform Adaboost grid search to find optimal hyperparameters
    #location = 'analysis_results_on_normalized_logtransformed_feature_filtered_dataset_LR_AOR/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr_AOR.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #adaboost_grid_search(df)

    # Perform LinearDiscriminantAnalysis grid search
    #location = 'analysis_results_on_normalized_logtransformed_feature_filtered_dataset_LR/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #lda_grid_search(df)

    # Perform GaussianProcessClassifier grid search
    #location = 'analysis_results_on_normalized_logtransformed_feature_filtered_dataset_LR/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #gaussianprocessclassifier_grid_search(df)

    #######
    # small test with Adaboost feature reduction
    #out_fn_prefix = 'Recursive_Feature_Eliminated_LR_dataset_AFE'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #df = adaboost_feature_selector(df, out_fn_prefix)

    # perform the comparison of the log-transformed feature-filtered dataset produced with LR + Adaboost Feature selection
    #location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LR_AdaboostFE/'
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_AFE.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # Test with AdaboostFE instead of RFE
    #out_fn_prefix = 'Adaboost_Feature_Eliminated_logtr'
    #sample_names, x, y = get_x_y(df)
    #x_new = logtransform(x)
    #df = pd.concat([sample_names, x_new, y], axis=1)
    #df = adaboost_feature_selector(df, out_fn_prefix)


    # perform the comparison of the log-transformed dataset Adaboost Feature selection
    #location = 'analysis_results_on_logtransformed_AdaboostFE_dataset/'
    #filename = 'Adaboost_Feature_Eliminated_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # Testing the combination of BorderSmote oversampling with SelectKBest Feature selection.
    # use SelectKBest's results to check the classifier performance
    #location = 'analysis_results_on_raw_dataset'
    #filename = 'Comparison_BorderSMOTE_SelectKbest-Variants_on_LogTr_dataset'
    #df = load_dataset(MAIN_DIR + filename)
    #test_model_collections(df, scoring, model_collections_to_test, location)

    # XGBoost grid search for the optimal scale_pos_weight hyperparameter value
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_AFE.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #xgboost_grid_search(df)

    # Testing the principal component analysis method
    #filename = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    #df = load_dataset(MAIN_DIR + filename)
    #principal_component_analysis(df)


    # Predict the class values for the samples in the kept-separate test dataset with LR.
    file_to_use = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    df = load_dataset(MAIN_DIR + file_to_use)
    result_location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LR/'
    file_to_save = 'Predictions_LR.csv'
    model_function = prediction_models_LR()
    make_predictions(df, test_set, result_location, file_to_save, model_function)

    # Predict the class values for the samples in the kept-separate test dataset with GPC.
    file_to_use = 'Recursive_Feature_Eliminated_LR_dataset_logtr.csv'
    df = load_dataset(MAIN_DIR + file_to_use)
    result_location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LR/'
    file_to_save = 'Predictions_GPC.csv'
    model_function = prediction_models_GPC()
    make_predictions(df, test_set, result_location, file_to_save, model_function)

    # Predict the class values for the samples in the kept-separate test dataset with the LDA.
    file_to_use = 'Recursive_Feature_Eliminated_LR_dataset_logtr_AOR.csv'
    df = load_dataset(MAIN_DIR + file_to_use)
    result_location = 'analysis_results_on_logtransformed_feature_filtered_dataset_LR_AOR/'
    file_to_save = 'Predictions_LDA.csv'
    model_function = prediction_models_LDA()
    make_predictions(df, test_set, result_location, file_to_save, model_function)


# main
if __name__ == "__main__":
    MAIN_DIR = 'D:/thesis/INF_data_080320/'     # main subdirectory location
    FILENAME = MAIN_DIR + 'Sample_info.csv'     # expected file containing your input data
    IMG_LOCATION = MAIN_DIR + 'Images/'         # location for storing images
    TABLE_LOCATION = MAIN_DIR + 'Tables/'       # location for storing tables
    run_main()


