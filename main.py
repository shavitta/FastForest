import pandas as pd
import numpy as np
import os
import glob
import time
import collections
import xgboost
import shap
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

from algorithms.FastTreeClassifier import FastForestClassifier, MyRandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=FutureWarning)

SEED = 33               # for randomness
SPLIT_FOLDS = 10        # the number of folds for train-test splits (the outer loop)
TUNE_FOLDS = 3          # the number of folds for the parameters tuning
VERBOSE = False         # for debug prints
MISSING = 999           # missing data code


def calc_tpr_fpr(y_true, y_prediction):
    """
    The function calculates the TPR and FPR metrics
    :param y_true: the true classification of the set
    :param y_prediction: the predicted classification of the set
    :return:TPR and FPR arrays
    """
    cnf_matrix = confusion_matrix(y_true, y_prediction)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR, FPR


def create_performance_summary_record_in_results(y_test, y_pred, algo_name, cv_index, best_parameters, training_time,
                                                 inference_time_for_1000, record_results, ds_name):
    """
    The functions calculates all the metrics required and adds it as a row to the record_results dataframe
    :param y_test: the Y test set
    :param y_pred: the predicted probabilities outputted by the forest
    :param algo_name: FastForest or RandomForest
    :param cv_index: the CV index of this run [1, ... , SPLITS_FOLDS)
    :param best_parameters: the best parameters found by the RandomizedSearchCV
    :param training_time: the measured time for the training of the forest
    :param inference_time_for_1000: the measured time for the prediction for 1000 samples
    :param record_results: the results dataframe of all the datasets
    :param ds_name: the dataset name
    :return: the updated results dataframe
    """
    is_multi_class = False
    if len(np.unique(y_test)) > 2:
        is_multi_class = True

    if y_pred.shape[-1] == 1:
        # Binary classification
        y_pred_class = (y_pred >= 0.5).astype("int")
    else:
        y_pred_class = np.argmax(y_pred, -1)

    acc = accuracy_score(y_test, y_pred_class)

    auc = 0
    pr = 0
    if is_multi_class and len(y_pred[0]) > 2:
        auc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="weighted")
        pr = calc_precision_recall(X_train, X_test, y_train, y_test, algo_name, best_parameters)
    else:
        auc = roc_auc_score(y_test, y_pred[:, 1])
        pr = average_precision_score(y_test, y_pred_class)      # PR is relevant only for binary classification
    tpr, fpr = calc_tpr_fpr(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, average="weighted")

    if VERBOSE:
        print(
            'ds_name:{}, cv:{}, best_parameters:{}, acc:{}, fpr:{}, tpr:{}, auc:{}, precision:{}, pr:{}, '
            'training_time:{}, infer_time:{}, '.format(
                ds_name, cv_index, best_parameters, acc, fpr, tpr, auc, precision, pr, training_time,
                inference_time_for_1000))

    record_results = record_results.append(
        {'Dataset Name': ds_name, 'Algorithm Name': algo_name, 'Cross Validation': cv_index,
         'Hyper-Parameters Values': best_parameters,
         'Accuracy': round(acc, 4), 'TPR': round(np.mean(tpr), 4), 'FPR': round(np.mean(fpr), 4),
         'Precision': round(precision, 4), 'AUC': round(auc, 4),
         'PR-Curve': round(pr, 4), 'Training Time': training_time, 'Inference Time': inference_time_for_1000},
        ignore_index=True)

    return record_results


def create_and_run_estimator(X_train, y_train, X_test, y_test, index, algo_name, results, ds_name):
    """
    Creates FastForest or Random Forest, tunes its parameters using RandomizedSearchCV, runs it and calculated its
    performance metrics
    :param X_train
    :param y_train
    :param X_test
    :param y_test
    :param index: the CV index of this run [1, ... , SPLITS_FOLDS)
    :param algo_name: FastForest or RandomForest,
    :param results: the results dataframe of all the datasets
    :param ds_name: the dataset name
    :return:
    """
    if VERBOSE:
        print('index: {}, algo_name: {}'.format(index, algo_name))

    # 3-Fold CV on the train set to tune the hyper-parameters with Random Search
    rs = None
    if algo_name == 'RandomForest':
        rs = RandomizedSearchCV(estimator=MyRandomForestClassifier(), param_distributions=random_grid, n_iter=50,
                                cv=TUNE_FOLDS, random_state=SEED)
    else:
        rs = RandomizedSearchCV(estimator=FastForestClassifier(), param_distributions=random_grid_ff, n_iter=50,
                                cv=TUNE_FOLDS, random_state=SEED)
    rs.fit(X_train, y_train)
    best_parameters = rs.best_params_

    train_start_time = time.time()
    if algo_name == 'FastForest':
        clf = FastForestClassifier(n_estimators=best_parameters['n_estimators'],
                                   max_depth=best_parameters['max_depth'],
                                   min_samples_leaf=best_parameters['min_samples_leaf'])
    else:
        clf = MyRandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                       max_depth=best_parameters['max_depth'],
                                       min_samples_leaf=best_parameters['min_samples_leaf'],
                                       max_features=best_parameters['max_features'],
                                       bootstrap=best_parameters['bootstrap'])
    clf.fit(X_train, y_train)
    training_time_random = round((time.time() - train_start_time), 2)

    # Create the estimator with the best parameters and predict the class for the test set on the best hyper-parameters
    inference_start_time = time.time()
    y_pred = rs.best_estimator_.predict_proba(X_test)
    inference_time_for_1000_random = round(((time.time() - inference_start_time) / len(X_test) * 1000), 2)

    # Calculate performance measurements
    results = create_performance_summary_record_in_results(y_test, y_pred, algo_name, index,
                                                           best_parameters, training_time_random,
                                                           inference_time_for_1000_random, results, ds_name)
    return results


def split_to_X_and_y(df, class_col):
    """
    Splits a dataframe into 2 arrays: the features and the classification label (class)
    :param df: the dataframe
    :param class_col: the index of the class column
    :return: X, y arrays
    """
    X = np.array(df.drop(df.columns[class_col], axis=1))
    y = np.array(df[df.columns[class_col]])
    return X, y


def encode_categorical_features(df, to_exclude=[]):
    """
    Finds the categorical columns in a dataframe and code the with one-hot encoding
    :param df: the dataframe
    :param to_exclude: columns to exclude from the encoding (for example, the class column)
    :return: the updated dataframe
    """
    df_cat_columns = df.select_dtypes(include=['category', object]).columns.tolist()
    columns = [x for x in df_cat_columns if x not in to_exclude]
    df = pd.get_dummies(df, columns=columns)
    return df


def fill_missing_values(df):
    """
    Fills missing values in the dataframe - for categorical columns with MISSING and for numerical columns with 0
    :param df: the dataframe
    :return: the updated dataframe
    """
    df_cat_columns = df.select_dtypes(include=['category', object]).columns.tolist()
    df_num_columns = [x for x in df.columns.tolist() if x not in df_cat_columns]
    df[df_cat_columns] = df[df_cat_columns].fillna(MISSING)
    df[df_num_columns] = df[df_num_columns].fillna(0)
    return df


def check_extreme_minority_class(df, class_col, unique, counts):
    """
    Checkes whether a dataframe has an extreme minor classes, that is, class values with less then 5 occurrences. In that
    case the function removes them and returns the updated dataframe
    :param df: the dataframe
    :param class_col: the index of the class column
    :param unique: the unique class values
    :param counts: the count of each unique class value
    :return: the updated dataframe
    """
    for i in range(len(counts)):
        if counts[i] < 5:
            df = df[df[df.columns[class_col]] != unique[i]]
    return df

def calc_precision_recall(X_train, X_test, y_train, y_test, algo, best_params):
    """
        Calculates the Precision-Recall average for multi-class problems
        :param y_true: the true classification of the set
        :param y_prediction: y_prediction: the predicted classification of the set
        :return: Precision-Recall curve average
        """
    n_classes = len(np.unique(y_train))
    y_train_bin = label_binarize(y_train, classes=[*range(n_classes)])
    clf = None
    if algo == 'FastForest':
        clf = OneVsRestClassifier(FastForestClassifier(n_estimators=best_params['n_estimators'],
                                                       max_depth=best_params['max_depth'],
                                                       min_samples_leaf=best_params['min_samples_leaf']))
    else:
        clf = OneVsRestClassifier(MyRandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                           max_depth=best_params['max_depth'],
                                                           min_samples_leaf=best_params['min_samples_leaf'],
                                                           max_features=best_params['max_features'],
                                                           bootstrap=best_params['bootstrap']))
    clf.fit(X_train, y_train_bin)
    y_score = clf.predict_proba(X_test)

    average_precision = []
    y_test_bin = label_binarize(y_test, classes=[*range(n_classes)])
    for i in range(n_classes):
        average_precision.append(average_precision_score(y_test_bin[:, i], y_score[:, i]))
    return round(np.mean(average_precision), 4)

    #######################################  MAIN SECTION #######################################


if __name__ == "__main__":
    total = 0

    # Initialize the results dataframe with all the mterics columns
    results = pd.DataFrame(
        columns=['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper-Parameters Values', 'Accuracy', 'TPR',
                 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time'])

    # Find all the datasets file name in the data folder and store then in the datasets array
    datasets = []
    datasets_folder = os.getcwd() + '/classification_datasets/'
    for d in glob.glob(datasets_folder + '*.csv'):
        datasets.append(d)

    # Initialize the grid for the random search
    random_grid = {'bootstrap': [True, False],          # relevant only for Random Forest: whether to use bootstrapping or not
                   'max_depth': [3, 6, 12],             # the Decision Tree max depth
                   'max_features': ['auto', 'sqrt'],    # relevant only for Random Forest: the subspacing method
                   'min_samples_leaf': [1, 3],       # the minimal number of samples in the Decision Tree's leaves
                   'n_estimators': [5, 15, 40]}         # the number of trees the forest should grow

    random_grid_ff = {'max_depth': [3, 6, 12],
                      'min_samples_leaf': [1, 3],
                      'n_estimators': [5, 15, 40]}

    ###########################  FastForest AND RandonForest ALGORITHMS PERFORMANCE ################################

    # Go over all the datasets and -
    # 1. Read the csv - last column is the class feature (unless otherwise stated)
    # 2. Perform minimal preprocessing: encode the categorical features and class column, fill in missing values and so on
    # 3. Perform 10-Fold CV for splitting for training and testing (for FastForest (FF) and RandomForest (RF))
    # 4. Perform 3-Fold CV on the train set to tune the hyper-parameters (for FastForest (FF) and RandomForest (RF))
    # 5. Create the estimator with the best parameters and predict the class for the test set on the best
    #    hyper-parameters (for FF anf RF)
    # 6. Calculate performance measurements (for FF anf RF)

    for i in range(len(datasets)):
    # if False:
        # 1. read the csv
        file_name = datasets[i]
        # TODO: this works in windows. Comment this and uncomment following line if working in Linux
        ds_name = file_name[(file_name.rindex('\\') + 1):file_name.rindex('.')]
        # ds_name = file_name[(file_name.rindex('/') + 1):file_name.rindex('.')]  # TODO: for Linux!

        total = total + 1
        print('Working on dataset {} out of {}, ds name: {}'.format(total, len(datasets), ds_name))
        df = pd.read_csv(file_name)

        # 2. Minimal preprocessing

        # check for extreme minority class - if their count is less than 5. Remove them
        class_col = df.shape[1] - 1
        if ds_name == 'solar-flare':
            class_col = 0
        unique, counts = np.unique(df[df.columns[class_col]], return_counts=True)
        if ds_name != 'lenses':
            df = check_extreme_minority_class(df, class_col, unique, counts)

        # fill in missing values
        df = fill_missing_values(df)

        # One hot encoding for the categorical features
        class_name = df.columns[class_col]
        df = encode_categorical_features(df, [df.columns[class_col]])

        X, y = split_to_X_and_y(df, df.columns.get_loc(class_name))

        is_multi_class = False
        if len(np.unique(y)) > 2:
            is_multi_class = True

        # Encode the class column (it might be a string, or not continuous in case we removed minority classes)
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

        # Check that the number of samples of each class value is greater than the folds size. if so can use stratification
        to_stratify = True
        unique, counts = np.unique(df[df.columns[class_col]], return_counts=True)
        counts = np.sort(counts)
        if (counts[0] < SPLIT_FOLDS) and (ds_name != 'lung-cancer'):
            to_stratify = False

        # 3. 10-Fold CV for splitting for training and testing
        skf = None
        SPLIT_FOLDS = 10
        # I used only 4 folds for the lenses dataset since it is very small and couldn't be split with stratification for 10 folds
        if ds_name == 'lenses':
            SPLIT_FOLDS = 4

        if to_stratify:
            skf = StratifiedKFold(n_splits=SPLIT_FOLDS, shuffle=True, random_state=SEED)
        else:
            skf = KFold(n_splits=SPLIT_FOLDS, shuffle=True, random_state=SEED)

        j = 0
        for train, test in skf.split(X, y):
            if j<=3:
                j = j + 1
                continue
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            # make sure y_test has all classes represented, otherwise scikit-learn issues errors during the metrics calculations
            train_unique = np.unique(y_train)
            test_unique = np.unique(y_test)
            redundant = []
            indices = []
            if collections.Counter(train_unique) != collections.Counter(test_unique):
                redundant = np.setdiff1d(y_train, y_test)
                print(redundant)
                # remove all the lines with the redundant classes
                for r in redundant:
                    indices = np.where(y_train == r)
                    if VERBOSE:
                        print(indices)
                        print(len(X_train), len(y_train))
                    X_train = np.delete(X_train, indices, axis=0)
                    y_train = np.delete(y_train, indices, axis=0)
                    if VERBOSE:
                        print(len(X_train), len(y_train))

            # 4. 5. 6. Search for best parameters, fit, predict and calculate performance for FF
            results = create_and_run_estimator(X_train, y_train, X_test, y_test, j + 1, 'FastForest', results, ds_name)

            # 4. 5. 6. Search for best parameters, fit, predict and calculate performance for RF
            results = create_and_run_estimator(X_train, y_train, X_test, y_test, j + 1, 'RandomForest', results,
                                               ds_name)
            results.to_csv("results.csv", index=False)

            j = j + 1

    results = pd.read_csv("results.csv")

    # Add averages (and standard deviation) per feature as the last 2 rows
    avgs = []
    algos = ['FastForest', 'RandomForest']
    for a in algos:
        row = []
        for c in results.columns:
            if c == 'Algorithm Name':
                row.append(a)
            elif c == 'Dataset Name':
                row.append('Average (Std)')
            elif c in ['Cross Validation', 'Hyper-Parameters Values']:
                row.append('')
            else:
                avg = round(np.mean(results[results['Algorithm Name'] == a][c]), 4)
                std = round(np.std(results[results['Algorithm Name'] == a][c]), 4)
                row_str = str(avg) + '(' + str(std) + ')'
                row.append(row_str)
        avgs.append(row)

    results_with_avg = results.copy();
    for avg in avgs:
        results_with_avg = results_with_avg.append(
            {'Dataset Name': avg[0], 'Algorithm Name': avg[1], 'Cross Validation': avg[2], 'Hyper-Parameters Values': avg[3],
             'Accuracy': avg[4], 'TPR': avg[5], 'FPR': avg[6], 'Precision': avg[7], 'AUC': avg[8], 'PR-Curve': avg[9],
             'Training Time': avg[10], 'Inference Time': avg[11]},
            ignore_index=True)
        results_with_avg.to_csv("results_with_avg.csv", index=False)


    #######################################  RESULTS SIGNIFICANCE  #######################################

    # Check if results for are statistically significant - first check the Training Time. The expectation is that FF is
    # faster than RF. And then make sure the Accuracy is statistically the same
    ff_acc = results[results['Algorithm Name'] == 'FastForest'].Accuracy
    rf_acc = results[results['Algorithm Name'] == 'RandomForest'].Accuracy
    stat_acc, p_acc = mannwhitneyu(ff_acc, rf_acc)
    print("FF avg accuracy: {}, RF avg accuracy: {}".format(np.mean(ff_acc), np.mean(rf_acc)))

    ff_auc = results[results['Algorithm Name'] == 'FastForest'].AUC
    rf_auc = results[results['Algorithm Name'] == 'RandomForest'].AUC
    stat_auc, p_auc = mannwhitneyu(ff_auc, rf_auc)
    print("FF avg AUC: {}, RF avg AUC: {}".format(np.mean(ff_auc), np.mean(rf_auc)))

    ff_time = results[results['Algorithm Name'] == 'FastForest']['Training Time']
    rf_time = results[results['Algorithm Name'] == 'RandomForest']['Training Time']
    stat_time, p_time = mannwhitneyu(ff_time, rf_time)
    print("FF avg training time: {}, RF avg training time: {}".format(np.mean(ff_time), np.mean(rf_time)))

    print('Mann–Whitney U test value for training time: {} ({}), '
          'for accuracy: {} ({}), for AUC:{} ({})'.format(round(stat_time, 2), round(p_time, 4), round(stat_acc, 2),
                                                          round(p_acc, 4), round(stat_auc, 2), round(p_auc, 4)))

    #######################################  META LEARNING MODEL #######################################

    # Build the Meta-features dataset. The learnt parameter is the AUC, and the purpose is show that FF is as good as RF  -
    # 1. duplicate each line from ClassificationAllMetaFeatures csv file
    # 2. Add the algorithm name column and populate it
    # 3. Add a class column
    # 4. Calculate for each dataset which algorithm has the best AUC and set '1' for it in the relevant row, and 0 for
    #    the row of the other algorithm for that dataset

    meta_dataset = pd.read_csv("ClassificationAllMetaFeatures.csv")
    ds_names = np.unique(results['Dataset Name'])

    # Add the 'Best AUC' column
    meta_dataset['Best AUC'] = -1

    meta_dataset_dup = meta_dataset.copy()
    meta_dataset_dup['Algorithm Name'] = 'FastForest'

    meta_dataset['Algorithm Name'] = 'RandomForest'

    # Concatenate the 2 DFs
    meta_dataset = pd.concat([meta_dataset, meta_dataset_dup], ignore_index=True)

    # Populate the 'BEST AUC' column with 1 or 0 according to whether the Best AUC score was obtained with that algorithm
    for d in ds_names:
        ff_ds = results[(results['Algorithm Name'] == 'FastForest') & (results['Dataset Name'] == d)]
        rf_ds = results[(results['Algorithm Name'] == 'RandomForest') & (results['Dataset Name'] == d)]
        ff_avg_training_time = ff_ds['AUC'].mean()
        rf_avg_training_time = rf_ds['AUC'].mean()
        is_ff_faster = int(ff_avg_training_time > rf_avg_training_time)

        meta_dataset.loc[((meta_dataset['Algorithm Name'] == 'FastForest') & (
                    meta_dataset['dataset'] == d)), 'Best AUC'] = is_ff_faster
        meta_dataset.loc[((meta_dataset['Algorithm Name'] == 'RandomForest') & (
                    meta_dataset['dataset'] == d)), 'Best AUC'] = 1 - is_ff_faster

    # Create the XGBoost model, fit and predict, with Leave-one-dataset-out
    meta_dataset = encode_categorical_features(meta_dataset, ['dataset'])
    meta_results = pd.DataFrame(
        columns=['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper-Parameters Values', 'Accuracy', 'TPR',
                 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time'])
    i = 1
    for d in ds_names:
        train_df = meta_dataset[meta_dataset['dataset'] != d]
        test_df = meta_dataset[meta_dataset['dataset'] == d]

        # the 'dataset' column is redundant at this stage
        ds_col = meta_dataset.columns.get_loc('dataset')
        train_df = train_df.drop(train_df.columns[ds_col], axis=1)
        test_df = test_df.drop(test_df.columns[ds_col], axis=1)

        class_col = meta_dataset.columns.get_loc('Best AUC')
        X_train, y_train = split_to_X_and_y(train_df, class_col)
        X_test, y_test = split_to_X_and_y(test_df, class_col)

        # TODO: temp!!!!
        if len(X_test) == 0:
            continue

        xgb = XGBClassifier()

        train_start_time = time.time()
        xgb.fit(X_train, y_train)
        training_time = round((time.time() - train_start_time), 2)

        inference_start_time = time.time()
        y_pred = xgb.predict_proba(X_test)
        inference_time_for_1000 = round(((time.time() - inference_start_time) / len(X_test) * 1000), 2)

        meta_results = create_performance_summary_record_in_results(y_test, y_pred, '', i, '', training_time,
                                                                    inference_time_for_1000, meta_results, d)
        meta_results.to_csv("meta_results.csv", index=False)
        i = i + 1


    #######################################  FEATURES IMPORTANCE AND SHAP #######################################


    # First fit the model on the DF - drop the 'dataset' column and nan values
    ds_col = meta_dataset.columns.get_loc('dataset')
    meta_dataset = meta_dataset.drop(meta_dataset.columns[ds_col], axis=1)
    meta_dataset.fillna(0, inplace=True)

    class_col = meta_dataset.columns.get_loc('Best AUC')
    X, y = split_to_X_and_y(meta_dataset, class_col)

    xgb = XGBClassifier(booster='gbtree')
    xgb.fit(X, y)

    weight_res = xgb.get_booster().get_score(importance_type='weight')
    gain_res = xgb.get_booster().get_score(importance_type='gain')
    cover_res = xgb.get_booster().get_score(importance_type='cover')

    tryy = xgb.feature_importances_

    print(weight_res)
    print(gain_res)
    print(cover_res)

    # SHAP
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values, X)
    # shap.dependence_plot("f1", shap_values, X)
    shap.summary_plot(shap_values, X)


    #################################  FOR THE REPORT ILLUSTRATION SECTION ###################################

    ds = pd.read_csv('classification_datasets/illustration.csv')
    X, y = split_to_X_and_y(ds, ds.shape[1] - 1)
    ff = FastForestClassifier(n_estimators=1, max_depth=3, min_samples_leaf=2)
    ff.fit(X, y)
