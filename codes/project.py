#!/usr/bin/env python
# coding: utf-8

# In[1]:


from my_svm import My_SVM
from util import Util
from pca import MY_PCA
from data_cleaning import DataCleaning
from visualization import Visualization
import seaborn as sns
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from nn.dataset import PatientDataset
from nn.net import Net
import torch
import matplotlib.pyplot as plt

# custom


def main_deprecated():
    # This is deprecated, never use this please.
    print("This is main, alhumdulliah")
    ##### This block is for data cleaning #####
    missing_values = ["n/a", "na", "--", "?"]
    raw_data = pd.read_csv('../dataset_diabetes/diabetic_data.csv',
                           delimiter=',', na_values=missing_values)
    # print(raw_data.head()) # print head of the data
    # print(raw_data.describe()) # shows numerical columns statistics e.g. count, mean, std, min, max etc
    # print(raw_data.shape) # prints shape of the dataset (101766, 50)
    # print(raw_data["weight"].isnull().sum()) #prints number of null values in weight column
    # print(raw_data["weight"].shape[0]) #prints number of columns in weight column
    data_cleaning = DataCleaning()
    raw_data = data_cleaning.clean_columns(raw_data, missing_bound=.2)
    cols_having_missing_values = data_cleaning.get_cols_having_missing_values(
        raw_data, False)  # cols having missing values
    # raw_data.dtypes #shows the column data types
    raw_data = data_cleaning.fill_missing_values(
        raw_data, cols_having_missing_values)
    # print(get_cols_having_missing_values(raw_data, False)) #no columns with missing values
    raw_data = data_cleaning.just_remove_columns(raw_data, columns=[
                                                 "encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id", "admission_source_id", "num_procedures"])
    df = raw_data
    my_util = Util()
    my_util.save_df(df, "../only_calculated_datasets/cleaned_df.pkl")
    print("Filled the missing values either by the mode or mean value")


def clean():
    missing_values = ["n/a", "na", "--", "?"]
    raw_data = pd.read_csv('../dataset_diabetes/diabetic_data.csv',
                           delimiter=',', na_values=missing_values)
    data_cleaning = DataCleaning()
    raw_data = data_cleaning.clean_columns(raw_data, missing_bound=.2)
    cols_having_missing_values = data_cleaning.get_cols_having_missing_values(
        raw_data, False)  # cols having missing values
    raw_data = data_cleaning.fill_missing_values(
        raw_data, cols_having_missing_values)
    raw_data = data_cleaning.just_remove_columns(raw_data, columns=[
                                                 "encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id", "admission_source_id", "num_procedures"])
    df = raw_data
    my_util = Util()
    my_util.save_df(df, "../only_calculated_datasets/cleaned_df.pkl")


def visualize():
    my_util = Util()
    df = my_util.load_df("../only_calculated_datasets/cleaned_df.pkl")
    my_visu = Visualization()
    # to draw boxplot x or y must be numeric
    # my_visu.make_boxplot(df, x='num_lab_procedures', y='num_medications',
    #                      image_path="../graphs/data_visualization/num_lab_procedures_vs_num_medications.png")
    # my_visu.make_boxplot(df, x='readmitted', y='num_lab_procedures',
    #                      image_path="../graphs/data_visualization/num_lab_procedures_vs_readmitted_box_plot.png")
    # my_visu.make_boxplot(df, x='readmitted', y='num_medications',
    #                      image_path="../graphs/data_visualization/num_medications_vs_readmitted_box_plot.png")
    # my_visu.make_boxplot(df, x='readmitted', y='number_diagnoses',
    #                      image_path="../graphs/data_visualization/number_diagnoses_vs_readmitted_box_plot.png")
    # my_visu.make_histograms(df, ["age", "race", "gender", "max_glu_serum", "A1Cresult", "num_lab_procedures",
    #                              "time_in_hospital", "change", "diabetesMed"], image_path="../outputs/attr_hist_plot.png")  # all the attributes distributions
    # my_visu.make_histograms(df, ["readmitted"], cmap="spring",
    #                         image_path="../outputs/class_attr_hist_plot.png")
    # my_visu.build_scatter_2attr_plot(df, x_col="num_lab_procedures", y_col="num_medications",
    #                                  image_path="../outputs/num_of_procedures_vs_medications_plot.png")
    # my_visu.build_3d_scatter_plot(df, x_axis_col="number_outpatient", y_axis_col="number_emergency", z_axis_col="number_inpatient",
    #                               class_col="readmitted", image_path="../outputs/outPt_emergencyPt_inPt_vs_readmitted_sctter_plot.png", is_show=False)
    # my_visu.build_3d_scatter_plot(df, x_axis_col="diag_1", y_axis_col="diag_2", z_axis_col="diag_3",
    #                               class_col="readmitted", image_path="../outputs/diag_1_2_3_vs_readmitted_sctter_plot.png", is_show=False)


def do_pca():
    my_util = Util()
    print("Loading data ... ...")
    clean_df = my_util.load_df("../only_calculated_datasets/cleaned_df.pkl")
    my_pca = MY_PCA()
    keep_variances = [.80, .90, .95]
    for keep_variance in keep_variances:
        df = my_pca.do_pca_with_one_hot_encoding(
            clean_df, keep_variance=keep_variance)
        print(df.head())
        print("after running pca on one-hot shape: ", df.shape)
        print("saving df after running pca ... ...")
        my_util.save_df(
            df, "../only_calculated_datasets/one_hot_pca_all_cols_{}_variance_df.pkl".format(keep_variance))


def do_one_hot():
    my_util = Util()
    print("Loading data ... ...")
    clean_df = my_util.load_df("../only_calculated_datasets/cleaned_df.pkl")
    my_util.one_hot(clean_df)


def one_hot_and_standardize():
    my_util = Util()
    print("Loading data ... ...")
    clean_df = my_util.load_df("../only_calculated_datasets/cleaned_df.pkl")
    my_util.one_hot_and_standardize(clean_df)


def do_svm():
    my_util = Util()
    print("Loading data ... ...")
    df = my_util.load_df(
        "../only_calculated_datasets/one_hot_pca_only_categorical_cols_0.9_variance_df.pkl")
    main_df, readmitted_mapped_index = my_util.convert_into_numeric_class_attr_readmitted(
        df)
    print(main_df.shape)
    fracs = [.001, .01, .1]
    for frac in fracs:
        df = main_df.sample(frac=frac)
        print("Splitting train/validation/test set ... ...")
        x_train, y_train, x_test, y_test = my_util.do_train_test_split(
            df=df, class_name="readmitted")
        my_svm = My_SVM()
        # my_svm = My_SVM(kernel='poly', gamma='scale')
        my_svm.train_validate_test(x_train, y_train, x_test, y_test)
        x, y = my_util.separate_class_cols(df, classes=["readmitted"])
        my_svm.k_fold_cross_validation(x, y.values.ravel(), k=5)


def print_net_details(net):
    print(net)
    params = list(net.parameters())
    print(len(params))
    for i in range(len(params)):
        print(params[i].size())


def do_nn():
    my_util = Util()
    print("loading data ... ...")
    df = my_util.load_df(
        "../only_calculated_datasets/oneHot_standardized_for_nn.pkl")
    main_df, readmitted_mapped_index = my_util.convert_into_numeric_class_attr_readmitted(
        df)
    print(main_df)
    print(readmitted_mapped_index)
    x_train, y_train, x_test, y_test = my_util.do_train_test_split(
        df=main_df, class_name="readmitted", test_size=.4)
    print("creating pytorch dataset object ... ...")
    trainset = PatientDataset(x_df=x_train, y=y_train)
    x, y = trainset[0]
    print("initializing neural net ... ... ")
    net = Net(input_size=x_train.shape[1], output_size=2)
    print_net_details(net)
    net.cuda(device=net.get_available_device())
    print("training ... ... ")
    net.my_train(trainset, x_train.shape[0])
    # print("testing ... ...")
    # testset = PatientDataset(x_df=x_test, y=y_test)
    # net.my_test(testset)


def do_nn_test():
    my_util = Util()
    print("loading data ... ...")
    df = my_util.load_df(
        "../only_calculated_datasets/categorical_1hot_numerical_asB4_only_yes_no_class.pkl")
    main_df, readmitted_mapped_index = my_util.convert_into_numeric_class_attr_readmitted(
        df)
    print(main_df)
    print(readmitted_mapped_index)
    x_train, y_train, x_test, y_test = my_util.do_train_test_split(
        df=main_df, class_name="readmitted", test_size=.4)
    print("creating pytorch dataset object ... ...")
    net = Net(input_size=x_train.shape[1], output_size=2)
    net.cuda(device=net.get_available_device())
    net.load_state_dict(torch.load("../nn_models/nn_model.pth"))
    print("testing ... ...")
    testset = PatientDataset(x_df=x_test, y=y_test)
    net.my_test(testset)


def generate_accuracy_and_loss_graph():

    loss_train = [0.006889165564535658, 0.0068818967517972176, 0.006833871806975375, 0.006612495635584673, 0.006408794532955122, 0.006301125029263068, 0.006226343237359391, 0.006169632908191305, 0.006105207002827322, 0.006031618879863618, 0.005969584100450529, 0.0058977607494938335, 0.0058309783486933065, 0.005751859112078274, 0.0056700767698253446,
                  0.005587785699552256, 0.005505353785862992, 0.005420425864344007, 0.005349784749781758, 0.0052371007401927705, 0.005171942382684362, 0.00507779189018994, 0.005001218676533887, 0.0048999092526916055, 0.004841816498244218, 0.0047469109994280135, 0.004683451596865006, 0.00462831947537937, 0.0045151081595786865, 0.004446639051489545]
    acc_train = [53.88688524590164, 53.88688524590164, 55.747540983606555, 60.56393442622951, 63.268852459016394, 64.51803278688524, 65.3360655737705, 66.01147540983607, 66.74426229508197, 67.58524590163934, 67.98360655737704, 68.5344262295082, 69.26393442622951, 69.79836065573771, 70.31967213114754,
                 71.00655737704918, 71.6016393442623, 72.2672131147541, 72.72622950819672, 73.59016393442623, 73.84754098360656, 74.47049180327869, 75.20327868852459, 75.8672131147541, 76.07377049180327, 76.78688524590164, 76.98852459016393, 77.43442622950819, 77.97213114754098, 78.4311475409836]

    loss_train = [i * 6000 for i in loss_train]
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train, label="Train loss")
    plt.plot(range(len(acc_train)), acc_train, label="Train accuracy")
    plt.legend()
    plt.title('Train accuracy-loss graph')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.savefig("../graphs/nn_accuracy_loss/nn_out_5.png")


if __name__ == "__main__":
    # clean()
    visualize()
    # do_one_hot()
    # one_hot_and_standardize()
    # do_pca()
    # do_svm()
    # do_nn()
    # do_nn_test()
    # generate_accuracy_and_loss_graph()
