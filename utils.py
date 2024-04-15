from bean_dataset import Dataset
import numpy as np
from prettytable import PrettyTable

""" Create a dataset object from given path """


def get_dataset_from_path(dataset_path):
    bean_dataset = Dataset(dataset_path=dataset_path)
    return bean_dataset


""" Compare the distribution of classes before and after sampling """


def compare_distr(dataset):
    print("Before sampling:")
    print(dataset.get_distr())
    features, classes = dataset.oversample_smote()
    bean_oversampled_dataset = Dataset(classes=classes, features=features)
    print("After sampling:")
    print(bean_oversampled_dataset.get_distr())


""" Return the mean, median and standard deviation of the cross validation scores """


def print_scores(scores, sampling_method=None):
    if sampling_method == None:
        sampling_method = "Before Sampling"

    print(sampling_method)
    table = PrettyTable()
    table.field_names = ["Measure", "Mean", "Median", "Standard Deviation"]
    measures = ["F1", "Recall", "Precision", "MCC"]
    for i in range(len(scores)):
        table.add_row(
            [measures[i], np.mean(scores[i]), np.median(scores[i]), np.std(scores[i])]
        )

    print(table)
