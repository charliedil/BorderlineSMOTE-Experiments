from bean_dataset import Dataset
import numpy as np

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


""" Print the F1 cross validation scores and their mean """


def print_scores(scores, sampling_method=None):
    if sampling_method == None:
        sampling_method = "Before Sampling"

    print(sampling_method)
    print("Cross-validation scores:")
    print("F1:", scores[0])
    print("Mean F1:", np.mean(scores[0]))
    print("Recall:", scores[1])
    print("Mean Recall:", np.mean(scores[1]))
    print("Precision:", scores[2])
    print("Mean Precision:", np.mean(scores[2]))
