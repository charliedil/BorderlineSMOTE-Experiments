"""
Authors Charlie Dil, Nafeez Fahad, Roshan James Kurisummootil

We're comparing Borderline Smote1 and 2 with regular SMOTE and no oversampling on the bean dataset from UC Irvine's repository.

Please install requirements before running. You can use
pip install -r requirements.txt

Run with
python main.py <path/to/arff/file>

Python version: python3.10
But 3.11 should also work.
"""

from bean_dataset import Dataset
from classifier import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from utils import get_dataset_from_path, compare_distr, print_scores
import pandas as pd

import sys

"""Preprocessing the dataset. Create dataset object and oversample!"""


def preprocess(dataset_path):
    bean_dataset = Dataset(dataset_path=dataset_path)
    print("Before sampling:")
    print(bean_dataset.get_distr())
    features, classes = bean_dataset.oversample_smote()
    bean_oversampled_dataset = Dataset(classes=classes, features=features)
    print("After sampling:")
    print(bean_oversampled_dataset.get_distr())


"""Compare the performance of the logistic regression classifier with various SMOTE variants"""


def logistic_regression(cl, param_grid_regular, param_grid_borderline):

    print("Cross-validation scores for Logistic Regression:")
    print_scores(
        cl.cross_val_score(
            estimator=LogisticRegression(solver="liblinear", max_iter=1000)
        )
    )

    for i in range(0, len(param_grid_regular)):
        print("K: ", param_grid_regular[i]["k"])
        print_scores(
            cl.cross_val_score(
                estimator=LogisticRegression(solver="liblinear", max_iter=1000),
                smote_strategy="smote",
                k=param_grid_regular[i]["k"],
            ),
            "Regular Smote",
        )

    for i in range(0, len(param_grid_borderline)):
        print(
            "Kind: ",
            param_grid_borderline[i]["kind"],
            "K: ",
            param_grid_borderline[i]["k"],
            "M: ",
            param_grid_borderline[i]["m"],
        )
        print_scores(
            cl.cross_val_score(
                estimator=LogisticRegression(solver="liblinear", max_iter=1000),
                smote_strategy=param_grid_borderline[i]["kind"],
                k=param_grid_borderline[i]["k"],
                m=param_grid_borderline[i]["m"],
            ),
            param_grid_borderline[i]["kind"],
        )


"""Compare the performance of the decision tree classifier with various SMOTE variants"""


def decision_tree(cl, param_grid_regular, param_grid_borderline):
    print("Cross-validation scores for Decision Tree:")
    print_scores(cl.cross_val_score(estimator=DecisionTreeClassifier()))

    for i in range(0, len(param_grid_regular)):
        print("K: ", param_grid_regular[i]["k"])
        print_scores(
            cl.cross_val_score(
                estimator=DecisionTreeClassifier(),
                smote_strategy="smote",
                k=param_grid_regular[i]["k"],
            ),
            "Regular Smote",
        )

    for i in range(0, len(param_grid_borderline)):
        print(
            "Kind: ",
            param_grid_borderline[i]["kind"],
            "K: ",
            param_grid_borderline[i]["k"],
            "M: ",
            param_grid_borderline[i]["m"],
        )
        print_scores(
            cl.cross_val_score(
                estimator=DecisionTreeClassifier(),
                smote_strategy=param_grid_borderline[i]["kind"],
                k=param_grid_borderline[i]["k"],
                m=param_grid_borderline[i]["m"],
            ),
            param_grid_borderline[i]["kind"],
        )


"""Just a main method, calls other methods, handles command line input"""


def main():
    if len(sys.argv) != 2:
        print(
            "Please provide one command line argument - path to the .arff file for the dry bean dataset"
        )
        exit(1)
    dataset_path = sys.argv[1]
    dataset = get_dataset_from_path(dataset_path=dataset_path)

    compare_distr(dataset=dataset)

    cl = Classifier(features=dataset.features, classes=dataset.classes, n_splits=5)

    param_grid_regular = {"k": [2, 5, 10, 15, 20]}
    param_grid_regular = ParameterGrid(param_grid_regular)

    param_grid_borderline = {
        "kind": ["borderline-1", "borderline-2"],
        "k": [2, 5, 10, 15, 20],
        "m": [2, 5, 10, 15, 20],
    }
    param_grid_borderline = ParameterGrid(param_grid_borderline)

    logistic_regression(cl, param_grid_regular, param_grid_borderline)
    decision_tree(cl, param_grid_regular, param_grid_borderline)


"""weird python stuff"""
if __name__ == "__main__":
    main()
