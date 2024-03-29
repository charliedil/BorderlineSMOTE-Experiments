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

from bean_dataset import BeanDataset
import pandas as pd

import sys

"""Preprocessing the dataset. Create dataset object and oversample!"""
def preprocess(dataset_path):
    bean_dataset = BeanDataset(dataset_path=dataset_path)
    print("Before sampling:")
    print(bean_dataset.get_distr())
    features, classes = bean_dataset.oversample_smote()
    bean_oversampled_dataset = BeanDataset(classes = classes, features=features)
    print("After sampling:")
    print(bean_oversampled_dataset.get_distr())

"""Just a main method, calls other methods, handles command line input"""
def main():
    if len(sys.argv) != 2:
        print("Please provide one command line argument - path to the .arff file for the dry bean dataset")
        exit(1)
    dataset_path = sys.argv[1]
    print(dataset_path)
    preprocess(dataset_path)
    
"""weird python stuff"""
if __name__ == '__main__':
    main()