import arff
from collections import defaultdict
from imblearn.over_sampling  import BorderlineSMOTE, SMOTE
import numpy as np

"""BEAN DATA SET CLASS"""
class BeanDataset:

    """Constructor, two ways to init, one with a path to the arff file or with two array like ojects for the features and classes"""
    def __init__(self, dataset_path=None, features=None, classes=None):
        if dataset_path is not None:
            with open(dataset_path, "r") as f:
                data = np.array(arff.load(f)["data"])
                self.features = data[:, :-1].astype(float)
                self.classes = data[:, -1]
        elif features is not None and classes is not None:
            self.features = features
            self.classes = classes
        else:
            print("Please pass in either a file path or two arraylike lists for features and classes respectively")
            exit(1)

    def split(self, k=5):
        """Fill this in with the code to split into k buckets for CV, might not need this, or maybe it shouldnt go in this class. but its here for now"""


    """borderline smote code"""
    def oversample_bsmote(self, kind, sampling_strategy='not majority', random_state=42, k_neighbors=5, m_neighbors=10): #may want to play with the parameters here
        features, classes = BorderlineSMOTE(kind=kind, sampling_strategy=sampling_strategy, 
                        random_state=random_state,k_neighbors=k_neighbors,m_neighbors=m_neighbors).fit_resample(self.features, self.classes)
        return features, classes
    
    """regular smote code"""
    def oversample_smote(self, sampling_strategy='not majority', random_state=42, k_neighbors=5): #also here, play with them
        features, classes = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors).fit_resample(self.features, self.classes)
        return features, classes
    
    """distributions of labels, useful for presentation"""
    def get_distr(self):
        labels = defaultdict(lambda: 0)
        for label in self.classes:
            labels[label]+=1
        return labels


