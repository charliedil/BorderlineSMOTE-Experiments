from sklearn.model_selection import StratifiedKFold
from bean_dataset import Dataset
from sklearn.metrics import f1_score, recall_score, precision_score, matthews_corrcoef
import time

""" CLASSIFIER CLASS """


class Classifier:
    def __init__(self, features, classes, n_splits=5):
        self.features = features
        self.classes = classes
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=int(time.time())
        )

    """ Get the F1 score of the estimator after cross validation """

    def cross_val_score(self, estimator, smote_strategy="borderline-1", k=5, m=10):
        f1_scores = []
        recall_scores = []
        precision_scores = []
        mcc_scores = []
        X = self.features
        y = self.classes
        cv = self.cv
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            dataset = Dataset(features=X_train, classes=y_train)

            if smote_strategy == "borderline-1" or smote_strategy == "borderline-2":
                X_train, y_train = dataset.oversample_bsmote(
                    kind=smote_strategy, k_neighbors=k, m_neighbors=m
                )
            elif smote_strategy == "smote":
                X_train, y_train = dataset.oversample_smote(k_neighbors=k)

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)

            f1_scores.append(f1_score(y_val, y_pred, average="macro"))
            recall_scores.append(recall_score(y_val, y_pred, average="macro"))
            precision_scores.append(precision_score(y_val, y_pred, average="macro"))
            mcc_scores.append(matthews_corrcoef(y_val, y_pred))

        return f1_scores, recall_scores, precision_scores, mcc_scores
