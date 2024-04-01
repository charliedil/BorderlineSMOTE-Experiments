from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from bean_dataset import Dataset
from sklearn.metrics import f1_score

""" CLASSIFIER CLASS """


class Classifier:
    def __init__(self, features, classes, n_splits=5):
        self.features = features
        self.classes = classes
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    """ Get the F1 score of the estimator after cross validation """

    def cross_val_score(self, estimator, smote_strategy="borderline-1"):
        scores = []
        X = self.features
        y = self.classes
        cv = self.cv
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            dataset = Dataset(features=X_train, classes=y_train)

            if smote_strategy == "borderline-1" or smote_strategy == "borderline-2":
                X_train, y_train = dataset.oversample_bsmote(kind=smote_strategy)
            elif smote_strategy == "smote":
                X_train, y_train = dataset.oversample_smote()

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)

            score = f1_score(y_val, y_pred, average="macro")
            scores.append(score)
        return scores
