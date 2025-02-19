import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import itertools
from joblib import dump, load

def train_test_eq_split(X, y, n_per_class, random_state=None):
    if random_state:
        np.random.seed(random_state)
    sampled = X.groupby(y, sort=False).apply(
        lambda frame: frame.sample(n_per_class))
    mask = sampled.index.get_level_values(1)

    X_train = X.drop(mask)
    X_test = X.loc[mask]
    y_train = y.drop(mask)
    y_test = y.loc[mask]

    return X_train, X_test, y_train, y_test

print("Reading data.")
transactional_data = pd.read_csv("transactional_data.csv")
print("Doing sanity checks.")
if any(transactional_data.duplicated()):
    print("Duplicate rows detected.")
if any(transactional_data["amt"] <= 0):
    print("Impossible amount detected.")
if transactional_data.isnull().any().any():
    print("Missing values detected.")

# Numerical columns to use immediately as is
numerical = ["amt"]

relevant = transactional_data[numerical]
# Convert jobs to simplified form by taking everything before the first comma.
print("Simplifing job titles.")
transactional_data = transactional_data.assign(job=transactional_data["job"].str.split(",", expand=True)[0])
# Jobs are unused because they did not show any usable trends or easy (automated) way to group.

print("Parsing times.")
# This is horrible hacky code but it works.
relevant["second of day"] = pd.to_timedelta(pd.to_datetime(transactional_data["unix_time"], unit="s").dt.time.astype(str)).dt.total_seconds()
relevant["month"] = pd.to_datetime(transactional_data["unix_time"], unit="s").dt.month
# Age in days because years won't compute (easily) in pandas XD
now = pd.Timestamp('now')
relevant["age"] = pd.to_timedelta(now - pd.to_datetime(transactional_data["dob"], format="%d/%m/%Y")).dt.days

# Columns that need encoding from string to class.
print("Classifying categorical columns.")
# Including jobs here has a hefty memory and compute load.
enc_columns = ["gender", "category"]
features = transactional_data[enc_columns]
enc = OneHotEncoder(min_frequency=5, sparse_output=False)
arr = enc.fit_transform(features)
c = list(itertools.chain(*enc.categories_))
relevant = pd.concat([relevant, pd.DataFrame(arr, columns=c)], axis=1)

# 3 feature testing
#relevant = relevant[["second of day", "amt", "month"]]

# rename amt to amount
relevant = relevant.rename(columns={"amt": "amount"})

target_column = "is_fraud"

X = relevant
Y = transactional_data[target_column]

def train_test(model):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    # This version is for specialised test scoring by changing the distribution of classes.
    #X_train, X_test, Y_train, Y_test = train_test_eq_split(X, Y, 5000)
    print(f"\nTraining {model}.")
    clf = model.fit(np.array(X_train), np.array(Y_train))

    Y_pred = clf.predict(np.array(X_test))
    confusion = confusion_matrix(np.array(Y_test), Y_pred)

    FP = confusion.sum(axis=0) - np.diag(confusion)  
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print(f"Accuracy: {ACC[0]*100:.2f}%")
    return {
        "confusion": confusion,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "TN": TN,
        "Accuracy": ACC,
        "model": clf
    }

def main():
    # Ordered ascending by 'accuracy' mostly.
    methods = [
        # Decision tree particularly fluctuates a LOT when changing parameters slightly.
        tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=9, class_weight="balanced"),
        LinearSVC(class_weight="balanced", dual="auto"),
        # Due to the nature of SGD, these two fluctuate a lot.
        SGDClassifier(n_jobs=-1, class_weight="balanced"),
        Pipeline([('scaler', StandardScaler()), ("sgd", SGDClassifier(n_jobs=-1, class_weight="balanced"))]),
        # KNeighbors can take a long time to train.
        KNeighborsClassifier(n_jobs=-1, algorithm = 'kd_tree'),
        HistGradientBoostingClassifier(max_iter=200, class_weight="balanced"),
        RandomForestClassifier(n_jobs=-1, class_weight="balanced")
    ]
    print("Starting Trainings")
    results = list(map(train_test, methods))
    return results

def display(results):
    for res in results:
        ConfusionMatrixDisplay(res["confusion"]).plot()
        plt.title(str(res["model"]))
        plt.show()

def plot_trees(rf, n=range(0, 5)):
    fn = list(relevant.columns)
    cn = ["Legit", "Fraud"]
    num = len(n)
    fig, axes = plt.subplots(nrows = 1, ncols = num,figsize = (10,2), dpi=1000)
    for ax, i in zip(axes, n):
        tree.plot_tree(rf.estimators_[i],
            feature_names = fn,
            class_names = cn,
            filled = True,
            label = "none",
            ax = ax)
        ax.set_title(f"Estimator: {i}", fontsize = 11)
    fig.savefig("rf_trees.png")

if __name__ == "__main__":
    results = main()
