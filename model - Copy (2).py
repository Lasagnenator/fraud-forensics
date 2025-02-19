import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import itertools

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

# Shuffle data and sort by fraud class
#transactional_data = transactional_data.sample(frac=1.0)
#transactional_data.sort_values("is_fraud", inplace=True)

# Could consider including the time of day
relevant_columns = ["gender", "city", "state", "city_pop",
                    "job", "category", "amt"]

relevant = transactional_data[relevant_columns].convert_dtypes()
# Convert jobs to simplified form
print("Simplifing job titles.")
relevant = relevant.assign(job=relevant["job"].str.split(",", expand=True)[0])

# Turn times into seconds since midnight.
print("Parsing times.")
# This is horrible but it works.
#relevant["second of day"] = pd.to_timedelta(pd.to_datetime(transactional_data["unix_time"], unit="s").dt.time.astype(str)).dt.total_seconds()
relevant["second of day"] = transactional_data["unix_time"] % 86400

# Columns that need encoding from string to class.
print("Classifying categorical columns.")
enc_columns = ["gender", "city", "state", "job", "category"]
encoders = [LabelEncoder() for _ in enc_columns]
for c, enc in zip(enc_columns, encoders):
    relevant.loc[:, c] = enc.fit_transform(relevant[c]).reshape((-1, 1))

#relevant = relevant[["second of day", "amt"]]
relevant = relevant.rename(columns={"amt": "amount"})

target_column = "is_fraud"

X = relevant
Y = transactional_data[target_column].convert_dtypes()

parameters = [(4, 9, 0.016)] # This seems to work fine.
#parameters = [(5, 15, 0.013)]
#parameters = itertools.product(range(4,20), range(9,20,2))
scores = []
for md, mln, a in parameters:
#for a in np.linspace(0, 0.020, 21):
    # 5k of each class for testing
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    X_train, X_test, Y_train, Y_test = train_test_eq_split(X, Y, 5000)

    # The hyperparameters are very arbitrary. Need to see what works best.
    print(f"Training model with parameters {md}, {mln}, {a}.")
    clf = tree.DecisionTreeClassifier(max_depth=md, max_leaf_nodes=mln, class_weight="balanced", ccp_alpha=0.016)
    #clf = tree.DecisionTreeClassifier(max_depth=md, max_leaf_nodes=mln, class_weight="balanced", ccp_alpha=0.013)
    clf = clf.fit(X_train, Y_train)

    score = clf.score(X_test, Y_test)*100
    print(f"Model accuracy (score): {score:.2f}%")
    scores.append(score)

# Model description.
classes = ["not fraud", "is fraud"]
text = tree.export_text(clf, feature_names=relevant.columns, class_names=classes)
print(text)

# Mappings


# Category mapping.
categories = np.array(list(zip(range(encoders[-1].classes_.shape[0]), encoders[-1].classes_)))
#print("Category mapping:")
#print(categories)
#display = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", plot_method="pcolormesh", shading="auto", grid_resolution=1000)
#display.ax_.scatter(X["second of day"], X["amount"], c=Y, edgecolor="black", alpha=0.05)
#display.ax_.set_ylim(0, 1500)
#plt.plot(np.linspace(0, 0.020, 21), scores, marker="o",drawstyle="steps-post")
tree.plot_tree(clf, feature_names=list(relevant.columns), class_names=classes, label="none", impurity=False, filled=True, rounded=True)
plt.show()
