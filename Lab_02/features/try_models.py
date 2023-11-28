import pickle

import pandas as pd

with open('train_features.pkl', 'rb') as f:
    train_features = pickle.load(f)

with open('train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

with open('val_features.pkl', 'rb') as f:
    val_features = pickle.load(f)

with open('val_labels.pkl', 'rb') as f:
    val_labels = pickle.load(f)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_features, train_labels)

# Predict on the validation set
rf_predictions = rf_classifier.predict(val_features)

# Example for Random Forest
report = classification_report(val_labels, rf_predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("report_RandomForest.csv")



from sklearn.svm import SVC

# Create a SVM classifier
svm_classifier = SVC()
svm_classifier.fit(train_features, train_labels)

# Predict on the validation set
svm_predictions = svm_classifier.predict(val_features)

report = classification_report(val_labels, svm_predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("report_SVC.csv")



from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(train_features, train_labels)
knn_predictions = knn_classifier.predict(val_features)

report = classification_report(val_labels, knn_predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("report_KNeighbors.csv")



from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression()
lr_classifier.fit(train_features, train_labels)
lr_predictions = lr_classifier.predict(val_features)

report = classification_report(val_labels, lr_predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("report_LogisticRegression.csv")



from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()
nb_classifier.fit(train_features, train_labels)
nb_predictions = nb_classifier.predict(val_features)

report = classification_report(nb_predictions, lr_predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("report_naiveBayes.csv")



