import statistics
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Read in dataset
dataset = pd.read_csv('cStick.csv')

# Split dataset into features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

list_accuracy = []
list_precision = []
list_recall = []
list_f1 = []
list_sensitivity = []

i = 0
while i < 100:
    pca = PCA(n_components=0.95)
    pca.fit(X)
    X = pca.transform(X)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalize the data
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # create a decision tree classifier
    clf = DecisionTreeClassifier()

    # train the classifier on the training data
    clf.fit(X_train, y_train)

    # make prediction on the testing data
    y_pred = clf.predict(X_test)

    # calculate the accuracy, precision, recall, sensitivity and F1 score of the classifier
    score_list = []
    score_list.append(accuracy_score(y_test, y_pred))
    score_list.append(precision_score(y_test, y_pred, labels=[0,1], average='weighted'))
    score_list.append(recall_score(y_test, y_pred, labels=[0,1], average='weighted'))
    score_list.append(recall_score(y_test, y_pred, labels=[0,1], average='weighted'))
    score_list.append(f1_score(y_test, y_pred, average='weighted', labels=[0, 1]))

    list_accuracy.append(score_list[0])
    list_precision.append(score_list[1])
    list_recall.append(score_list[2])
    list_sensitivity.append(score_list[3])
    list_f1.append(score_list[4])

    i = i + 1     

print("mean accuracy:", statistics.mean(list_accuracy))
print("standard deviation of accuracy:", np.std(list_accuracy))

print("mean precision:", statistics.mean(list_precision))
print("standard deviation of precision:", np.std(list_precision))

print("mean recall:", statistics.mean(list_recall))
print("standard deviation of recall:", np.std(list_recall))

print("mean sensitivity:", statistics.mean(list_sensitivity))
print("standard deviation of sensitivity:", np.std(list_sensitivity))

print("mean F1-score:", statistics.mean(list_f1))
print("standard deviation of F1-score:", np.std(list_f1))
