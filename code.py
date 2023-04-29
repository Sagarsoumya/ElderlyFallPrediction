import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
# Define file path for CSV file
file_path = 'C:\\Users\\sagar\\OneDrive\\Documents\\cStick.csv'

# Read in data from CSV file
dataset = pd.read_csv(file_path)

# Split data into features (X) and target (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

pca = PCA(n_components=0.95)
pca.fit(X)
X = pca.transform(X)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a Random Forest Classifier model
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate model performance using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#precision and recall
precision=precision_score(y_test,y_pred,labels=[0, 1], average='weighted')
recall=recall_score(y_test,y_pred,labels=[0, 1], average='weighted')

print("Precision: ",precision)
print("Recall: ",recall)

#f1 score
fscore=f1_score(y_test,y_pred,average='weighted')
print(fscore)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)