# Content of file supervised_learning.py
from sklearn.datasets import load_iris  # Esempio di dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (for example, iris)
data = load_iris()
X, y = data.data, data.target

# Split the data in training e testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalization of data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train the classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#Valutation of the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Content of file unsupervised_learning.py
from sklearn.cluster import DBSCAN
import numpy as np

# Data
data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Clustering of data using DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(data)

#Extraction of cluster labels
labels = db.labels_

