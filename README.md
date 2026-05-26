# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the spam dataset.<br>
2.Convert text messages into TF-IDF vectors.<br>
3.Split the dataset into training and testing data.<br>
4.Train the SVM classifier using training data.<br>
5.Predict and evaluate results using a confusion matrix, classification report and accuracy score.<br>

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Athul Krishna A V
RegisterNumber:  212252410017
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['spam','label']]

df['spam'] = df['spam'].map({'ham':0, 'spam':1})

X = df['label']
y = df['spam']

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Ham','Spam'],yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()

score = accuracy_score(y_test,y_pred)
print("Accuracy Score:",score)
print(f"\nConfusion Matrix: {confusion_matrix(y_test,y_pred)}\n")
print(f"Classification Report: {classification_report(y_test,y_pred)}")
```

## Output:
<img width="651" height="706" alt="image" src="https://github.com/user-attachments/assets/09b64e75-9a3c-412f-931b-ef1f56fe08ec" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
