# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
#1.Import the standard libraries.
#2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
#3.Import LabelEncoder and encode the dataset.
#4.Import LogisticRegression from sklearn and apply the model on the dataset.
#5.Predict the values of array.
#6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
#7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.Rajalakshmi
RegisterNumber: 212219040116
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## Original data(first five columns):
![original data](https://user-images.githubusercontent.com/87656716/173214355-337ae45c-51ca-43c8-a862-b57c84bda97e.png)

## Data after dropping unwanted columns(first five):
![data after dropping](https://user-images.githubusercontent.com/87656716/173214386-2b9c9312-4e9e-490c-aa74-5647550ecf95.png)

## Checking the presence of null values:
![null value](https://user-images.githubusercontent.com/87656716/173214406-206b76c5-d89e-4280-9491-ecd7902a5021.png)

## Checking the presence of duplicated values:
![duplicate value](https://user-images.githubusercontent.com/87656716/173215049-b2da5f01-0255-484c-9519-580312a1c228.png)

## Data after Encoding:
![data after encoding](https://user-images.githubusercontent.com/87656716/173215068-57b4c79c-1ae1-4e08-8d0b-f268c4063886.png)

## X Data:
![x data](https://user-images.githubusercontent.com/87656716/173215099-f654d68b-b30e-4f7c-b4dc-50d1a69db152.png)

## Y Data:
![y data](https://user-images.githubusercontent.com/87656716/173215147-a3eb3f38-f527-4f6a-8421-54fa9f0afcdf.png)

## Predicted Values:
![predicted value (2)](https://user-images.githubusercontent.com/87656716/173215235-9e706f9b-ad98-44bd-9717-422797f74b59.png)
 
## Accuracy Score:
![accuracy](https://user-images.githubusercontent.com/87656716/173215211-3761c20f-32a1-4808-aecb-f70c2ebfc7a1.png)

## Confusion Matrix:
![confusion matrix](https://user-images.githubusercontent.com/87656716/173215267-6dda6572-e3c1-48ab-89d0-7a143e2f55fc.png)

## Classification Report:
![classification report](https://user-images.githubusercontent.com/87656716/173215303-076036ac-e8d2-4a7f-8d3e-782710103be6.png)

## Predicting output from Regression Model:
![predicted output](https://user-images.githubusercontent.com/87656716/173215313-09a613b9-7f02-479f-90d1-c427b52d384d.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
