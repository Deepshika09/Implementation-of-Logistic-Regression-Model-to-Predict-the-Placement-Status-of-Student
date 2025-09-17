# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Deepshika Hemanth kumar 
RegisterNumber:  212224220020
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

<img width="1221" height="226" alt="311938296-0a5cda11-f165-4e1b-86ec-5f16a2f1ee09" src="https://github.com/user-attachments/assets/83da49b5-627f-4201-8cd9-d08abdd8903e" />

<img width="1091" height="240" alt="311938319-01a8cd00-a0ac-49e9-bdc5-116dc5c20f3d" src="https://github.com/user-attachments/assets/bb03144c-c5ea-4fee-a7d9-211ad798c1fb" />

<img width="982" height="497" alt="311938372-877b2b6f-3436-47e1-9833-e0f9ad9aa560" src="https://github.com/user-attachments/assets/e5e03648-a992-4fd9-b72b-61ea65239e11" />

<img width="982" height="497" alt="311938372-877b2b6f-3436-47e1-9833-e0f9ad9aa560" src="https://github.com/user-attachments/assets/e719e31d-17a1-4ef7-9158-6348ec13c7e5" />

<img width="61" height="48" alt="311938462-9f5231e0-796f-4f94-a0bf-d38784643278" src="https://github.com/user-attachments/assets/12f348f8-9224-4ed2-836c-6093fd05d7b7" />

<img width="982" height="502" alt="311938555-8ff146fd-7c1b-4323-8bba-b9fedaee4ab1" src="https://github.com/user-attachments/assets/0687931f-f625-4e7e-a05f-20517dd27978" />

<img width="922" height="510" alt="311938584-8e99861d-c573-4987-9024-596a68482332" src="https://github.com/user-attachments/assets/b90fe1e7-e103-40ab-b839-0e212f98d554" />

<img width="586" height="263" alt="311938611-a0cf6d5c-79d4-485e-84ea-e6601f115379" src="https://github.com/user-attachments/assets/17f9fb65-b442-4a31-80c4-0ee6aba9d698" />

<img width="762" height="71" alt="311938646-58525a0d-f694-4ddf-ac84-5b596381c5ef" src="https://github.com/user-attachments/assets/64196ed1-1438-429f-865e-7db6b07b2af7" />

<img width="582" height="176" alt="311938696-9a18c485-4dcb-4116-b6d5-f8fdab5de668" src="https://github.com/user-attachments/assets/423b52a3-539d-4b13-8262-82113192c7a9" />

<img width="303" height="33" alt="311938726-e0aeefa2-a16d-40cd-b5ab-b1b1a22044f1" src="https://github.com/user-attachments/assets/880b885b-73e4-4075-bbb1-f98beb6e063f" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
