#######A Python Script to predict the sex of a person based on user input of height weight and bmi value
#######values are recorded in an excel file a data set of 500 people


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd
from sklearn.metrics import accuracy_score

from pandas import ExcelFile
#predicts the gender of a person after feeding the following
#[hieght, weight, body mass index(integers)]

df= pd.read_excel("C:/Users/RV/Desktop/DataSets/PersonData.xlsx",sheet_name='500 Person')
print("Column Headings")
print(df.columns)

##print(df['Gender'])

listGender = df['Gender']
##print(listGender)

listHeight = df['Height']
listWeight = df['Weight']
listIndex = df['Index']

###Make a tuples of the height, weight and index list to feed the code for learning

Combo = zip(listHeight,listWeight,listIndex)
listCombo = list(Combo)

#for x in range(len(listCombo)):
 #   print(listCombo[x]),


inputList=[]
inputHeight=int(input("Enter the Height(cms):"))
inputWeight=int(input("Enter the Weight(lbs):"))
inputIndex=int(input("Enter the BMI(0-5):"))
inputList.append(inputHeight)
inputList.append(inputWeight)
inputList.append(inputIndex)
print(inputList)


##PRedicting Gender Model


####DecisionTreeClassifier
clfDT=tree.DecisionTreeClassifier()
clfDT=clfDT.fit(listCombo,listGender)
predictionDT= clfDT.predict([inputList])
print("DTClassifier Prediction")
print(predictionDT)
#print("Accuracy : %.2f percent" % (100 * accuracy_score(['Male', 'Male', 'Female'], ['Male', 'Male', 'Female'])))

###########RandomForestClassifier
clfRF=RandomForestClassifier()
clfRF=clfRF.fit(listCombo,listGender)
predictionRF=clfRF.predict([inputList])
print("RFClassifier Prediction")
print(predictionRF)
#print("Accuracy : %.2f percent" % (100 * accuracy_score(['Male', 'Male', 'Female'], ['Male', 'Male', 'Female'])))

#######GaussianPRocessClassifier
clfGP=GaussianProcessClassifier()
clfGP=clfGP.fit(listCombo,listGender)
predictionGP=clfGP.predict([inputList])
print("GPClassifier Prediction")
print(predictionGP)

#####KNeighboursClassifiers
clfKN=KNeighborsClassifier()
clfKN=clfKN.fit(listCombo,listGender)
predictionKN=clfKN.predict([inputList])
print("KNClassifier Prediction")
print(predictionKN)

#####MLPClassifier
clfMLP=MLPClassifier()
clfMLP=clfMLP.fit(listCombo,listGender)
predictionMLP=clfMLP.predict([inputList])
print("MLPClassifier Prediction")
print(predictionMLP)

#MLPprob=clfMLP.predict_proba([inputList])
#print(MLPprob)















