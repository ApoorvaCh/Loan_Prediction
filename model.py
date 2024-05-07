import pandas as pd 
import numpy as np 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("LoanApprovalPrediction.csv") 
data.drop(['Loan_ID'],axis=1,inplace=True)
    
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
  data[col] = label_encoder.fit_transform(data[col])


for col in data.columns: 
  data[col] = data[col].fillna(data[col].mean())  

  
X = data.drop(['Loan_Status'],axis=1) 
Y = data['Loan_Status'] 
  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.4, 
                                                    random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

rfc = RandomForestClassifier(n_estimators = 7, criterion = 'entropy', random_state =7)
rfc.fit(X_train, Y_train) 
predictions = rfc.predict(X_test) 

pickle.dump(rfc, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

predict = model.predict(X_test.head(1))
print("Loan Status:", "Yes" if predict[0] == 1 else "No")

