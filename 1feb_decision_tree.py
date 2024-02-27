# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:16:38 2024

@author: rohit
"""
import pandas as pd
df=pd.read_csv('C:/2-dataset/salaries.csv')
df.head()
df.columns
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
target
from sklearn import tree
model= tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
#is salary of Google,Computer Engineer,Bachelor degree >100K?
model.predict([[2,1,0]])
#is salary of Google,Computer Engineer, Bachelor degree ....
model.predict([[2,1,1]])
