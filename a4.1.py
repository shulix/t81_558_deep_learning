
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from pprint import pprint
df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/crx.csv", 
    na_values=['NA', '?','nan'])

#display(df)

pd.set_option('display.max_columns', 40) 
pd.set_option('display.max_rows', 5)

df = pd.concat([df,pd.get_dummies(df['a9'],prefix="a9")],axis=1)
df.drop('a9', axis=1, inplace=True)
df = pd.concat([df,pd.get_dummies(df['a10'],prefix="a10")],axis=1)
df.drop('a10', axis=1, inplace=True)
df = pd.concat([df,pd.get_dummies(df['a11'],prefix="a11")],axis=1)
df.drop('a11', axis=1, inplace=True)
df = pd.concat([df,pd.get_dummies(df['a12'],prefix="a12")],axis=1)
df.drop('a12', axis=1, inplace=True)
df = pd.concat([df,pd.get_dummies(df['a13'],prefix="a13")],axis=1)
df.drop('a13', axis=1, inplace=True)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 5)
print(df)
#x = df[['s3', 'a8', 'a9', 'a10','a11', 'a12', 'a13','a15']].values
z=df['a2'].values
#df['a14'] = zscore(df['a14'])
#med = df['a2'].median()
df['a2'] = df['a2'].fillna(0)
#print(list(df.columns))
x = df[['s3', 'a8', 'a15','a9_f', 'a9_t', 'a10_f', 'a10_t', 'a11_0', 'a11_1', 'a11_2', 'a11_3', 'a11_4', 'a11_5', 'a11_6', 'a11_7', 'a11_8', 'a11_9', 'a11_10', 'a11_11', 'a11_12', 'a11_13', 'a11_14', 'a11_15', 'a11_16', 'a11_17', 'a11_19', 'a11_20', 'a11_23', 'a11_40', 'a11_67', 'a12_f', 'a12_t', 'a13_g', 'a13_p', 'a13_s']].values
print(x)
y = df['a2'].values
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(15, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x,y,verbose=0,epochs=100)


pred = model.predict(x)
print(pred.shape)

#print(pred[0:10])
score = np.sqrt(metrics.mean_squared_error(pred,y))
print("------------------score---------------------")
print(score)
# Sample predictions
for i in range(690):
    if 'nan' in str(z[i]):
        print(str(z[i])+"  "+str(y[i])+"  "+str(pred[i]))
