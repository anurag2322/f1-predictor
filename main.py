import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report

results = pd.read_csv('data\results.csv')
races = pd.read_csv('data\races.csv').drop(['time'],axis=1)
quali = pd.read_csv('data\qualifying.csv')
drivers = pd.read_csv('data\drivers.csv').drop(['number','url'],axis=1)
drivers['forename'] = drivers['forename']+drivers['surname']
constructors = pd.read_csv('data\constructors.csv').drop(['url'],axis=1)
circuit = pd.read_csv('data\circuits.csv').drop(['name','location','country','lat','lng','alt','url'],axis=1)


df1 = pd.merge(races,results,how='inner',on=['raceId']).drop(['positionText','url'],axis=1)
df1.loc[df1['position']=='\\N','position'] = 0
df1['position'] = df1['position'].astype(int)
df1.loc[df1['number']=='\\N','number'] = 0
df1['number'] = df1['number'].astype(int)

df2 = pd.merge(df1,quali,how='inner',on=['raceId','constructorId','driverId','position','number'])
df3 = pd.merge(df2,drivers,how='inner',on=['driverId'])
df4 = pd.merge(df3,constructors,how='inner',on=['constructorId'],suffixes = ('_race','_con')).drop(['surname'],axis = 1)
df5 = pd.merge(df4,circuit,how='inner',on=['circuitId'])

label_encoder = preprocessing.LabelEncoder()

#labelling
df5['name_race'] = label_encoder.fit_transform(df5['name_race'])
df5['driverRef'] = label_encoder.fit_transform(df5['driverRef'])
df5['code'] = label_encoder.fit_transform(df5['code'])
df5['forename'] = label_encoder.fit_transform(df5['forename'])
df5['nationality_race'] = label_encoder.fit_transform(df5['nationality_race'])
df5['constructorRef'] = label_encoder.fit_transform(df5['constructorRef'])
df5['name_con'] = label_encoder.fit_transform(df5['name_con'])
df5['nationality_con'] = label_encoder.fit_transform(df5['nationality_con'])
df5['circuitRef'] = label_encoder.fit_transform(df5['circuitRef'])

df5 = df5.drop(['date','time','milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed','q1','q2','q3','dob'],axis=1)
df5 = df5.drop(['raceId','circuitId','resultId','driverId','constructorId'],axis=1)

print(df5.dtypes)

print(df5['position'])

def position_index(x):
    if x<4:
        return 1
    if x>10:
        return 3
    else :
        return 2

mainfeatures = df5.drop(['position'],axis=1)
X = np.asarray(mainfeatures)
y = df5['position'].apply(lambda x: position_index(x))

print(y)

X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2,random_state=4)
classifier = svm.SVC(gamma = 'auto',C=2)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
#print(y_predict)

print(classification_report(y_test,y_predict))