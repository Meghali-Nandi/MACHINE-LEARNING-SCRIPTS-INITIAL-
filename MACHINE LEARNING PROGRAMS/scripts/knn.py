import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

df=pd.read_csv("bc.data")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train, X_test,Y_train,Y_test=cross_validation.train_test_split(X,y,test_size=0.1)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy=clf.score(X_test,Y_test)
#print(accuracy)

example=np.array([[4,2,1,1,1,2,3,2,1],[4,2,2,4,1,2,3,2,1]])

example=example.reshape(len(example),-1)
pred=clf.predict(example)

print(pred)
