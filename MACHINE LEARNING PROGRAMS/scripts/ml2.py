import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style #style to show decent view of graph
style.use('ggplot') #to specify which type of style we wanna use
import pickle

df = quandl.get('WIKI/GOOGL')
import pickle

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close' 
df.fillna(-99999, inplace=True) #to fill the NAN spaces with -99999

forecast_out = int(math.ceil(0.01*len(df))) #to collect 10% of the data in integer form


df['label'] = df[forecast_col].shift(-forecast_out) #to shift the data of forecast_col to top by forecast_out,i.e. 32 in this case



X = np.array(df.drop(['label'], 1))#to remove label to be included in y
X = preprocessing.scale(X)

X_lately = X[-forecast_out:] # grab everything from the last 32 rows onward. shall give us the data against which we are gonna predict. 
X= X[:-forecast_out] # grab everything up to the last 32 rows.

df.dropna(inplace=True)
y = np.array(df['label'])#to include only label

# Split data into training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


#one method of finding accuracy using linear regression

#clf = LinearRegression(n_jobs=10) # defining a classifier
#n_jobs is one of the algorithms that can be threaded. n_jobs signifies how many jobs are gonna run at a given time. (the default is 1. here its 10 jobs. if -1 given, then it calculates as many jobs as can be processed by the processor of the laptop/device) 

#clf.fit(X_train, y_train) # training the model for finding accuracy

#with open('linearregression.pickle','wb') as f: #opening a file with the intention to write as binary and assigning a temporary variable as f
#	pickle.dump(clf,f)# dumping the classifier (what, where)
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in) #Read a string from the open file object file and interpret it as a data stream, reconstructing and returning the original object hierarchy. 

accuracy = clf.score(X_test, y_test) # testing the model for finding accuracy


#another method for accuracy using svm although theres a huge difference between the accuracies in the two methods
#clf=svm.SVR(kernel='polynomial')#u can change 'polynomial' as per ur wish to get better accuracies. check out what kernel is! and how to use which kernel at which time!

#clf.fit(X_train, y_train) # training the model for finding accuracy
#accuracy = clf.score(X_test, y_test) # testing the model for finding accuracy

#print accuracy

fore_set=clf.predict(X_lately) #next 32 days of unknown values.

#print(fore_set, accuracy, forecast_out)

#to get the dates
df['Forecast']=np.nan #to show that the entire data is full of nan value
last_date=df.iloc[-1].name #to get the name of the last date 

last_unix=last_date.timestamp()
one_day=86400 #no. of sec. in a day
next_unix=last_unix+one_day

for i in fore_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)#converts seconds sinch epoch to a naive datetime object that represents local time

	next_unix+=one_day
	df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i] #np.nan is for replacing all zeros with nan values
#loc takes the index of the dates. if the date is already present, it stays there. n if absent, it gives the date to the corresponding data.


  #iterating in the fore_set and taking out the values and making them the future features and not numbers. takes the first columns and sets them. the last column is the forecast.

df['Adj. Close'].plot()
df['Forecast'].plot()

plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

 




	
  




