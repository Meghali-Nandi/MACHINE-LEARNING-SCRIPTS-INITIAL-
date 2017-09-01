import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

dataset={'k':[[1,2],[2,3],[3,1]],'x':[[6,5],[7,7],[8,6]]}

new_features=[5,100]
#[[plt.scatter(ii[0],ii[1],s=100) for ii in dataset[i]] for i in dataset]
#plt.show()

def knn(data,predict,k=3):
	if len(data)>=k:
		warnings.warn('k is set to a value less')
	
	distance=[]
	for group in data:
		for features in data[group]:
			eucl_dist=np.linalg.norm(np.array(features)-np.array(predict))
			distance.append([eucl_dist, group])

	votes=[i[1] for i in sorted(distance)[:k]]
	print(Counter(votes).most_common(1))
	vote_res=Counter(votes).most_common(1)[0][0]
	return vote_res
#res=knn(dataset, new_features, k=1)
#print(res)

df=pd.read_csv("bc.data")
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
full_data=df.astype(float).values.tolist()#converting to float list of list AS otherwise there were string values and may cause error while calculations

#shuffle data as it is convcerted to list of lists
random.shuffle(full_data)

test_size=0.2

