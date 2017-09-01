#to determine how accurate the best fit line is!
# using R- SQUARED THEORY




from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight') #copy the 538.com style

xs=np.array([1,2,3,4,5,6], dtype=np.float64)# describes how the bytes corresponding to the array shall be interpreted. here, in double precision float.


ys=np.array([5,4,6,5,6,7], dtype=np.float64)
#plt.scatter(xs,ys)
#plt.show()

def best_fit_slope_and_intercept(xs,ys):
	m = (((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs**2)))
	b=mean(ys)-m*mean(xs)
	return (m,b)
m,b=best_fit_slope_and_intercept(xs,ys)

#print(m,b)
# [PEMDAS!!!!!!!]

def squared_error(ys_orig, ys_line):
	return sum((ys_line-ys_orig)**2)

def coeff_of_determination(ys_orig,ys_line):
	y_mean_line= [mean(ys_orig) for y in ys_orig]
	squared_error_regr= squared_error(ys_orig, ys_line)
	squared_error_ymean= squared_error(ys_orig, y_mean_line)
	return 1-(squared_error_regr / squared_error_ymean)

regression_line=[(m*x)+b for x in xs]	
predict_x=8
predict_y=(m*predict_x)+b

r_squared = coeff_of_determination(ys, regression_line) #print r-squared
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs, regression_line)

plt.show()  
 
