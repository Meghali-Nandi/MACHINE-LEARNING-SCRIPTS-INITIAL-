import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D



def data_analysis():
    data=pd.read_csv('data.csv')
    drop_features=['color','clarity','cut','table','depth','y','x']
    for f in drop_features:
        del data[f]
    final_list=[list(data.loc[random.randint(1,53000)])for i in range(1500)]    #can change the value according to the requirements
    return([[i[1],i[3],i[2]]for i in final_list])
    
def plot_graph(points,plane):
    
    f1 = [p[0] for p in points] 
    f2 = [p[1] for p in points] 
    f3 = [p[2] for p in points]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis([0, 5, 3,11])
    ax.set_xlabel('Mass')
    ax.set_ylabel('Width')
    ax.set_zlabel('Price')
    plt.scatter(f1, f2, f3)
    
    plt.show()
    
def error_analysis(c,a,b,points):
    error=0
    for p in points:
        x,y,z=p[0],p[1],p[2]
        error+=(z-(c+a*x+b*y))**2
    return error/float(len(points))

def gradient(curr_c,curr_a,curr_b,points,rate):
    a_grad=0
    b_grad=0
    c_grad=0
    N=float(len(points))
    for i in points:
        x, y, z = i[0], i[1], i[2]
       
        fx = curr_c + curr_a*x + curr_b*y
        
        curr_c += -(2/N) * (z - fx)
        curr_a += -(2/N) * x * (z - fx)
        curr_b += -(2/N) * y * (z - fx)
    new_c=curr_c-(rate*c_grad)
    new_a=curr_a-(rate*a_grad)
    new_b=curr_b-(rate*b_grad)
    return [new_c,new_a,new_b]

def gradient_descent_runner(points,start_c,start_a,start_b,rate,iterations):
    z = start_c
    x = start_a
    y = start_b
    for i in range(iterations):
        z,x,y = gradient(z,x,y, np.array(points),rate)
    return [z,x,y]


def run(points):
    rate = 0.001
    initial_z = initial_x = initial_y= 0 
    iterations = 50
    
    print("Starting gradient descent at z = {0}, x = {1}, y = {2} errors = {3}".format(initial_z, initial_x, initial_y, error_analysis(initial_z, initial_x, initial_y, points)))
    
    [z, x, y] = gradient_descent_runner(points, initial_z, initial_x, initial_y,rate,iterations)
    print("After {0} iterations z = {1}, x = {2}, y = {3} errors = {4}".format(iterations, z, x,y, error_analysis(z,x,y, points)))
    plot_graph(points, [z,x,y])

if __name__ == '__main__':
    run(data_analysis())
    
