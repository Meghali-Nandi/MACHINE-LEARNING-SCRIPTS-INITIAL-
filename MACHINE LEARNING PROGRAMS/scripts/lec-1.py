from numpy import *
from array import array
import pandas as pd

def compute_error(b,m,points):
    error=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        error+=(y-(m*x-b))**2
    return error/float(len(points))

def gradient(curr_b,curr_m,points,rate):
    b_grad=0
    m_grad=0
    N=float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_grad+=-(2/N)*(y-((curr_m*x)+curr_b))
        m_grad+=-(2/N)*x*(y-((curr_m*x)+curr_b))
    new_b=curr_b-(rate*b_grad)
    new_m=curr_m-(rate*m_grad)
    return [new_b,new_m]

def gradient_descent_runner(points,start_b,start_m,rate,iterations):
    b=start_b
    m=start_m
    for i in range(iterations):
        b,m=gradient(b,m,array(points),rate)
    return [b,m]

def run():
    points=pd.read_csv('data.csv')
    rate=0.0001
    ini_b=0
    ini_m=0
    iteration=1000
    [b,m]=gradient_descent_runner(points,ini_b,ini_m,rate)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(iteration, b, m, compute_error(b, m, points)))

if __name__=='__main__':
    run()
