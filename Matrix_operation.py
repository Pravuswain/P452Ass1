
import numpy as np
import copy
import random
from matplotlib import pyplot as plt 

#MATRIX OPERATIONS
#b is any matrix
#Function to read the file and getting numbers in a Matrix
def file_opener(name):
    with open(name,'r') as f:
        a =[[float(num) for num in line.split(',')] for line in f]
    return a


#GAUSS JORDAN
#step 1 - creation of augmented Matric(aug_mat)
def aug_mat(A,B):
    #creation of zero mat with required rows and column numbers 
    aug_AB = [[0 for a in range(len(A))]for b in range(len(A)+1)]
    for i in range(len(A)):
        for j in range(len(A)+1):
            if j>=(len(A)):
                aug_AB[j][i] = B[i]
            else :
                aug_AB[j][i] = A[j][i]
    return aug_AB
#done

#step 2 - creating reduced REF form
def GaussJordan(aug_mat):      
    for i in range(len(aug_mat[0])):
        p = aug_mat[i][i]
        for j in range(len(aug_mat)):
            aug_mat[j][i] = aug_mat[j][i]/p
        for k in range(len(aug_mat[0])):
            if k == i or aug_mat[i][k] == 0:
                next
            else:
                factor = aug_mat[i][k]
                for l in range(len(aug_mat)):
                    aug_mat[l][k] = aug_mat[l][k] - factor*aug_mat[l][i]

    return aug_mat
#for getting data 
#step 2 - creating reduced REF form
def GaussJordan_data(aug_mat,B,name):      
    for i in range(len(aug_mat[0])):
        p = aug_mat[i][i]
        for j in range(len(aug_mat)):
            aug_mat[j][i] = aug_mat[j][i]/p
        for k in range(len(aug_mat[0])):
            if k == i or aug_mat[i][k] == 0:
                next
            else:
                factor = aug_mat[i][k]
                for l in range(len(aug_mat)):
                    aug_mat[l][k] = aug_mat[l][k] - factor*aug_mat[l][i]
        with open(name, 'a') as data:
            print(aug_mat[len(B)],',',i, file= data)

    return aug_mat



#Define Jacobi Method 
def Jacobi_Inv(A,B, x=None, tol = 10**(-5)):
    # Initial guess if required
    if x is None:
        x = np.zeros(len(A))

    # vector of the diagonal elements of A and subtract

    D = np.diag(A)
    LU = A - np.diagflat(D)

    # Iterate till tolerance
    err = np.inf
    while err>tol:
        # Storing previous x assumed
        x_p = copy.deepcopy(x)
        x = (B - np.dot(LU,x)) / D
        x_p = x-x_p
        err = sum(x_p[i]**2 for i in range(len(x)))
    return x

#Define Gauss-Seidel 
def Gauss_Seidel(A, B, x=None, tol = 1e-5):
    n = len(A)
    if x is None: x = np.zeros(n)
    err = np.inf

    while err>tol:
        sum = 0
        #calculation of x 
        for i in range(n):
            d = B[i]
            for j in range(n):
                if(i != j):
                    d-=A[i][j] * x[j]
            # Storing previous and updating the value of our solution
            temp = x[i]
            x[i] = d / A[i][i]
            sum += (x[i]-temp)**2
 
        
        #Error update
        err = sum
    
    return x
def Gauss_Seidel_data(A, B,name, x=None, tol = 1e-5):
    n = len(A)
    if x is None: x = np.zeros(n)
    err = np.inf

    while err>tol:
        sum = 0
        #calculation of x 
        for i in range(n):
            d = B[i]
            for j in range(n):
                if(i != j):
                    d-=A[i][j] * x[j]
            # Storing previous and updating the value of our solution
            temp = x[i]
            x[i] = d / A[i][i]
            sum += (x[i]-temp)**2
            with open(name, 'a') as data:
                print(x,',',i, file= data)
        
        #Error update
        err = sum
    
    return x

#define function for Conjugate gradient
def Conjugate_Grad(A,X,b, tol = 0.00001):        
    
    n = len(b)
    r = b - np.dot(A,X)
    d = r.copy()
    i = 1
    while i<=n:
        u = np.dot(A,d)
        alpha = np.dot(d,r)/np.dot(d,u)
        X = X + alpha*d
        r = b - np.dot(A,X)
        if np.sqrt(np.dot(r,r)) < tol:
            break
        else:
            beta = -np.dot(r,d)/np.dot(d,u)
            d = r + beta*d
            i = i+1

    return X



    
  

