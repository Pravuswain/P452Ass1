from numpy import append
import Matrix_operation as op
import numpy as np



A= op.file_opener('Ass1mat2A.txt')
B = [-5/3,2/3,3,-4/3,-1/3,5/3]
#Q2 part i
augmat = op.aug_mat(A,B)
#in Gauss Jordan method
sol = op.GaussJordan(augmat)
print(f"solution of equations in gauss Jordan Method: {sol[6]}")

#empty matrix to store the final matrix after inversion
I1 = []
I2 = []
I3 = []
I4 = []
I5 = []
I6 = []

# creating identity matrix of single column
for i in range(len(B)):
    I = [0 for j in range(len(B))]
    I[i] = 1
    I1.append(I[0])
    I2.append(I[1])
    I3.append(I[2])
    I4.append(I[3])
    I5.append(I[4])
    I6.append(I[5])
#augmenting and forming 6 augmented matrix
augmat1= op.aug_mat(A,I1)
augmat2= op.aug_mat(A,I2)
augmat3= op.aug_mat(A,I3)
augmat4= op.aug_mat(A,I4)
augmat5= op.aug_mat(A,I5)
augmat6= op.aug_mat(A,I6)

#using gaussjordan on each one
X3_1 = op.GaussJordan(augmat1)
X3_2 = op.GaussJordan(augmat2)
X3_3 = op.GaussJordan(augmat3)
X3_4 = op.GaussJordan(augmat4)
X3_5 = op.GaussJordan(augmat5)
X3_6 = op.GaussJordan(augmat6)
#taking all arrays together and forming matrix as X(for gauss jordan), Y(for Gauss Seidel), Z(for Jacobi Inverse), W(for conjugate gradient)
x =[X3_1[6],X3_2[6],X3_3[6],X3_4[6],X3_5[6],X3_6[6]]



print('Inverse using Gauss Jordan :')
print(np.array(x))
#using gauss Seidel on each
X1_1=op.Gauss_Seidel(A,I1,x=None,tol=1e-4)
X1_2=op.Gauss_Seidel(A,I2,x=None,tol=1e-4)
X1_3=op.Gauss_Seidel(A,I3,x=None,tol=1e-4)
X1_4=op.Gauss_Seidel(A,I4,x=None,tol=1e-4)
X1_5=op.Gauss_Seidel(A,I5,x=None,tol=1e-4)
X1_6=op.Gauss_Seidel(A,I6,x=None,tol=1e-4)

Y=[X1_1,X1_2,X1_3,X1_4,X1_5,X1_6]

#using jacobi inv on each
X2_1=op.Jacobi_Inv(A,I1,x=None,tol=1e-4)
X2_2=op.Jacobi_Inv(A,I2,x=None,tol=1e-4)
X2_3=op.Jacobi_Inv(A,I3,x=None,tol=1e-4)
X2_4=op.Jacobi_Inv(A,I4,x=None,tol=1e-4)
X2_5=op.Jacobi_Inv(A,I5,x=None,tol=1e-4)
X2_6=op.Jacobi_Inv(A,I6,x=None,tol=1e-4)

Z=[X2_1,X2_2,X2_3,X2_4,X2_5,X2_6]


print("Matrix Inverse by Gauss Seidal:")
print(np.array(Y))
print("Matrix inverse using jacobi Inverse:")
print(np.array(Z))

#assume a value of x 
x = [1,1,1,1,1,1]
X4_1 = op.Conjugate_Grad(A,x,I1,tol = 0.0001)
X4_2 = op.Conjugate_Grad(A,x,I2,tol=0.0001)

print("first two column of Inverted matrix in Conjugate gradient form")
print(X4_1)
print(X4_2)

#For residual plot we get data using Same function with a little variation to open and edit a file
sol2 = op.GaussJordan_data(augmat,B,'A1Q2data1.dat')
sol3 = op.Gauss_Seidel_data(A,B,'A1Q2data2.dat')

with open(r"A1Q2data1.dat") as datFile:
    P = ([data.split()[0] for data in datFile])
print(P)
'''

solution of equations in gauss Jordan Method: [-1.3347763347763348, -1.002886002886003, 0.5613275613275613, -1.2842712842712842, -1.235209235209235, 0.2481962481962482]
Inverse using Gauss Jordan :
[[0.93506494 0.87012987 0.25974026 0.20779221 0.41558442 0.16883117]
 [0.29004329 0.58008658 0.17316017 0.13852814 0.27705628 0.11255411]
 [0.08658009 0.17316017 0.32034632 0.05627706 0.11255411 0.10822511]
 [0.20779221 0.41558442 0.16883117 0.93506494 0.87012987 0.25974026]
 [0.13852814 0.27705628 0.11255411 0.29004329 0.58008658 0.17316017]
 [0.05627706 0.11255411 0.10822511 0.08658009 0.17316017 0.32034632]]
Matrix Inverse by Gauss Seidal:
[[0.92198926 0.28420782 0.08447958 0.19709903 0.13375434 0.05455848]
 [0.85840923 0.57485497 0.17127691 0.4059948  0.27277561 0.11101313]
 [0.24567971 0.16688531 0.31808767 0.15733333 0.10742101 0.10637717]
 [0.19709903 0.13375434 0.05455848 0.92631173 0.28613641 0.08517372]
 [0.4059948  0.27277561 0.11101313 0.86228245 0.57658374 0.17189922]
 [0.15733333 0.10742101 0.10637717 0.25032796 0.16895916 0.31883408]]
Matrix inverse using jacobi Inverse:
[[0.91552693 0.28132617 0.08189324 0.19182681 0.12788454 0.05244444]
 [0.85262345 0.57227282 0.16895916 0.40126302 0.26750868 0.10911696]
 [0.23872948 0.16378647 0.31530643 0.15166497 0.10110998 0.1041041 ]
 [0.19182681 0.12788454 0.05244444 0.91552693 0.28132617 0.08189324]
 [0.40126302 0.26750868 0.10911696 0.85262345 0.57227282 0.16895916]
 [0.15166497 0.10110998 0.1041041  0.23872948 0.16378647 0.31530643]]
first two column of Inverted matrix in Conjugate gradient form
[1.12190246 0.41852244 0.10284161 0.43807836 0.2082644  0.11385215]
[1.17614357 0.71296611 0.24508093 0.74125484 0.45167467 0.16563752]
'''
