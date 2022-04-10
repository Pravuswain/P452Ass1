from numpy import append
import Matrix_operation as op
import numpy as np



A= op.file_opener('Ass1mat2A.txt')
B = [-5/3,2/3,3,-4/3,-1/3,5/3]

I1 = []
I2 = []
I3 = []
I4 = []
I5 = []
I6 = []
for i in range(len(B)):
    I = [0 for j in range(len(B))]
    I[i] = 1
    I1.append(I[0])
    I2.append(I[1])
    I3.append(I[2])
    I4.append(I[3])
    I5.append(I[4])
    I6.append(I[5])








augmat1= op.aug_mat(A,I1)
augmat2= op.aug_mat(A,I2)
augmat3= op.aug_mat(A,I3)
augmat4= op.aug_mat(A,I4)
augmat5= op.aug_mat(A,I5)
augmat6= op.aug_mat(A,I6)

X3_1 = op.GaussJordan(augmat1)
X3_2 = op.GaussJordan(augmat2)
X3_3 = op.GaussJordan(augmat3)
X3_4 = op.GaussJordan(augmat4)
X3_5 = op.GaussJordan(augmat5)
X3_6 = op.GaussJordan(augmat6)


X =[X3_1[6],X3_2[6],X3_3[6],X3_4[6],X3_5[6],X3_6[6]]



print('Inverse using Gauss Jordan :')
print(np.array(X))
X1_1=op.Gauss_Seidel(A,I1,x=None,tol=1e-5)
X1_2=op.Gauss_Seidel(A,I2,x=None,tol=1e-5)
X1_3=op.Gauss_Seidel(A,I3,x=None,tol=1e-5)
X1_4=op.Gauss_Seidel(A,I4,x=None,tol=1e-5)
X1_5=op.Gauss_Seidel(A,I5,x=None,tol=1e-5)
X1_6=op.Gauss_Seidel(A,I6,x=None,tol=1e-5)

Y=[X1_1,X1_2,X1_3,X1_4,X1_5,X1_6]

X2_1=op.Jacobi_Inv(A,I1,x=None,tol=1e-5)
X2_2=op.Jacobi_Inv(A,I2,x=None,tol=1e-5)
X2_3=op.Jacobi_Inv(A,I3,x=None,tol=1e-5)
X2_4=op.Jacobi_Inv(A,I4,x=None,tol=1e-5)
X2_5=op.Jacobi_Inv(A,I5,x=None,tol=1e-5)
X2_6=op.Jacobi_Inv(A,I6,x=None,tol=1e-5)

Z=[X2_1,X2_2,X2_3,X2_4,X2_5,X2_6]


print("Matrix Inverse by Gauss Seidal:")
print(np.array(Y))
print("Matrix inverse using jacobi Inverse:")
print(np.array(Z))

'''

Inverse using Gauss Jordan :
[[0.93506494 0.87012987 0.25974026 0.20779221 0.41558442 0.16883117]
 [0.29004329 0.58008658 0.17316017 0.13852814 0.27705628 0.11255411]
 [0.08658009 0.17316017 0.32034632 0.05627706 0.11255411 0.10822511]
 [0.20779221 0.41558442 0.16883117 0.93506494 0.87012987 0.25974026]
 [0.13852814 0.27705628 0.11255411 0.29004329 0.58008658 0.17316017]
 [0.05627706 0.11255411 0.10822511 0.08658009 0.17316017 0.32034632]]

Matrix Inverse by Gauss Seidal:
[[0.93114123 0.28829187 0.08594961 0.20458171 0.13709504 0.05576116]
 [0.86661184 0.57851622 0.17259487 0.41270575 0.27577131 0.11209154]
 [0.25552117 0.17127691 0.31966838 0.16537899 0.11101313 0.10767038]
 [0.20458171 0.13709504 0.05576116 0.93243781 0.28887061 0.08615794]
 [0.41128511 0.27513716 0.11186326 0.86661184 0.57851622 0.17259487]
 [0.16537899 0.11101313 0.10767038 0.25691536 0.17189922 0.3198924 ]]
 
Matrix inverse using jacobi Inverse:
[[0.92920462 0.28742748 0.08517372 0.2029974  0.1353316  0.05512633]
 [0.86487561 0.57658374 0.17189922 0.40916341 0.27419007 0.11101313]
 [0.25343874 0.17034745 0.31883408 0.16367544 0.10911696 0.10698776]
 [0.2029974  0.1353316  0.05512633 0.92920462 0.28742748 0.08517372]
 [0.40916341 0.27419007 0.11101313 0.86487561 0.57658374 0.17189922]
 [0.16367544 0.10911696 0.10698776 0.25343874 0.17034745 0.31883408]]
'''
