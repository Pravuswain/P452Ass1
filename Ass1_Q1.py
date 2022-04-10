import Matrix_operation as op


with open('Ass1mat1A.txt','r') as f:
	a =[[float(num) for num in line.split(',')] for line in f]
with open('Ass1mat1B.txt','r') as f:
	b=[float(num) for num  in f]


#creating Augmented matrix using Augmentation function
X = op.aug_mat(a,b)
#solving by Gauss Jordan method
Sol = op.GaussJordan(X)
print("Solutions using Gauss Jordan :")
print(f"a1= {Sol[6][0]},a2={Sol[6][1]},a3={Sol[6][2]},a4={Sol[6][3]},a5={Sol[6][4]},a6={Sol[6][5]}")
'''
Solutions using Gauss Jordan :
a1= 1.1580753585739938,a2=-3.9985768070008967,a3=-0.059642889586468684,a4=2.298941184782938,a5=-1.0269801055787147,a6=3.929122969931263
'''