# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math
import time



# Q1 (a)
print("Q1 (a)")
def LGM_uniform(seed, n):
     m = pow(2,31)-1
     b = 0
     a = pow(7,5)
     X = np.zeros(n)
     U = np.zeros(n)
     X[0] = seed
     for i in range(1,n):
         X[i] = np.mod(X[i-1]*a+b,m)
     for i in range(0,n):
         U[i] = X[i]/m
     return U

U1a = LGM_uniform(101,10000)
plt.hist(U1a, bins=30)
plt.title("Uniform_LGM Histogram")
plt.xlabel("$U1a_i$")
plt.ylabel("Frequency")
print("The mean and std for LGM_Uniform is:",round(np.mean(U1a),4),round(np.std(U1a),4))

plt.show()

# Q1 (b)
print("Q1 (b)")
np.random.seed(5)
U1b = np.random.uniform(size=10000)
plt.hist(U1b, bins=30, color='yellow')
plt.title("Uniform_Built in Function")
plt.xlabel("$U1b_i$")
plt.ylabel("Frequency")
print("The mean and std for Builtin_Uniform is:",round(np.mean(U1b),4),round(np.std(U1b),4))

plt.show()

# Q1 (c)
print("Q1 (c)")
print("the histogram of two look very similar,means and std are also very similar,means of two distributions are separately:")
print(round(np.mean(U1a),4),round(np.mean(U1b),4))
print("and std of two distributions are separately:")
print(round(np.std(U1a),4),round(np.std(U1b),4))

# Q2 (a)
print("Q2 (a)")
U2 = LGM_uniform(101,10000)
X2 = np.zeros(10000)
for i in range(0,10000):
    if U2[i] < 0.3:
        X2[i] = -1
    if U2[i] >= 0.3 and U2[i] < 0.65:
        X2[i] = 0
    if U2[i] >= 0.65 and U2[i] < 0.85:
        X2[i] = 1
    if U2[i] >= 0.85:
        X2[i] = 2
print("X2 is,",X2)

# Q2 (b)
print("Q2 (b)")
plt.hist(X2, color='grey')
plt.title("Histogram of $X_2$")
plt.xlabel("$X_2$")
plt.ylabel("Frequency")
plt.show()

print("The mean and standard deviation are separately",round(np.mean(X2),4),round(np.std(X2),4))

# Q3 (a)
print("Q3 (a)")
# generate 44,000 uniform (0,1)
U3 = LGM_uniform(2,44000)

# generate 44,000 Bernoulli random numbers
X3 = np.zeros(44000)
for i in range(0,44000):
    if U3[i] < 0.64:
        X3[i] = 1
    if U3[i] >= 0.64:
        X3[i] = 0

# Define Binomial variable Y3[i]=sum of sublist
split= list(range(0, 44000, 44))
sub=[X3[i: i + 44] for i in split]
Y3=np.zeros(1000)
for i in range(1,1000):
    Y3[i]=sum(sub[i])
print(Y3)

# Q3 (b)
print("Q3 (b)")

# plot the histogram
plt.hist(Y3, bins=100, color='pink')
plt.title("Histogram of Binomial B(44,0.64) $Y_3$")
plt.xlim(15,40)
plt.xlabel("$Y_3$")
plt.ylabel("Frequency")
plt.show()

# compute the probability P(Y3>=40)
m=1000
m1=0
for i in range(1,1000):
    if Y3[i]>40 or Y3[i]==40:
        m1=m1+1
print(m1)
prob=m1/1000.0
print ("The empirical probability that the random variable Y3 is at least 40 is",prob)

# Q4 (a)
print("Q4 (a)")
#np.random.exponential(1.5,10000)
lamb=1.5
Y4=np.zeros(10000)
Y4=-(1/lamb)*np.log(U1a)
print(Y4)


# Q4 (b)
print("Q4 (b)")
prob1=np.size(np.where(Y4 >= 1))/10000.0
prob2=np.size(np.where(Y4 >= 4))/10000.0
print ("The probability that the random variable Y4 is at least 1 is",prob1)
print ("The probability that the random variable Y4 is at least 4 is",prob2)

# Q4 (c)
print("Q4 (c)")
# plot the histogram
plt.hist(Y4, bins=30, color='orange')
plt.title("Histogram of Exponential $Y_4$")
plt.xlabel("$Y_4$")
plt.ylabel("Frequency")
plt.show()

# caculate the mean and standard deviation 
print("The mean and standard deviation are separately",round(np.mean(Y4),4),round(np.std(Y4),4))


# Q5 (a)
print("Q5 (a)")
# generate 5000 uniformly distributed random numbers
U51=LGM_uniform(101,5000)
print(U51)

# Q5 (b)
print("Q5 (b)")
# generate normal dist random numbers
U52=LGM_uniform(123,5000)
Z51=np.zeros(5000)
start_time = time.time()
for i in range(1,5000):
    Z51[i]=np.sqrt(-2*np.log(U51[i]))*math.cos(2*math.pi*U52[i])
print(Z51)
t1=time.time() - start_time

# plot the histogram
plt.hist(Z51,bins=30,color="red")
plt.title("Histogram of Box-Muller Normal $Z_51$")
plt.xlabel("$Z_51$")
plt.ylabel("Frequency")
plt.show()

# Q5 (c)
print("Q5 (c)")
# caculate the mean and variance
print("The mean and standard deviation are separately",round(np.mean(Z51),4),round(np.std(Z51),4))

# Q5 (d)
print("Q5 (d)")
W=np.zeros(5000)
V1=np.zeros(5000)
V2=np.zeros(5000)
Z52=np.zeros(5000)
V1[:]=2*U51[:]-1
V2[:]=2*U52[:]-1
W[:]=pow(V1[:],2)+pow(V2[:],2)
j=0
start_time = time.time()
for i in range(0,5000):
    if W[i]<=1:
        Z52[j]=V1[i]*np.sqrt(-2*np.log(W[i])/(W[i]))
        j=j+1
Z52=Z52[0:j]
t2=time.time() - start_time

# plot the histogram
plt.hist(Z52, bins=30,color="orange")
plt.title("Histogram of  Polar-Marsaglia Normal $Z_52$")
plt.xlabel("$Z_52$")
plt.ylabel("Frequency")
plt.show()

# Q5 (e)
print("Q5 (e)")
# caculate the mean and variance
print("The mean and standard deviation are separately",round(np.mean(Z52),4),round(np.std(Z52),4))

# Q5 (f)
print("Q5 (f)")
print("By using python's build-in function:")
print("The effiency for Box-Muller is:", round(t1,5), "s.")
print("The effiency for Polar-Marsaglia is:", round(t2,5), "s.")
print("Polar-Marsaglia method is more efficient based on the result presented above, because trigonometric function is time consuming than polynomial function.")
























