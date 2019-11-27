# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy.stats import norm 
import random as random

# Q1

#find x and y with correlation -0.7, and variance 1, and caculate rho
def bi_var(seed):
    n=1000
    np.random.seed([seed])
    # first simulate 2 independent random normals
    # then get x and y
    Z1=np.random.normal(0,1,n)
    Z2=np.random.normal(0,1,n)
    corr=-0.7
    mu1=mu2=0
    x=mu1+Z1
    y=mu2+corr*Z1+np.sqrt(1-corr*corr)*Z2
    #to get the rho
    sd_x=np.std(x,ddof=1)
    sd_y=np.std(y,ddof=1)
    nomi=(1.0/(n-1.0))*np.dot((x-np.mean(x)),(y-np.mean(y)))
    rho=nomi/(sd_x*sd_y)
    return (rho)

# Q2
 
# Expected Values by Monte Carlo Simulation 
def mc_val(seed):
    n=10000
    np.random.seed([seed])   
    # first simulate 2 independent random normal
    corr=0.6
    mu1=mu2=0
    z1=np.random.normal(0,1,n)
    z2=np.random.normal(0,1,n)
    X=mu1+z1
    Y=mu2+corr*z1+np.sqrt(1-corr*corr)*z2
    E=np.zeros(n)
    for i in range(n):
        E[i]=max(0,(pow(X[i],3)+math.sin(Y[i])+(pow(X[i],2))*Y[i]))
    return (np.mean(E))


# Q3 (a)

# standard Wiener Process follows normal distribution of N(0,sqrt(t))
n=10000 
z1=np.random.normal(0,1,n)
# define Wiener Process
# W5 simulation
def w(t):
    w=np.sqrt(t)*z1
    return w
w5=w(5)
F1=np.zeros(n)
for i in range(n):
    F1[i]=w5[i]*w5[i]+math.sin(w5[i])
print("mean(Ea1) and variance are separatly:")
print (round(np.mean(F1),4),round(np.var(F1),8))

# Wt simulation
def F2(t):
    w=np.zeros(n)
    F2=np.zeros(n)
    for i in range(n):
        w[i]=np.sqrt(t)*z1[i]
        F2[i]=np.exp(t/2)*math.cos(w[i])
    return(F2)
print ("mean for t=0.5(Ea2),3.2(Ea3) and 6.5(Ea4) are separately:")
print(round(np.mean(F2(0.5)),4),round(np.mean(F2(3.2)),4),round(np.mean(F2(6.5)),4))
print ("variance for t=0.5,3.2 and 6.5 are separately:")
print(round(np.var(F2(0.5)),4),round(np.var(F2(3.2)),4),round(np.var(F2(6.5)),4))


# Q3 (b)

# Q3 (c)
# use control variate method to reduce variance
Y1=Y2=Y3=Y4=np.zeros(n)
T1=np.zeros(n)
for i in range(n):
    Y1[i]=w5[i]*w5[i]
gamma1=np.cov(Y1,F1)[1,0]/np.var(Y1)
T1=F1-gamma1*(Y1-np.mean(Y1))
print("mean(Eb1) and variance after variance reduction are separatly:")
print (round(np.mean(T1),4),round(np.var(T1),8))

T2=T3=T4=np.zeros(n)
for i in range(n):
    Y2[i]=w(0.5)[i]*w(0.5)[i]
    Y3[i]=w(3.2)[i]*w(3.2)[i]
    Y4[i]=w(6.5)[i]*w(6.5)[i]
gamma2=np.cov(Y2,F2(0.5))[1,0]/np.var(Y2)
gamma3=np.cov(Y3,F2(3.2))[1,0]/np.var(Y3)
gamma4=np.cov(Y4,(F2(6.5)))[1,0]/np.var(Y4)
T2=F2(0.5)-gamma2*(Y2-np.mean(Y2))
T3=F2(3.2)-gamma3*(Y3-np.mean(Y3))
T4=F2(6.5)-gamma4*(Y4-np.mean(Y4))
print("means for t=0.5(Eb2),3.2(Eb3) and 6.5(Eb4) are separately:")
print(round(np.mean(T2),4),round(np.mean(T3),4),round(np.mean(T4),4))
print ("variance for t=0.5,3.2 and 6.5 are separately:")
print(round(np.var(T2),4),round(np.var(T3),4),round(np.var(T4),4))


# Q4 (a)
# simulate the stock process 
T = 5.0
S0 = 88.0
r = 0.04
sd = 0.2
K = 100.0
S_p=S0*np.exp(sd*w(T)+(r-(sd*sd)/2)*T)
C=np.zeros(10000)
for i in range(10000):
    if S_p[i]>K:
        C[i]=S_p[i]-K
    else:
        C[i]=0
print("The price of call option is:")       
print(round(np.exp(-r*T)*np.mean(C),4))
print("The variance of Ca1 is:")
print(round(np.var(np.exp(-r*T)*C),4))


# Q4(b)
# with Black-Sholes Formula
def euro_bs_call(S,K,T,r,sd):
    d1 = (np.log(S/K) + (r + 0.5 * sd ** 2) * T) / (sd * np.sqrt(T))
    d2 = d1 - sd*np.sqrt(T)
    Cb1 = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return(Cb1)
print("The call option price with Black-Sholes Formula Cb1 is:")
print(round(euro_bs_call(88.0,100.0,5.0,0.04,0.2),4))

# with variance reduction method 
# using antithetic variates method 
S_n=S0*np.exp(sd*w(T)*(-1)+(r-(sd*sd)/2)*T)
C_n=np.zeros(10000)
for i in range(10000):
    if S_n[i]>K:
        C_n[i]=S_n[i]-K
    else:
        C_n[i]=0
Cb2=np.exp(-r*T)*(0.5*np.mean(C_n)+0.5*np.mean(C))
print("The price of call option Cb2 is:")       
print(round(Cb2,4))
print("The variance of Cb2 is:")
print(round(np.var(np.exp(-r*T)*(0.5*C_n+0.5*C)),4))


# Q5 (a)
# simulate the stock process 
n_5 = 1000
S0_5 = 88.0
sd_5 = 0.18
r_5 = 0.04
ES=np.zeros(10)
for i in range (1,11):
    S_t=np.zeros(n)
    S_t= S0_5*np.exp(sd_5*w(i)+(r_5-(sd_5*sd_5)/2)*i) 
    ES[i-1]=np.mean(S_t)

plt.plot(ES)
plt.title('$E[S]$, for n ranging from 1 to 10')
plt.xlabel('$n$')
plt.ylabel('$E[S]$')
plt.show()

# Q5 (b)
n5=1000
dt = np.sqrt(10.0/n5)
path = np.zeros((6,n5+1))
wdt = np.zeros((10,n5))

plt.figure
for i in range(6):
    Zi = np.random.normal(0,1,n5)
    wdt[i,0] = dt*Zi[0]
    for j in range(1,n5):
        wdt[i,j] = wdt[i,j-1] + dt*np.array(Zi[j])
    path[i,1:(n+1)] = S0*np.exp(sd_5*wdt[i,:]+(r-(sd_5**2/2))*(i+1))
    path[i,0] = S0
    plt.plot(np.arange(0,10,10.0/1001),path[i,:])
plt.plot(ES)   
plt.title("$E[S]$ & 6 paths of Stock Prices and Their Expected Value")
plt.xlabel("$t$")
plt.ylabel("$Stock Paths S_t$")
plt.show()    
    
# Q5 (c)
print("The graph is plotted in Q5(b)")

# Q5 (d)
# under sd=35%
sd_52=0.35
ESd=np.zeros(10)
for i in range (1,11):
    S_t=np.zeros(n5)
    S_t= S0_5*np.exp(sd_52*w(i)+(r_5-(sd_52*sd_52)/2)*i) 
    ESd[i-1]=np.mean(S_t)

path2 = np.zeros((6,n5+1))
plt.figure
for i in range(6):
    Zi = np.random.normal(0,1,n5)
    wdt[i,0] = dt*Zi[0]
    for j in range(1,n5):
        wdt[i,j] = wdt[i,j-1] + dt*np.array(Zi[j])
    path2[i,1:(n5+1)] = S0*np.exp(sd_52*wdt[i,:]+(r-(sd_52**2/2))*(i+1))
    path2[i,0] = S0
    plt.plot(np.arange(0,10,10.0/1001),path2[i,:])
plt.plot(ESd)   
plt.title("$E[Sd]$ & 6 paths of Stock Prices and Their Expected Value")
plt.xlabel("$t$")
plt.ylabel("$Stock Paths S_t$")
plt.show()

print("The graph for $E[S_n]$ didn't change at all. ")
print("But with a higher std,6 paths are much more volatile")

# Q6 (a)
n6=1001
x6=np.linspace(0,1,n6)
delta=1.0/(n6-1)
y6=0
for i in range(1,n6):
   y6+=np.sqrt(1-x6[i]*x6[i])*delta*4
print("By using Euler's Method,the integration is:" )
print(round(y6,4))

# Q6 (b)
x6b=np.random.uniform(0,1,n6)
y6b=np.zeros(n6)
for i in range(1,n6):
    y6b[i]=np.sqrt(1-x6b[i]*x6b[i])*4
print("By using Monte Carlo Simulation,the integration is:" )
print(round(np.mean(y6b),4)) 

# Q6 (c)
# define h(x)
n6c=10000
random.seed(123)
y=[]
u=np.random.uniform(0,1,n6c)

random.seed(121)
u2=np.random.uniform(0,1,n6c)

h = (1-0.74*(np.array(u)**2))/(1-0.74/3.0)

for i in range(1000):
    if u2[i] <= h[i]/1.5:
       y.append(u[i])
       
h_y= (1-0.74*(np.array(y)**2))/(1-0.74/3.0)
g_y= np.sqrt(1-np.array(y)**2)
sim_i=4*g_y*1/h_y
print("By using Importance Sampling method ,the integration is:" )
print(round(np.mean(sim_i),4))


















