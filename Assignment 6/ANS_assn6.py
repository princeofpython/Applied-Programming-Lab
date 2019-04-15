#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 03:42:46 2019

@author: adithya
"""

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

#Question 5

H=sp.lti(1,np.poly1d([1e-12,1e-4,1]))
w,S,phi=H.bode()
plt.figure(0)
plt.subplot(211)
plt.semilogx(w,S)
plt.xlabel('$w$')
plt.ylabel('$magnitude$')
plt.title("Magnitude plot")
plt.subplot(212)
plt.semilogx(w,phi)
plt.xlabel('$w$')
plt.ylabel('$phase$')
plt.title("Phase plot")
plt.show()

# Question 6
t=np.linspace(0,30e-6,1000)
tn=np.linspace(0,30e-3,10000)
f=np.cos(1e3*t)-np.cos(1e6*t)
fn=np.cos(1e3*tn)-np.cos(1e6*tn)
t,y,svec=sp.lsim(H,f,t)
tn,yn,svec=sp.lsim(H,fn,tn)

plt.figure(1)
plt.xlabel('$t$')
plt.ylabel('$Output Voltage$')
plt.title("Short Term Response")
plt.plot(t,y)
plt.show()

plt.figure(2)
plt.xlabel('$t$')
plt.ylabel('$Output Voltage$')
plt.title("Long Term Response")
plt.plot(tn,yn)
plt.show()

#Question 1
t=np.linspace(0,50,2000) 

p1=np.polymul([1,1,2.5],[1,0,2.25])
X1=sp.lti([1,0.5],p1)

t,x1=sp.impulse(X1,None,t)

plt.figure(3)
plt.plot(t,x1)
plt.title("Spring with 0.5 decay factor")
plt.xlabel('$time$')
plt.ylabel('$x(t)$')
plt.show()

#Question 2
p2=np.polymul([1,0.1,2.2525],[1,0,2.25])
X2=sp.lti([1,0.05],p2)

t,x2=sp.impulse(X2,None,t)

plt.figure(4)
plt.plot(t,x2)
plt.title("Spring with 0.05 decay factor")
plt.xlabel('$time$')
plt.ylabel('$x(t)$')
plt.show()

#Question 3
H=sp.lti([1],[1,0,2.25])
t,h=sp.impulse(H,None,t)

for i in range(5):
    f=1.4+i*0.05
    p=np.polymul([1,0.1,0.0025+f*f],[1,0,2.25])
    X=sp.lti([1,0.05],p)
    t,xx=sp.impulse(X,None,t)
    plt.figure(i+5)
    plt.plot(t,xx)
    plt.title("Spring with 0.05 decay factor and frequency "+str(f))
    plt.xlabel('$time$')
    plt.ylabel('$x(t)$')
    plt.show()
    
#Question 4
X=sp.lti([1,0,2],[1,0,3,0])
Y=sp.lti([2],[1,0,3,0])

t=np.linspace(0,20,1000)

t,x=sp.impulse(X,None,t)
t,y=sp.impulse(Y,None,t)
plt.figure(10)
plt.plot(t,x)
plt.xlabel('$time$')
plt.ylabel('$x(t)$')
plt.show()

plt.figure(11)
plt.plot(t,y)
plt.xlabel('$time$')
plt.ylabel('$y(t)$')
plt.show()