# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import numpy as np

Nx=25; # size along x
Ny=25; # size along y
radius=0.35;# radius of central lead
Niter=1500; # number of iterations to perform

potential=np.zeros((Ny,Nx))
x=np.linspace(-12,12,25)
y=np.linspace(-12,12,25)
Y,X=np.meshgrid(y,x)
ii=np.where(X*X+Y*Y<(0.35*25)**2)
potential[ii]=1.0
'''
levels = np.linspace(15,165,10) # your colorbar range, starting at 15
cmap = plt.get_cmap()
cmap.set_under(color='white')
plt.contourf(counts.transpose(), levels=levels, cmap=cmap, extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()])
'''

# All bims under 15 are plotted white
plt.figure(0)
plt.contourf(X,Y,potential)
xn,yn=np.where(X*X+Y*Y<(0.35*25)**2)
plt.plot(xn-12,yn-12,'ro')
plt.title('Contour Plot of Potential')
plt.show()

errors=np.zeros(Niter)

for k in range(Niter):
    oldphi=potential.copy()
    potential[1:-1,1:-1]=0.25*(oldphi[1:-1,0:-2]+ oldphi[1:-1,2:]+ oldphi[0:-2,1:-1] + oldphi[2:,1:-1])
    potential[1:-1,0]=potential[1:-1,1] # Left Boundary
    potential[1:-1,Nx-1]=potential[1:-1,Nx-2] # Right Boundary
    potential[0,1:-1]=potential[1,1:-1] # Top Boundary
    potential[Ny-1,1:-1]=0
    potential[xn,yn]=1.0
    errors[k]=(np.abs(potential-oldphi)).max();

plt.figure(1)
plt.grid(True)
plt.title(r'Semilogy Plot of Error values')
plt.xlabel('Number of Iterations')
plt.ylabel('Maximum Change')
plt.semilogy(np.arange(len(errors))+1,errors)
plt.show()

plt.figure(2)
plt.grid(True)
plt.title(r'Loglog Plot of Error values')
plt.xlabel('Number of Iterations')
plt.ylabel('Maximum Change')
plt.loglog(np.arange(len(errors))+1,errors,label='Every Point Error')
plt.loglog((np.arange(len(errors))+1)[::50],errors[::50],'ro',label=r'Every 50_th point Error')
plt.legend()
plt.show()

def soln(x,y):
    logy=np.log(y)
    vect=np.zeros((len(x),2))
    vect[:,0]=1
    vect[:,1]=x
    logA,B=np.linalg.lstsq(vect, logy.transpose())[0]
    return (np.exp(logA),B)

A1,B1=soln(np.arange(Niter)+1,errors)
A2,B2=soln((np.arange(Niter)+1)[500:],errors[500:])

plt.figure(3)
plt.grid(True)
plt.title("Original vs Fitted")
plt.loglog(np.arange(len(errors))+1,errors,label='Every Point Error')
plt.loglog((np.arange(len(errors))+1)[::50],A1*np.exp(B1*(np.arange(len(errors))+1))[::50],'go',label='Fit for all')
plt.loglog((np.arange(len(errors))+1)[::50],A2*np.exp(B2*(np.arange(len(errors))+1))[::50],'ro',label='Fit for all after 500 iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Maximum Change')
plt.legend()
plt.show()

fig1=plt.figure(4)
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
plt.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, potential .T, rstride=1, cstride=1, cmap=plt.cm.jet)
plt.show()

Jx,Jy=(1/2*(potential[1:-1,0:-2]-potential[1:-1,2:]),1/2*(potential[:-2,1:-1]-potential[2:,1:-1]))

plt.figure(5)
plt.contourf(Y,X[::-1],potential,200,cmap = plt.cm.jet)
plt.title('Contour Plot of Potential')
plt.show()

plt.figure(6)
plt.quiver(Y[1:-1,1:-1],-X[1:-1,1:-1],-Jx[:,::-1],-Jy)
xn,yn=np.where(X*X+Y*Y<(0.35*25)**2)
plt.plot(xn-12,yn-12,'ro')
plt.title('Vector Plot of current Flow')
plt.show()