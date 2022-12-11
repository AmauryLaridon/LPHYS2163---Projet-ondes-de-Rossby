import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Variables------------------------------------------------------------------
L_x=12000000
L_y=6000000
t_max=3600
ds=200000
dt=1
M=int(np.floor(L_x/ds)+1)
N=int(np.floor(L_y/ds)+1)
x=np.linspace(0, L_x, M)
y=np.linspace(0, L_y, N)
x_mat=np.array([x])
y_mat=np.array([y])
g=9.81
W_x=6000000
W_y=3000000
phi=((2*np.pi)/360)*45
Omega=7.27*10**(-5)



# Functions------------------------------------------------------------------
def f_0(Omega, phi):
    return(2*Omega*np.sin(phi))

def psi_in():     
    k=(2*np.pi)/W_x
    j=(2*np.pi)/W_y
    psi_0=np.zeros((M,N))
    psi_0=(g/f_0(Omega, phi)*(100*np.sin(k*x_mat.T)@np.cos(j*y_mat)))
    return (psi_0)

def zeta_in():
    zeta_0=np.zeros((M,N))
    zeta_0[:,N-1]=zeta_0[:,0]
    zeta_0[M-1,:]=zeta_0[0,:]
    psi_0=psi_in()
    for i in range (M-1):
        for j in range (N-1) :
            zeta_0[i,j]=(1/ds**2)*(-4*psi_0[i,j]+psi_0[i+1,j]+psi_0[i-1,j]+psi_0[i,j+1]+psi_0[i,j-1])
    return (zeta_0)

def v_in():
    psi_0=psi_in()
    u_0=np.zeros((M,N))
    v_0=np.zeros((M,N))
    u_0[:,N-1]=u_0[:,0]
    u_0[M-1,:]=u_0[0,:]
    v_0[:,N-1]=v_0[:,0]
    v_0[M-1,:]=v_0[0,:]
    for i in range (M-1):
        for j in range (N-1):
            u_0[i,j]=(1/(2*ds))*(-psi_0[i,j+1]+psi_0[i,j-1])
            v_0[i,j]=(1/(2*ds))*(psi_0[i+1,j]-psi_0[i-1,j])
    return(u_0,v_0)

# Plots------------------------------------------------------------------
[xx,yy]=np.meshgrid(x,y)
psi=plt.contourf(xx,yy,psi_in().T,100, cmap=cm.viridis)
plt.title('Graphe de la fonction de courant initiale $\psi_0(x,y)$', fontsize=11)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(psi)
plt.tight_layout()
plt.show()


zeta=plt.contourf(xx,yy,zeta_in().T,100, cmap=cm.viridis)
plt.title('Graphe de la vorticité initiale $\zeta_0(x,y)$', fontsize=11)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(zeta)
plt.tight_layout()
plt.show()

U=v_in()[0].T
V=v_in()[1].T
v = np.hypot(U, V)
vel=plt.quiver(xx,yy,U,V,v)
plt.title('Graphe de la vitesse initiale $u_0(x,y)$', fontsize=11)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(vel)
plt.show()
