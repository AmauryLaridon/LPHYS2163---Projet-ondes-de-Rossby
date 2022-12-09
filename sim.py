import matplotlib.pyplot as plt
import numpy as np


###################################################### Paramètres de la simulation ##################################################

phi_0 = ((2*np.pi)/360)*45  # latitude en radian
Delta_s = 200000  # résolution de la maille spatiale en mètre
Delta_t = 1  # résolution de la maille temporelle en seconde
T = 360  # temps total de simulation en seconde
L_x = 12000000  # longueur du domaine selon x en mètre
L_y = 6000000  # longueur du domaine selon y en mètre
M = int(L_x/Delta_s)  # nombre d'itération selon x
N = int(L_y/Delta_s)  # nombre d'itération selon x
W_x = 6000000  # longueurs d'onde champ initial selon x en mètre
W_y = 3000000  # longueurs d'onde champ initial selon y en mètre
Omega = 7.292115*10**(-5)  # vitesse angulaire de la rotation de la Terre
g = 9.81  # norme de l'accélaration gravitationnelle
a = 6371000  # rayon moyen de la Terre en km

###################################################### Discrétisation des mailles ##################################################

x = np.arange(0, L_x, Delta_s)  # discrétisation de l'axe horizontal
y = np.arange(0, L_y, Delta_s)  # discrétisation de l'axe vertical
x_grid = np.matrix(x,)
y_grid = np.matrix(y,)
t_grid = np.arange(0, T, Delta_t)

###################################################### Fonctions de la simulation ##################################################


# Fonctions de la simulation

def f():
    """Donne la valeur du paramètre de Coriolis dans l'approximation du plan beta en fonction de la position"""
    def f_0():
        """Donne la valeur du paramètre de Coriolis f_0"""
        F_0 = np.full((N, M), 2*Omega*np.sin(phi_0))
        return F_0

    def beta():
        """Donne la valeur du paramètre beta"""
        return (2*Omega*np.cos(phi_0)/a)

    def y():
        """Donne la composante verticale y pour chaque point de la maille"""
        y_axis = np.arange(-(N/2)*Delta_s, (N/2)*Delta_s, Delta_s)
        y_label = np.zeros((N, M))
        Y = np.full((N, M), 1)
        for i in range(N):
            for j in range(M):
                y_label[i, j] = Y[i, j]*y_axis[i]*(-1)
        return y_label

    f = np.zeros((N, M))
    # print(beta())
    # print(f_0()[9, 5])
    # print(y())
    #print("f_0 = ", f_0())
    #print("beta fois y = ", beta()*y())
    #print("f =", f_0() + beta()*y())
    f = f_0()+beta()*y()
    # print(f[-1])
    return f


'''
def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/W_x
    j = (2*np.pi)/W_y
    psi_0_T = np.zeros((N, M))
    psi_0_T = (g/f_0(Omega, phi_0))*(100*np.sin(k*(x_grid.T))@(np.cos(j*y_grid)))
    psi_0 = psi_0_T.T
    return psi_0


def vort_init():
    """Donne la composante verticale de la vorticité relative en fonction de la fonction de courant initiale à l'aide de l'équation(2)"""
    vort = np.zeros((N, M))
    psi_0 = psi_init()
    for i in range(N):
        # print("i = ", i)
        for j in range(M):
            # print("j = ", j)
            if i == 0:
                # conditions aux bords périodiques : le Nord du domaine = le Sud du domaine
                vort[0, :] = vort[-1, :]
            elif j == 0:
                # conditions aux bords périodiques : l'Ouest du domaine = l'Est du domaine
                vort[:, 0] = vort[:, -1]
            elif i == (N-1):
                # conditions aux bords périodiques : le Sud du domaine = le Nord du domaine
                vort[N-1, :] = vort[0, :]
            elif j == (M-1):
                # conditions aux bords périodiques : l'Est du domaine = l'Ouest du domaine
                vort[:, M-1] = vort[:, 0]
            else:
                vort[i, j] = (1/(Delta_s**2))*(-4*psi_0[i, j]+psi_0[i, j+1] +
                                               psi_0[i, j-1]+psi_0[i+1, j]+psi_0[i-1, j])
                # print("vort({},{}) =".format(i, j), vort[i, j])
    return vort


def velocity_field():
    """Donne champ de vitesse à partir de la fonction de courant"""
    psi_0 = psi_init()
    u = np.zeros((N, M))
    v = np.zeros((N, M))
    for i in range(N):
        # print("i = ", i)
        for j in range(M):
            # print("j = ", j)
            if i == (N-1):
                # conditions aux bords périodiques : le Sud du domaine = le Nord du domaine
                u[N-1, :] = u[0, :]
                v[N-1, :] = v[0, :]
            elif j == (M-1):
                # conditions aux bords périodiques : l'Est du domaine = l'Ouest du domaine
                u[:, M-1] = u[:, 0]
                v[:, M-1] = v[:, 0]
            else:
                u[i, j] = (-1/(2*Delta_s))*(psi_0[i+1, j]-psi_0[i-1, j])
                v[i, j] = (1/(2*Delta_s))*(psi_0[i, j+1]-psi_0[i, j-1])
    return [u, v]




def vort_flux([u, v], zeta beta):
    """Donne la valeur du flux de la composante verticale de la vorticité relative. Prend comme argument le champ de vitesse"""
    F = np.zeros((N, M))
    for i in range(N):
        # print("i = ", i)
        for j in range(M):
            # print("j = ", j)
            if i == (N-1):
                # conditions aux bords périodiques : le Sud du domaine = le Nord du domaine
                F[N-1, :] = F[0, :]
            elif j == (M-1):
                # conditions aux bords périodiques : l'Est du domaine = l'Ouest du domaine
                F[:, M-1] = F[:, 0]
            else:
                F[i, j] = (-1/(2*Delta_s))*(zeta[i+1, j]-zeta[i-1, j])
    return F



###################################################### Résultats et figures ############################################################


# Output de la résolution des mailles
print("\n\nRésolution numérique avec une grille spatiale de {} points".format(np.shape(x_grid)[1]))
print("Résolution numérique avec une grille temporelle de {} points".format(len(t_grid)))

# Contourplot de la fonction de courant initiale
'''
X, Y = np.meshgrid(x, y)

'''
plt.contourf(X, Y, psi_init(), 100)
plt.colorbar()
plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=24)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.show()

# Contourplot de la composante verticale de la vorticité relative initiale
plt.contourf(X, Y, vort_init(), 100)
plt.colorbar()
plt.title("Contour plot de $\zeta_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=24)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.show()


# Plot du champ de vitesse initial

U = velocity_field()[0]
V = velocity_field()[1]
zero = np.zeros((N, M))
slice_interval = 1  # Slicer index for smoother quiver function
skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
vels = np.hypot(U, V)  # Velocity norm of each velocity vector
Quiver = plt.quiver(X[skip], Y[skip], U[skip], V[skip], vels[skip],
                    units='height', angles='xy')
plt.colorbar(Quiver)
# plt.quiverkey(Quiver, 1.01, 1.01, 30000, label="?m/s",
# labelcolor = 'blue', labelpos = 'N', coordinates = "axes")
plt.title("Champ de vitesse initial $(u_0(x,y),v_0(x,y))$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=24)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.show()
'''
# Plot du champ du paramètre de Coriolis

F = f()
plt.contourf(X, -Y, F, 100)
plt.colorbar()
plt.title("Valeur du paramètre de Coriolis f dans l'approximation du plan \beta \n f = f_0 + \beta y", fontsize=24)
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.grid()
plt.show()
