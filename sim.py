import matplotlib.pyplot as plt
import numpy as np


###################################################### Paramètres de la simulation ##################################################

phi_0 = ((2*np.pi)/360)*45  # latitude en radian
Delta_s = 200000  # résolution de la maille spatiale en mètre
Delta_t = 3600  # résolution de la maille temporelle en seconde
nbr_jour = 1  # nombre de jours de simulation
T = 86400*nbr_jour  # temps total de simulation en seconde
L_x = 12000000  # longueur du domaine selon x en mètre
L_y = 6000000  # longueur du domaine selon y en mètre
M = int(L_x/Delta_s)  # nombre d'itération selon x
N = int(L_y/Delta_s)  # nombre d'itération selon y
K = int(T/Delta_t)  # nombre d'itération selon t
W_x = 6000000  # longueurs d'onde champ initial selon x en mètre
W_y = 3000000  # longueurs d'onde champ initial selon y en mètre
Omega = 7.292115*10**(-5)  # vitesse angulaire de la rotation de la Terre
g = 9.81  # norme de l'accélaration gravitationnelle
a = 6371000  # rayon moyen de la Terre en mètres

###################################################### Discrétisation des mailles ##################################################

x = np.arange(0, L_x, Delta_s)  # discrétisation de l'axe horizontal
y = np.arange(0, L_y, Delta_s)  # discrétisation de l'axe vertical
x_grid = np.matrix(x,)
y_grid = np.matrix(y,)
t_grid = np.arange(0, T, Delta_t)  # discrétisation de l'axe temporel

###################################################### Fonctions de la simulation ##################################################


# Fonctions de la simulation

def f_0_scal():
    """Donne la valeur du paramètre de Coriolis f_0 en un scalaire à phi_0 fixé"""
    return 2*Omega*np.sin(phi_0)


def f_0_tabl():
    """Donne la valeur du paramètre de Coriolis f_0 à phi_0 fixé en un tableau sur toute la maille"""
    F_0 = np.full((N, M), f_0_scal())
    return F_0


def beta_scal(phi):
    """Donne la valeur du paramètre beta pour une valeur de latitude phi donnée"""
    return (2*Omega*np.cos(phi)/a)


# doit encore être transformer pour donner une valeur sur toute la maille dépendante de la latitude.
def beta_grid():
    """Donne la valeur du paramètre beta pour une valeur de phi_0 dépendant de la grille"""
    dphi = Delta_s/a
    beta_grid = np.zeros((N, M))
    beta_grid[int(N/2), :] = beta_scal(phi_0)
    phi_i = phi_0
    for i in range(1, int(N/2)+1):
        print(i)
        phi_i = phi_i + dphi
        beta_grid[-i, :] = beta_scal(phi_i)
    return beta_grid


X, Y = np.meshgrid(x, y)
b = beta_grid()
print(b[0])
plt.contourf(X, Y, b)
plt.colorbar()
plt.show()


def pos_y():
    """Donne la composante verticale y pour chaque point de la maille"""
    y_axis = np.arange(-(N/2)*Delta_s, (N/2)*Delta_s, Delta_s)
    y_label = np.zeros((N, M))
    Y = np.full((N, M), 1)
    for i in range(N):
        for j in range(M):
            y_label[i, j] = Y[i, j]*y_axis[i]*(-1)
    return y_label


def f():
    """Donne la valeur du paramètre de Coriolis dans l'approximation du plan beta en fonction de la position"""
    f = np.zeros((N, M))
    f = f_0_tabl()+beta()*pos_y()
    return f


# print(beta())
# print(f_0()[9, 5])
# print(y())
# print("f_0 = ", f_0())
# print("beta fois y = ", beta()*y())
# print("f =", f_0() + beta()*y())
f_value = f()
# print(f[-1])


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/W_x
    j = (2*np.pi)/W_y
    psi_0_T = np.zeros((N, M))
    psi_0_T = (g/f_0_scal())*(100*np.sin(k*(x_grid.T))@(np.cos(j*y_grid)))
    psi_0 = psi_0_T.T
    print(np.shape(psi_0))
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


def velocity_field(psi):
    """Donne champ de vitesse à partir de la fonction de courant"""
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
                u[i, j] = (-1/(2*Delta_s))*(psi[i+1, j]-psi[i-1, j])
                v[i, j] = (1/(2*Delta_s))*(psi[i, j+1]-psi[i, j-1])
    return [u, v]


def vort_flux(u, v, zeta):
    """Donne la valeur du flux de la composante verticale de la vorticité relative. Prend comme argument le champ de vitesse.
    Prend comme argument la matrice de la composante selon x du champ de vitesse, puis la matrice de la composante y du champs de vitesse
    , la matrice de la composante verticale de la vorticité relative."""
    F = np.zeros((N, M))
    beta = np.zeros((N, M))
    beta_grid = beta()
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
                F[i, j] = (1/(2*Delta_s))*((zeta[i, j+1]*u[i, j+1]-zeta[i, j-1]*u[i, j-1]
                                            ) + (zeta[i+1, j]*v[i+1, j] - zeta[i-1, j]*v[i-1, j])) + beta_grid[i, j]*v[i, j]
    return F


def poisson():
    return []

###################################################### Intégration temporelle ############################################################


def vort_dynamic():
    stream_func_dyn = np.zeros((N, M, K))
    U = np.zeros((N, M, K))
    V = np.zeros((N, M, K))
    vort_dyn = np.zeros((N, M, K))
    F_dyn = np.zeros((N, M, K))
    for i in range(K):
        if i == 0:  # On pose les conditions initiale à t=0
            stream_func_dyn[:, :, 0] = psi_init()
            U[:, :, 0] = velocity_field(psi_init())[0]
            V[:, :, 0] = velocity_field(psi_init())[1]
            vort_dyn[:, :, 0] = vort_init()
            F_dyn[:, :, 0] = vort_flux(U[:, :, 0], V[:, :, 0], vort_dyn[:, :, 0])
        else:  # On résoud dans le temps avec le schéma d'Euler avant.
            stream_func_dyn[:, :, i] = poisson()
            U[:, :, i] = velocity_field(stream_func_dyn[:, :, i])[0]
            V[:, :, i] = velocity_field(stream_func_dyn[:, :, i])[1]
            F_dyn[:, :, i] = vort_flux(U[:, :, i], V[:, :, i], vort_dyn[:, :, i])
            vort_dyn[:, :, i+1] = F_dyn[:, :, i]*Delta_t + vort_dyn[:, :, i]
    return vort_dyn


solution = vort_dynamic()

###################################################### Résultats et figures ############################################################
# Output de la résolution des mailles
print("-----------------------------------------------------------------------------------------------")
print("Résolution numérique avec une maille spatiale de {}x{} points".format(
    M, N))
print("Résolution numérique avec une maille temporelle de {} points".format(K))
print("-----------------------------------------------------------------------------------------------")

# Contourplot de la fonction de courant initiale

X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, psi_init(), 100)
plt.colorbar()
plt.title("Contour plot de la fonction de courant initiale $\psi_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.show()

# Contourplot de la composante verticale de la vorticité relative initiale
plt.contourf(X, Y, vort_init(), 100)
plt.colorbar()
plt.title("Contour plot de $\zeta_0(x,y)$ \n $L_x = {}km, L_y = {}km, \Delta_s = {}km, W_x = {}km, W_y = {}km$ ".format(
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.show()


# Plot du champ de vitesse initial

U = velocity_field(psi_init())[0]
V = velocity_field(psi_init())[1]
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
    int(L_x/1000), int(L_y/1000), int(Delta_s/1000), int(W_x/1000), int(W_y/1000)), fontsize=11)
plt.legend(loc='best', shadow=True, fontsize="large")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.tight_layout()
plt.show()

# Plot du champ du paramètre de Coriolis

F = f()
plt.contourf(X, Y, F, 100)
plt.colorbar()
plt.title("Valeur du paramètre de Coriolis f dans l'approximation du plan \beta \n f = f_0 + \beta y", fontsize=11)
plt.xlabel("$x$", fontsize=15)
plt.ylabel("$y$", fontsize=15)
plt.tight_layout()
plt.grid()
plt.show()
