##### Fonctions initiales avec CBP sans utiliser le modulo ######


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
    """Donne la valeur du flux de la composante verticale de la vorticité relative F. Prend comme argument le champ de vitesse.
    Prend comme argument la matrice de la composante selon x du champ de vitesse, puis la matrice de la composante y du champs de vitesse
    , la matrice de la composante verticale de la vorticité relative."""
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
                F[i, j] = (1/(2*Delta_s))*((zeta[i, j+1]*u[i, j+1]-zeta[i, j-1]*u[i, j-1]
                                            ) + (zeta[i+1, j]*v[i+1, j] - zeta[i-1, j]*v[i-1, j])) + beta_scal(phi_0)*v[i, j]
    return F


def psi_init():
    """Donne une condition initiale pour la fonction psi_0"""
    k = (2*np.pi)/W_x
    j = (2*np.pi)/W_y
    psi_0_T = np.zeros((N, M))
    psi_0_T = (g/f_0_scal())*(100*np.sin(k*(x_grid.T))@(np.cos(j*y_grid)))
    psi_0 = psi_0_T.T
    return psi_0


def zeta_dynamic():
    # On crée des tableaux à trois dimensions pour les deux dimensions spatiales et une dimension temporelle.
    psi_dyn = np.zeros((N, M, K))
    U = np.zeros((N, M, K))
    V = np.zeros((N, M, K))
    zeta_dyn = np.zeros((N, M, K))
    F_dyn = np.zeros((N, M, K))
    for t in range(K-1):
        if t == 0:  # On pose les conditions initiale à t=0
            U[:, :, 0] = u_0
            V[:, :, 0] = v_0
            psi_dyn[:, :, 0] = psi_0
            zeta_dyn[:, :, 0] = zeta_0
            F_dyn[:, :, 0] = vort_flux(zeta_0, u_0, v_0)
            # On résoud avec un schéma d'Euler avant pour t =0
            zeta_dyn[:, :, 1] = -F_dyn[:, :, 0]*Delta_t + zeta_0

        else:  # On résoud dans le temps avec un schéma centré
            psi_dyn[:, :, t] = psi(zeta_dyn[:, :, t])
            U[:, :, t] = u(psi_dyn[:, :, t])
            V[:, :, t] = v(psi_dyn[:, :, t])
            F_dyn[:, :, t] = vort_flux(U[:, :, t], V[:, :, t], zeta_dyn[:, :, t])
            zeta_dyn[:, :, t+1] = -F_dyn[:, :, t]*2*Delta_t + zeta_dyn[:, :, t]
    return zeta_dyn, U, V, psi_dyn


solution = zeta_dynamic()
zeta_dyn = solution[0]
U = solution[1]
V = solution[2]
psi_dyn = solution[3]
