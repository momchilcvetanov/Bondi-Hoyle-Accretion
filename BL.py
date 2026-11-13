import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

start_wall_time = time.perf_counter()

def conserved_to_primitive(U, M, r, a, tol=1e-10, max_iter=50):
    D = U[0]
    Sr = U[1]
    Sphi = U[2]
    tau = U[3]

    E = tau + D
    kappa = Gamma/(Gamma - 1.0)

    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r

    S_square = Triangle*Sr*Sr/(r*r) + r*r*Sphi*Sphi/Sumsymbol

    p = np.maximum((Gamma - 1.0)*tau, 1e-12)

    for i in range(max_iter):
        Q = E + p
        v_square = S_square/(Q*Q)
        W = 1.0/np.sqrt(1.0 - v_square)

        f = D*W + kappa*p*W*W - Q
        df = -D*W*W*W*v_square/Q + kappa*W*W - 2.0*kappa*p*W**4*v_square/Q - 1.0
        dp = -f/df

        p_new = p + dp
        p_new = np.maximum(p_new, 1e-12)

        if np.all(np.abs(dp) < tol*(1.0 + p_new)):
            p = p_new
            break

        p = p_new

    Q = E + p
    v_square = S_square/(Q*Q)
    W = 1.0/np.sqrt(1.0 - v_square)

    vr = Triangle*Sr/(r*r*Q)
    vphi = r*r*Sphi/(Sumsymbol*Q)
    rho = D/W
    return np.array([rho, p, vr, vphi])

def primitive_to_conserved(rho, p, vr, vphi, M, r, a):
    h = 1 + (Gamma/(Gamma - 1))*(p/rho)

    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r

    v_square = r*r*vr*vr/Triangle + Sumsymbol*vphi*vphi/(r*r)
    W = 1.0/np.sqrt(1.0 - v_square)

    Q = rho*h*W*W

    D = rho*W
    Sr = Q*r*r*vr/Triangle
    Sphi = Q*Sumsymbol*vphi/(r*r)
    tau = Q - p - D
    return np.array([D, Sr, Sphi, tau])

def FunctionF(rho, p, vr, vphi, M, r, a):
    U = primitive_to_conserved(rho, p, vr, vphi, M, r, a)

    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    alpha = r*np.sqrt(Triangle/Sumsymbol)

    D    = U[0]
    Sr   = U[1]
    Sphi = U[2]
    tau  = U[3]

    term1 = D*vr
    term2 = Sr*vr + p
    term3 = Sphi*vr
    term4 = (tau + p)*vr
    return alpha*np.array([term1, term2, term3, term4])

def FunctionG(rho, p, vr, vphi, M, r, a):
    U = primitive_to_conserved(rho, p, vr, vphi, M, r, a)
    
    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    alpha = r*np.sqrt(Triangle/Sumsymbol)
    betaphi = -2*M*a*r/Sumsymbol

    D    = U[0]
    Sr   = U[1]
    Sphi = U[2]
    tau  = U[3]

    term1 = D*(vphi - betaphi/alpha)
    term2 = Sr*(vphi - betaphi/alpha)
    term3 = Sphi*(vphi - betaphi/alpha) + p
    term4 = (tau + p)*vphi - tau*betaphi/alpha
    return alpha*np.array([term1, term2, term3, term4])

def Functionksi(rho, p, vr, vphi, M, r, a):
    U = primitive_to_conserved(rho, p, vr, vphi, M, r, a)

    D    = U[0]
    Sr   = U[1]
    Sphi = U[2]
    tau  = U[3]

    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    drTriangle = 2*(r - M)
    drSumsymbol = 4*r*r*r + 2*a*a*r + 2*M*a*a

    betaphi = -2*M*a*r/Sumsymbol

    gtphi_down = -2*M*a/r
    grr_down = r*r/Triangle
    gphiphi_down = r*r + a*a + 2*M*a*a/r

    drgrr_down = 2*r*(a*a - M*r)/(Triangle*Triangle)
    drgtphi_down = 2*M*a/(r*r)
    drgphiphi_down = 2*r - 2*M*a*a/(r*r)

    alpha_squared = r*r*Triangle/Sumsymbol
    alpha = np.sqrt(alpha_squared)
    dralpha = alpha*(1/r + 0.5*drTriangle/Triangle - 0.5*drSumsymbol/Sumsymbol)

    Gamma_r_tt = M*Triangle/(r**4)
    Gamma_r_rr = (a*a - M*r)/(r*Triangle)
    Gamma_r_thetatheta = -Triangle/r
    Gamma_r_phiphi = -Triangle*(r*r*r - M*a*a)/(r**4)
    Gamma_r_tphi = -M*a*Triangle/(r**4)

    Gamma_t_rphi = -M*a*(a*a + 3*r*r)/(r*r*Triangle)
    Gamma_t_tr = M*(a*a + r*r)/(r*r*Triangle)

    Gamma_phi_rphi = (r*r*r - 2*M*r*r - M*a*a)/(r*r*Triangle)
    Gamma_phi_tr = M*a/(r*r*Triangle)

    h = 1 + (Gamma/(Gamma - 1))*(p/rho)
    W = 1.0/np.sqrt(1.0 - r*r*vr*vr/Triangle - Sumsymbol*vphi*vphi/(r*r))
    Q = rho*h*W*W
    vphi_thilda = vphi - betaphi/alpha

    T_tt = (Q - p)/(alpha*alpha)
    T_tr = Q*vr/alpha
    T_tphi = Q*vphi_thilda/alpha + p*betaphi/(alpha*alpha)
    T_rr = Q*vr*vr + p*Triangle/(r*r)
    T_rphi = Q*vr*vphi_thilda
    T_thetatheta = p/(r*r)
    T_phiphi = Q*vphi_thilda**2 + p*(Triangle - a*a)/(r*r*Triangle)
    
    Omega = 1/r - 0.5*drTriangle/Triangle + 0.5*drSumsymbol/Sumsymbol
    A1 = T_tt*Gamma_r_tt + T_rr*Gamma_r_rr + T_thetatheta*Gamma_r_thetatheta + T_phiphi*Gamma_r_phiphi + 2*T_tphi*Gamma_r_tphi
    A2 = T_rphi*Gamma_t_rphi + T_tr*Gamma_t_tr
    A3 = T_rphi*Gamma_phi_rphi + T_tr*Gamma_phi_tr

    term1 = -alpha*D*vr*Omega
    term2 = -alpha*(Sr*vr + p)*Omega + alpha*T_rr*drgrr_down - alpha*A1*grr_down
    term3 = -alpha*Sphi*vr*Omega + alpha*T_tr*drgtphi_down + alpha*T_rphi*drgphiphi_down - 2*alpha*A2*gtphi_down - 2*alpha*A3*gphiphi_down
    term4 = -alpha*(tau + p)*vr*Omega + alpha*T_tr*dralpha - 2*alpha*alpha*A2
    return np.array([term1, term2, term3, term4])


def minmod(a, b):
    return np.where(np.abs(a) < np.abs(b), a, b)*(np.sign(a) == np.sign(b))

def characteristic_speeds(vn, vt, cs):
    v_square = vn*vn + vt*vt
    denominator = 1.0 - v_square*cs*cs
    root = np.sqrt((1.0 - v_square)*(1.0 - v_square*cs*cs - vn*vn*(1.0 - cs*cs)))
    lm = (vn*(1.0 - cs*cs) - cs*root)/denominator
    lp = (vn*(1.0 - cs*cs) + cs*root)/denominator
    return lm, lp

def reconstruct_x(w):
    w_im2 = w[:, :-3, 2:-2]
    w_im1 = w[:, 1:-2, 2:-2]
    w_i   = w[:, 2:-1, 2:-2]
    w_ip1 = w[:, 3:, 2:-2]

    slope_im1 = minmod(w_im1 - w_im2, w_i - w_im1)
    slope_i = minmod(w_i - w_im1, w_ip1 - w_i)
    w_left  = w_im1 + 0.5*slope_im1
    w_right = w_i - 0.5*slope_i
    return w_left, w_right

def rusanov_flux_x(w_left, w_right, M, r, a):
    rho_L = w_left[0]
    p_L   = w_left[1]
    vr_L  = w_left[2]
    vphi_L  = w_left[3]
    h_L = 1 + (Gamma/(Gamma - 1))*(p_L/rho_L)

    rho_R = w_right[0]
    p_R   = w_right[1]
    vr_R  = w_right[2]
    vphi_R  = w_right[3]
    h_R = 1 + (Gamma/(Gamma - 1))*(p_R/rho_R)

    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    alpha = r*np.sqrt(Triangle/Sumsymbol)

    v_square_L = r*r*vr_L*vr_L/Triangle + Sumsymbol*vphi_L*vphi_L/(r*r)
    v_square_R = r*r*vr_R*vr_R/Triangle + Sumsymbol*vphi_R*vphi_R/(r*r)

    if np.any(v_square_L >= 1.0):
        print("Invalid reconstructed radial LEFT state, max v^2 =", np.max(v_square_L))

    if np.any(v_square_R >= 1.0):
        print("Invalid reconstructed radial RIGHT state, max v^2 =", np.max(v_square_R))

    U_L = primitive_to_conserved(rho_L, p_L, vr_L, vphi_L, M, r, a)
    U_R = primitive_to_conserved(rho_R, p_R, vr_R, vphi_R, M, r, a)

    cs_L = np.sqrt(Gamma*p_L/(rho_L*h_L))
    cs_R = np.sqrt(Gamma*p_R/(rho_R*h_R))

    vr_physical_L = r*vr_L/np.sqrt(Triangle)
    vphi_physical_L = np.sqrt(Sumsymbol)*vphi_L/r

    vr_physical_R = r*vr_R/np.sqrt(Triangle)
    vphi_physical_R = np.sqrt(Sumsymbol)*vphi_R/r

    lm_L, lp_L = characteristic_speeds(vr_physical_L, vphi_physical_L, cs_L)
    lm_R, lp_R = characteristic_speeds(vr_physical_R, vphi_physical_R, cs_R)

    coordinate_factor = alpha*np.sqrt(Triangle)/r

    lm_L = coordinate_factor*lm_L
    lp_L = coordinate_factor*lp_L
    lm_R = coordinate_factor*lm_R
    lp_R = coordinate_factor*lp_R

    a_rusanov = np.maximum.reduce([np.abs(lm_L), np.abs(lp_L), np.abs(lm_R), np.abs(lp_R)])

    F_L = FunctionF(rho_L, p_L, vr_L, vphi_L, M, r, a)
    F_R = FunctionF(rho_R, p_R, vr_R, vphi_R, M, r, a)
    return 0.5*(F_L + F_R - a_rusanov[None, :]*(U_R - U_L)), a_rusanov

def reconstruct_y(w):
    w_im2 = w[:, 2:-2, :-3]
    w_im1 = w[:, 2:-2, 1:-2]
    w_i   = w[:, 2:-2, 2:-1]
    w_ip1 = w[:, 2:-2, 3:]

    slope_im1 = minmod(w_im1 - w_im2, w_i - w_im1)
    slope_i = minmod(w_i - w_im1, w_ip1 - w_i)
    w_left  = w_im1 + 0.5*slope_im1
    w_right = w_i - 0.5*slope_i
    return w_left, w_right

def rusanov_flux_y(w_left, w_right, M, r, a):
    rho_L = w_left[0]
    p_L   = w_left[1]
    vr_L  = w_left[2]
    vphi_L  = w_left[3]
    h_L = 1 + (Gamma/(Gamma - 1))*(p_L/rho_L)

    rho_R = w_right[0]
    p_R   = w_right[1]
    vr_R  = w_right[2]
    vphi_R  = w_right[3]
    h_R = 1 + (Gamma/(Gamma - 1))*(p_R/rho_R)

    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    alpha = r*np.sqrt(Triangle/Sumsymbol)
    betaphi = -2*M*a*r/Sumsymbol

    v_square_L = r*r*vr_L*vr_L/Triangle + Sumsymbol*vphi_L*vphi_L/(r*r)
    v_square_R = r*r*vr_R*vr_R/Triangle + Sumsymbol*vphi_R*vphi_R/(r*r)

    if np.any(v_square_L >= 1.0):
        print("Invalid reconstructed phi LEFT state, max v^2 =", np.max(v_square_L))

    if np.any(v_square_R >= 1.0):
        print("Invalid reconstructed phi RIGHT state, max v^2 =", np.max(v_square_R))

    U_L = primitive_to_conserved(rho_L, p_L, vr_L, vphi_L, M, r, a)
    U_R = primitive_to_conserved(rho_R, p_R, vr_R, vphi_R, M, r, a)

    cs_L = np.sqrt(Gamma*p_L/(rho_L*h_L))
    cs_R = np.sqrt(Gamma*p_R/(rho_R*h_R))

    vr_physical_L = r*vr_L/np.sqrt(Triangle)
    vphi_physical_L = np.sqrt(Sumsymbol)*vphi_L/r

    vr_physical_R = r*vr_R/np.sqrt(Triangle)
    vphi_physical_R = np.sqrt(Sumsymbol)*vphi_R/r

    lm_L, lp_L = characteristic_speeds(vphi_physical_L, vr_physical_L, cs_L)
    lm_R, lp_R = characteristic_speeds(vphi_physical_R, vr_physical_R, cs_R)

    coordinate_factor = alpha*r/np.sqrt(Sumsymbol)

    lm_L = coordinate_factor*lm_L - betaphi
    lp_L = coordinate_factor*lp_L - betaphi
    lm_R = coordinate_factor*lm_R - betaphi
    lp_R = coordinate_factor*lp_R - betaphi

    a_rusanov = np.maximum.reduce([np.abs(lm_L), np.abs(lp_L), np.abs(lm_R), np.abs(lp_R)])

    G_L = FunctionG(rho_L, p_L, vr_L, vphi_L, M, r, a)
    G_R = FunctionG(rho_R, p_R, vr_R, vphi_R, M, r, a)
    return 0.5*(G_L + G_R - a_rusanov[None, :, :]*(U_R - U_L)), a_rusanov

def asymptotic_state(M, r, phi, a):
    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r

    PHI_local = phi[None, :]

    rho = rhoinf*np.ones((r.shape[0], phi.size))
    p = pinf*np.ones((r.shape[0], phi.size))

    vr = np.sqrt(Triangle)/r*vinf*np.cos(PHI_local)
    vphi = -r/np.sqrt(Sumsymbol)*vinf*np.sin(PHI_local)
    return rho, p, vr, vphi

def coordinate_to_physical_velocity(vr, vphi, M, r, a):
    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    vr_physical = r*vr/np.sqrt(Triangle)
    vphi_physical = np.sqrt(Sumsymbol)*vphi/r
    return vr_physical, vphi_physical

def physical_to_coordinate_velocity(vr_physical, vphi_physical, M, r, a):
    Triangle = r*r - 2*M*r + a*a
    Sumsymbol = r**4 + a*a*r*r + 2*M*a*a*r
    vr = np.sqrt(Triangle)*vr_physical/r
    vphi = r*vphi_physical/np.sqrt(Sumsymbol)
    return vr, vphi

def boundary_condition(U, M, r, phi, a):
    w_inner = conserved_to_primitive(U[:, 2:4, 2:-2], M, r[2:4], a)

    vr_inner_physical, vphi_inner_physical = coordinate_to_physical_velocity(w_inner[2], w_inner[3], M, r[2:4], a)

    rho_inner_1 = w_inner[0, 0]
    p_inner_1 = w_inner[1, 0]
    vphi_inner_1_physical = vphi_inner_physical[0]
    vr_inner_1_physical = 2.0*vr_inner_physical[0] - vr_inner_physical[1]

    rho_inner_0 = w_inner[0, 0]
    p_inner_0 = w_inner[1, 0]
    vphi_inner_0_physical = vphi_inner_physical[0]
    vr_inner_0_physical = 2.0*vr_inner_1_physical - vr_inner_physical[0]

    v_square_inner_1 = vr_inner_1_physical*vr_inner_1_physical + vphi_inner_1_physical*vphi_inner_1_physical
    v_square_inner_0 = vr_inner_0_physical*vr_inner_0_physical + vphi_inner_0_physical*vphi_inner_0_physical

    if np.any(v_square_inner_1 >= 1.0):
        raise RuntimeError("Superluminal velocity at inner ghost 1")

    if np.any(v_square_inner_0 >= 1.0):
        raise RuntimeError("Superluminal velocity at inner ghost 0")

    vr_inner_1, vphi_inner_1 = physical_to_coordinate_velocity(vr_inner_1_physical, vphi_inner_1_physical, M, r[1], a)
    vr_inner_0, vphi_inner_0 = physical_to_coordinate_velocity(vr_inner_0_physical, vphi_inner_0_physical, M, r[0], a)

    U[:, 1, 2:-2] = primitive_to_conserved(rho_inner_1, p_inner_1, vr_inner_1, vphi_inner_1, M, r[1], a)
    U[:, 0, 2:-2] = primitive_to_conserved(rho_inner_0, p_inner_0, vr_inner_0, vphi_inner_0, M, r[0], a)

    w_outer = conserved_to_primitive(U[:, -3:-2, 2:-2], M, r[-3:-2], a)

    rho_downstream = w_outer[0, 0]
    p_downstream = w_outer[1, 0]

    vr_downstream_physical, vphi_downstream_physical = coordinate_to_physical_velocity(w_outer[2, 0], w_outer[3, 0], M, r[-3], a)

    vr_downstream_1, vphi_downstream_1 = physical_to_coordinate_velocity(vr_downstream_physical, vphi_downstream_physical, M, r[-2], a)
    vr_downstream_2, vphi_downstream_2 = physical_to_coordinate_velocity(vr_downstream_physical, vphi_downstream_physical, M, r[-1], a)

    U_downstream_1 = primitive_to_conserved(rho_downstream, p_downstream, vr_downstream_1, vphi_downstream_1, M, r[-2], a)
    U_downstream_2 = primitive_to_conserved(rho_downstream, p_downstream, vr_downstream_2, vphi_downstream_2, M, r[-1], a)

    rho_upstream, p_upstream, vr_upstream, vphi_upstream = asymptotic_state(M, r[-2:], phi, a)

    U_upstream = primitive_to_conserved(rho_upstream, p_upstream, vr_upstream, vphi_upstream, M, r[-2:], a)

    upstream = np.cos(phi) < 0.0

    U[:, -2, 2:-2] = np.where(upstream[None, :], U_upstream[:, 0], U_downstream_1)
    U[:, -1, 2:-2] = np.where(upstream[None, :], U_upstream[:, 1], U_downstream_2)

    U[:, :, 0] = U[:, :, -4]
    U[:, :, 1] = U[:, :, -3]
    U[:, :, -2] = U[:, :, 2]
    U[:, :, -1] = U[:, :, 3]
    
def RHS(U, M, R_full, R_face, R, a):
    w = conserved_to_primitive(U, M, R_full, a)
    w_left_x, w_right_x = reconstruct_x(w)
    Fr, ar = rusanov_flux_x(w_left_x, w_right_x, M, R_face, a)

    w_left_y, w_right_y = reconstruct_y(w)
    Fphi, aphi = rusanov_flux_y(w_left_y, w_right_y, M, R, a)

    w_physical = w[:, 2:-2, 2:-2]

    rho = w_physical[0]
    p = w_physical[1]
    vr = w_physical[2]
    vphi = w_physical[3]

    ksi = Functionksi(rho, p, vr, vphi, M, R, a)

    L = np.zeros_like(U)
    L[:, 2:-2, 2:-2] = -(Fr[:, 1:, :] - Fr[:, :-1, :])/dr - (Fphi[:, :, 1:] - Fphi[:, :, :-1])/dphi + ksi
    return L, ar, aphi

def shock_angles(rho, r, phi, target_r):
    i_r = np.argmin(np.abs(r - target_r))

    log_rho_ring = np.log10(rho[i_r]/rhoinf)
    gradient = np.abs(np.gradient(log_rho_ring, dphi))

    upper_mask = (phi > 0.0) & (phi < 0.5*np.pi)
    lower_mask = (phi > 1.5*np.pi) & (phi < 2.0*np.pi)

    upper_indices = np.where(upper_mask)[0]
    lower_indices = np.where(lower_mask)[0]

    i_upper = upper_indices[np.argmax(gradient[upper_indices])]
    i_lower = lower_indices[np.argmax(gradient[lower_indices])]

    phi_upper = phi[i_upper]
    phi_lower = phi[i_lower] - 2.0*np.pi

    cone_center = 0.5*(phi_upper + phi_lower)
    cone_half_angle = 0.5*(phi_upper - phi_lower)

    return phi_upper, phi_lower, cone_center, cone_half_angle

def accretion_rates(rho, p, vr, vphi, M, r, a, target_r):
    R_local = r[:, None]

    Triangle = R_local*R_local - 2*M*R_local + a*a
    Sumsymbol = R_local**4 + a*a*R_local*R_local + 2*M*a*a*R_local

    Triangle_r = 2*(R_local - M)
    Sumsymbol_r = 4*R_local**3 + 2*a*a*R_local + 2*M*a*a

    Omega = 1/R_local + Sumsymbol_r/(2*Sumsymbol) - Triangle_r/(2*Triangle)

    sqrt_gamma = R_local*np.sqrt(Sumsymbol/Triangle)

    U_local = primitive_to_conserved(rho, p, vr, vphi, M, R_local, a)

    D = U_local[0]
    Sr = U_local[1]
    Sphi = U_local[2]

    Fr = FunctionF(rho, p, vr, vphi, M, R_local, a)
    ksi = Functionksi(rho, p, vr, vphi, M, R_local, a)

    source_r = ksi[1] + Fr[1]*Omega
    source_phi = ksi[2] + Fr[2]*Omega

    i_r = np.argmin(np.abs(r - target_r))
    r_c = r[i_r]

    mdot = -r_c*r_c*dphi*np.sum(D[i_r]*vr[i_r])

    prdot_surface = -r_c*r_c*dphi*np.sum(Sr[i_r]*vr[i_r] + p[i_r])
    pphidot_surface = -r_c*r_c*dphi*np.sum(Sphi[i_r]*vr[i_r])

    prdot_source = dr*dphi*np.sum(sqrt_gamma[:i_r + 1]*source_r[:i_r + 1])
    pphidot_source = dr*dphi*np.sum(sqrt_gamma[:i_r + 1]*source_phi[:i_r + 1])

    prdot = prdot_surface + prdot_source
    pphidot = pphidot_surface + pphidot_source

    return mdot, prdot, pphidot, prdot_surface, prdot_source, pphidot_surface, pphidot_source

CFL = 0.4

Gamma = 5.0/3.0

M = 1.0
a = 0.99
Mach_inf = 5.0
vinf = 0.5
csinf = vinf/Mach_inf

r_min = 1.83
r_max = 35.0
Nr = 600
dr = (r_max - r_min)/Nr

phi_min = 0.0
phi_max = 2.0*np.pi
Nphi = 600
dphi = (phi_max - phi_min)/Nphi

r = r_min + (np.arange(Nr) + 0.5)*dr
phi = phi_min + (np.arange(Nphi) + 0.5)*dphi

r_face = r_min + np.arange(Nr + 1)*dr
r_full = r_min + (np.arange(Nr + 4) - 1.5)*dr

R = r[:, None]
R_face = r_face[:, None]
R_full = r_full[:, None]

R_plot, PHI = np.meshgrid(r, phi, indexing="ij")

X = R_plot*np.cos(PHI)
Y = R_plot*np.sin(PHI)

Triangle_R = R*R - 2*M*R + a*a
Sumsymbol_R = R**4 + a*a*R*R + 2*M*a*a*R

rhoinf = 1.0
pinf = rhoinf*csinf*csinf*(Gamma - 1)/(Gamma*(Gamma - 1 - csinf*csinf))

rho = rhoinf*np.ones((Nr, Nphi))
p = pinf*np.ones((Nr, Nphi))

vr = np.sqrt(Triangle_R)/R*vinf*np.cos(PHI)
vphi = -R/np.sqrt(Sumsymbol_R)*vinf*np.sin(PHI)

v_square = R*R*vr*vr/Triangle_R + Sumsymbol_R*vphi*vphi/(R*R)

print("max |v^2 - vinf^2| =", np.max(np.abs(v_square - vinf*vinf)))
print("v^2 min/max =", np.min(v_square), np.max(v_square))

r_plus = M + np.sqrt(M*M - a*a)

v_square_initial = R*R*vr*vr/Triangle_R + Sumsymbol_R*vphi*vphi/(R*R)

h_initial = 1 + (Gamma/(Gamma - 1))*(p/rho)
cs_initial = np.sqrt(Gamma*p/(rho*h_initial))

print("Model 4")
print("r_plus =", r_plus)
print("r_min =", r_min)
print("r_max =", r_max)
print("minimum ghost radius =", R_full[0, 0])
print("Mach_inf =", Mach_inf)
print("vinf =", vinf)
print("csinf =", csinf)
print("rhoinf =", rhoinf)
print("pinf =", pinf)
print("max |v^2 - vinf^2| =", np.max(np.abs(v_square_initial - vinf*vinf)))
print("cs initial min/max =", np.min(cs_initial), np.max(cs_initial))
print()

r_acc = M/(vinf*vinf + csinf*csinf)

i_inner = 0
i_acc = np.argmin(np.abs(r - r_acc))
i_10 = np.argmin(np.abs(r - 10.0))
i_20 = np.argmin(np.abs(r - 20.0))

print("r_acc =", r_acc)
print("r used for r_acc =", r[i_acc])

lambda_B = 0.25

mdot_canonical = 4*np.pi*lambda_B*M*M*rhoinf/(vinf*vinf + csinf*csinf)**1.5

print("Canonical mass accretion rate =", mdot_canonical)

assert R_full[0, 0] > r_plus

U = np.zeros((4, Nr + 4, Nphi + 4))
U[:, 2:-2, 2:-2] = primitive_to_conserved(rho, p, vr, vphi, M, R, a)

steps = 200000
simulation_time = 0.0
end_time = 500.0

time_history = []

mdot_inner_history = []
mdot_acc_history = []
mdot_10_history = []
mdot_20_history = []
mdot_normalized_history = []

prdot_history = []
pphidot_history = []

prdot_surface_history = []
prdot_source_history = []
pphidot_surface_history = []
pphidot_source_history = []

rho_max_history = []
rho_min_history = []
p_max_history = []
p_min_history = []
W_max_history = []

shock_center_history = []
shock_half_angle_history = []

pressure_asymmetry_history = []

for i in range(steps):
    boundary_condition(U, M, R_full, phi, a)
    L1, ax1, ay1 = RHS(U, M, R_full, R_face, R, a)

    dt = CFL/(np.max(ax1)/dr + np.max(ay1)/dphi)
    dt = min(dt, end_time - simulation_time)

    U_old = U.copy()
    U1 = U_old + dt*L1
    boundary_condition(U1, M, R_full, phi, a)

    L2, ax2, ay2 = RHS(U1, M, R_full, R_face, R, a)

    U = 0.5*U_old + 0.5*(U1 + dt*L2)
    boundary_condition(U, M, R_full, phi, a)

    simulation_time += dt
    if np.any(~np.isfinite(U[:, 2:-2, 2:-2])):
            print("NaN/Inf in U at step", i)
            break

    if i%100 == 0:
        elapsed = time.perf_counter() - start_wall_time
        steps_per_second = (i + 1)/elapsed
        remaining_steps_estimate = (end_time - simulation_time)/dt
        remaining_seconds_estimate = remaining_steps_estimate/steps_per_second

        print("progress: step =", i, "t =", simulation_time, "dt =", dt)
        print("steps/s =", steps_per_second)
        print("estimated remaining hours =", remaining_seconds_estimate/3600.0)
        print()

    if i%2000 == 0 or simulation_time >= end_time:
        w_check = conserved_to_primitive(U[:, 2:-2, 2:-2], M, R, a)

        rho_check = w_check[0]
        p_check = w_check[1]
        vr_check = w_check[2]
        vphi_check = w_check[3]

        if np.any(~np.isfinite(w_check)):
            print("NaN/Inf in primitives at step", i)
            break

        if np.any(rho_check <= 0):
            print("Negative density at step", i, "min rho =", np.min(rho_check))
            break

        if np.any(p_check <= 0):
            print("Negative pressure at step", i, "min p =", np.min(p_check))
            break

        v_square_check = R*R*vr_check*vr_check/Triangle_R + Sumsymbol_R*vphi_check*vphi_check/(R*R)

        if np.any(v_square_check >= 1):
            print("Superluminal velocity at step", i, "max v^2 =", np.max(v_square_check))
            break

        W_check = 1.0/np.sqrt(1.0 - v_square_check)

        D_check = rho_check*W_check

        mdot_inner = -dphi*np.sum(r[i_inner]*r[i_inner]*D_check[i_inner]*vr_check[i_inner])
        mdot_acc = -dphi*np.sum(r[i_acc]*r[i_acc]*D_check[i_acc]*vr_check[i_acc])
        mdot_10 = -dphi*np.sum(r[i_10]*r[i_10]*D_check[i_10]*vr_check[i_10])
        mdot_20 = -dphi*np.sum(r[i_20]*r[i_20]*D_check[i_20]*vr_check[i_20])

        mdot_normalized = mdot_acc/mdot_canonical

        mdot_rate, prdot, pphidot, prdot_surface, prdot_source, pphidot_surface, pphidot_source = accretion_rates(rho_check, p_check, vr_check, vphi_check, M, r, a, r_acc)

        phi_upper, phi_lower, cone_center, cone_half_angle = shock_angles(rho_check, r, phi, r_acc)

        inner_region = r <= 4.0
        upper_phi = np.sin(phi) > 0.0
        lower_phi = np.sin(phi) < 0.0

        p_upper = np.mean(p_check[inner_region][:, upper_phi])
        p_lower = np.mean(p_check[inner_region][:, lower_phi])

        pressure_asymmetry = (p_upper - p_lower)/(0.5*(p_upper + p_lower))

        time_history.append(simulation_time)

        mdot_inner_history.append(mdot_inner)
        mdot_acc_history.append(mdot_acc)
        mdot_10_history.append(mdot_10)
        mdot_20_history.append(mdot_20)
        mdot_normalized_history.append(mdot_normalized)

        prdot_history.append(prdot)
        pphidot_history.append(pphidot)

        prdot_surface_history.append(prdot_surface)
        prdot_source_history.append(prdot_source)
        pphidot_surface_history.append(pphidot_surface)
        pphidot_source_history.append(pphidot_source)

        rho_max_history.append(np.max(rho_check))
        rho_min_history.append(np.min(rho_check))
        p_max_history.append(np.max(p_check))
        p_min_history.append(np.min(p_check))
        W_max_history.append(np.max(W_check))

        shock_center_history.append(np.degrees(cone_center))
        shock_half_angle_history.append(np.degrees(cone_half_angle))

        pressure_asymmetry_history.append(pressure_asymmetry)

        print("step =", i, "t =", simulation_time, "dt =", dt)
        print("rho min/max =", np.min(rho_check), np.max(rho_check))
        print("p min/max =", np.min(p_check), np.max(p_check))
        print("v^2 max =", np.max(v_square_check))
        print("W max =", np.max(W_check))
        print()
        print("mdot inner =", mdot_inner)
        print("mdot r_acc =", mdot_acc)
        print("mdot r=10 =", mdot_10)
        print("mdot r=20 =", mdot_20)
        print("normalized mdot =", mdot_normalized)
        print()
        print("radial momentum rate =", prdot)
        print("radial surface =", prdot_surface)
        print("radial source =", prdot_source)
        print()
        print("angular momentum rate =", pphidot)
        print("angular surface =", pphidot_surface)
        print("angular source =", pphidot_source)
        print()
        print("shock center =", np.degrees(cone_center), "deg")
        print("shock half-opening =", np.degrees(cone_half_angle), "deg")
        print("pressure asymmetry =", pressure_asymmetry)
        print()

    if simulation_time >= end_time:
        break

w_final = conserved_to_primitive(U[:, 2:-2, 2:-2], M, R, a)

rho_final = w_final[0]
p_final = w_final[1]
vr_final = w_final[2]
vphi_final = w_final[3]

time_history = np.array(time_history)

prdot_surface_history = np.array(prdot_surface_history)
prdot_source_history = np.array(prdot_source_history)

pphidot_surface_history = np.array(pphidot_surface_history)
pphidot_source_history = np.array(pphidot_source_history)

mdot_inner_history = np.array(mdot_inner_history)
mdot_acc_history = np.array(mdot_acc_history)
mdot_10_history = np.array(mdot_10_history)
mdot_20_history = np.array(mdot_20_history)
mdot_normalized_history = np.array(mdot_normalized_history)

prdot_history = np.array(prdot_history)
pphidot_history = np.array(pphidot_history)

rho_max_history = np.array(rho_max_history)
rho_min_history = np.array(rho_min_history)
p_max_history = np.array(p_max_history)
p_min_history = np.array(p_min_history)
W_max_history = np.array(W_max_history)

shock_center_history = np.array(shock_center_history)
shock_half_angle_history = np.array(shock_half_angle_history)

pressure_asymmetry_history = np.array(pressure_asymmetry_history)


phi_upper, phi_lower, cone_center, cone_half_angle = shock_angles(rho_final, r, phi, r_acc)

print()
print("Shock diagnostic at r_acc")
print("upper shock angle =", np.degrees(phi_upper), "deg")
print("lower shock angle =", np.degrees(phi_lower), "deg")
print("cone center angle =", np.degrees(cone_center), "deg")
print("cone half-opening angle =", np.degrees(cone_half_angle), "deg")

for target_r in [3.0, r_acc, 5.0, 10.0]:
    phi_upper, phi_lower, cone_center, cone_half_angle = shock_angles(rho_final, r, phi, target_r)

    print("r =", target_r)
    print("upper =", np.degrees(phi_upper))
    print("lower =", np.degrees(phi_lower))
    print("center =", np.degrees(cone_center))
    print("half opening =", np.degrees(cone_half_angle))
    print()

inner_region = r <= 4.0

upper_phi = np.sin(phi) > 0.0
lower_phi = np.sin(phi) < 0.0

p_upper = np.mean(p_final[inner_region][:, upper_phi])
p_lower = np.mean(p_final[inner_region][:, lower_phi])

p_asymmetry = (p_upper - p_lower)/(0.5*(p_upper + p_lower))

print()
print("Inner pressure asymmetry")
print("mean p upper =", p_upper)
print("mean p lower =", p_lower)
print("relative pressure asymmetry =", p_asymmetry)

v_square_final = R*R*vr_final*vr_final/Triangle_R + Sumsymbol_R*vphi_final*vphi_final/(R*R)

W_final = 1.0/np.sqrt(1.0 - v_square_final)

h_final = 1 + (Gamma/(Gamma - 1))*(p_final/rho_final)
cs_final = np.sqrt(Gamma*p_final/(rho_final*h_final))

vr_physical_final = R*vr_final/np.sqrt(Triangle_R)
vphi_physical_final = np.sqrt(Sumsymbol_R)*vphi_final/R

vx_final = vr_physical_final*np.cos(PHI) - vphi_physical_final*np.sin(PHI)
vy_final = vr_physical_final*np.sin(PHI) + vphi_physical_final*np.cos(PHI)

speed_final = np.sqrt(v_square_final)

D_inner = U[0, 2, 2:-2]
mdot_final = -dphi*np.sum(r[0]*r[0]*D_inner*vr_final[0])

time_history_array = np.array(time_history)
mdot_acc_history_array = np.array(mdot_acc_history)

steady_mask = time_history_array >= simulation_time - 200.0

mdot_mean = np.mean(mdot_acc_history_array[steady_mask])
mdot_std = np.std(mdot_acc_history_array[steady_mask])

prdot_mean = np.mean(prdot_history[steady_mask])
prdot_std = np.std(prdot_history[steady_mask])

pphidot_mean = np.mean(pphidot_history[steady_mask])
pphidot_std = np.std(pphidot_history[steady_mask])

shock_center_mean = np.mean(shock_center_history[steady_mask])
shock_center_std = np.std(shock_center_history[steady_mask])

shock_half_angle_mean = np.mean(shock_half_angle_history[steady_mask])
shock_half_angle_std = np.std(shock_half_angle_history[steady_mask])

pressure_asymmetry_mean = np.mean(pressure_asymmetry_history[steady_mask])
pressure_asymmetry_std = np.std(pressure_asymmetry_history[steady_mask])

mdot_final_values = np.array([
    mdot_inner_history[-1],
    mdot_acc_history[-1],
    mdot_10_history[-1],
    mdot_20_history[-1]
])

mdot_radial_error = (np.max(mdot_final_values) - np.min(mdot_final_values))/np.mean(mdot_final_values)

print()
print("========== FINAL DIAGNOSTICS ==========")
print()
print("Stationary interval =", simulation_time - 200.0, "to", simulation_time)
print()
print("Mass accretion")
print("mean mdot =", mdot_mean)
print("std mdot =", mdot_std)
print("relative std =", mdot_std/np.abs(mdot_mean))
print("normalized mean mdot =", mdot_mean/mdot_canonical)
print("radial consistency error =", mdot_radial_error)
print()
print("Radial momentum accretion")
print("mean =", prdot_mean)
print("std =", prdot_std)
print("relative std =", prdot_std/np.abs(prdot_mean))
print()
print("Angular momentum accretion")
print("mean =", pphidot_mean)
print("std =", pphidot_std)
print("relative std =", pphidot_std/np.abs(pphidot_mean))
print("specific angular momentum accretion =", pphidot_mean/mdot_mean)
print()
print("Shock geometry at r_acc")
print("center =", shock_center_mean, "+/-", shock_center_std, "deg")
print("half-opening =", shock_half_angle_mean, "+/-", shock_half_angle_std, "deg")
print()
print("Pressure asymmetry")
print("mean =", pressure_asymmetry_mean)
print("std =", pressure_asymmetry_std)
print()
print("rho min/max =", np.min(rho_final), np.max(rho_final))
print("p min/max =", np.min(p_final), np.max(p_final))
print("W max =", np.max(W_final))
print("log rho min =", np.min(np.log10(rho_final/rhoinf)))
print("log rho max =", np.max(np.log10(rho_final/rhoinf)))


save_dir = Path(r"D:\OTO\Невидяни\Статии за дипломна\picktures\ConvergenceTests")
save_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    save_dir/"model4_histories.npz",
    time=time_history,
    mdot_inner=mdot_inner_history,
    mdot_acc=mdot_acc_history,
    mdot_10=mdot_10_history,
    mdot_20=mdot_20_history,
    mdot_normalized=mdot_normalized_history,
    prdot=prdot_history,
    prdot_surface=prdot_surface_history,
    prdot_source=prdot_source_history,
    pphidot=pphidot_history,
    pphidot_surface=pphidot_surface_history,
    pphidot_source=pphidot_source_history,
    rho_max=rho_max_history,
    rho_min=rho_min_history,
    p_max=p_max_history,
    p_min=p_min_history,
    W_max=W_max_history,
    shock_center=shock_center_history,
    shock_half_angle=shock_half_angle_history,
    pressure_asymmetry=pressure_asymmetry_history
)

U_final = U[:, 2:-2, 2:-2]

np.savez_compressed(
    save_dir/"model4_final_state.npz",
    r=r,
    phi=phi,
    rho=rho_final,
    p=p_final,
    vr=vr_final,
    vphi=vphi_final,
    U=U_final,
    M=M,
    a=a,
    Gamma=Gamma,
    Mach_inf=Mach_inf,
    vinf=vinf,
    csinf=csinf,
    rhoinf=rhoinf,
    pinf=pinf,
    r_min=r_min,
    r_max=r_max,
    r_plus=r_plus,
    r_acc=r_acc,
    simulation_time=simulation_time,
    Nr=Nr,
    Nphi=Nphi
)

summary_path = save_dir/"model4_summary.txt"

with open(summary_path, "w") as file:
    file.write(f"Model 4\n")
    file.write(f"M = {M}\n")
    file.write(f"a = {a}\n")
    file.write(f"Gamma = {Gamma}\n")
    file.write(f"Mach_inf = {Mach_inf}\n")
    file.write(f"vinf = {vinf}\n")
    file.write(f"csinf = {csinf}\n")
    file.write(f"Nr = {Nr}\n")
    file.write(f"Nphi = {Nphi}\n")
    file.write(f"Final time = {simulation_time}\n")
    file.write(f"Mean mdot = {mdot_mean}\n")
    file.write(f"Mean prdot = {prdot_mean}\n")
    file.write(f"Mean pphidot = {pphidot_mean}\n")
    file.write(f"Shock center = {shock_center_mean}\n")
    file.write(f"Shock half angle = {shock_half_angle_mean}\n")
    file.write(f"Mean mdot = {mdot_mean}\n")
    file.write(f"Std mdot = {mdot_std}\n")
    file.write(f"Normalized mean mdot = {mdot_mean/mdot_canonical}\n")
    file.write(f"Radial consistency error = {mdot_radial_error}\n")

    file.write(f"Mean prdot = {prdot_mean}\n")
    file.write(f"Std prdot = {prdot_std}\n")

    file.write(f"Mean pphidot = {pphidot_mean}\n")
    file.write(f"Std pphidot = {pphidot_std}\n")
    file.write(f"Specific angular momentum accretion = {pphidot_mean/mdot_mean}\n")

    file.write(f"Shock center = {shock_center_mean}\n")
    file.write(f"Shock center std = {shock_center_std}\n")
    file.write(f"Shock half angle = {shock_half_angle_mean}\n")
    file.write(f"Shock half angle std = {shock_half_angle_std}\n")

    file.write(f"Pressure asymmetry = {pressure_asymmetry_mean}\n")
    file.write(f"Pressure asymmetry std = {pressure_asymmetry_std}\n")

    file.write(f"rho min = {np.min(rho_final)}\n")
    file.write(f"rho max = {np.max(rho_final)}\n")
    file.write(f"log rho min = {np.min(np.log10(rho_final/rhoinf))}\n")
    file.write(f"log rho max = {np.max(np.log10(rho_final/rhoinf))}\n")

print()
print("Data saved to:")
print(save_dir)

























plt.figure(figsize=(8, 5))

plt.plot(time_history, mdot_inner_history, label="r_inner")
plt.plot(time_history, mdot_acc_history, label="r_acc")
plt.plot(time_history, mdot_10_history, label="r=10")
plt.plot(time_history, mdot_20_history, label="r=20")

plt.xlabel("t / M")
plt.ylabel("Mass accretion rate")
plt.title("Mass accretion rate at different radii")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

log_rho = np.log10(rho_final/rhoinf)

fig, ax = plt.subplots(figsize=(8, 8))

contour = ax.contour(X, Y, log_rho, levels=20)

ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, linestyle="--"))

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title("Model 4: log10(rho/rho_inf)")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

ax.contour(X, Y, log_rho, levels=20)

ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, linestyle="--"))

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title("Model 4: inner flow")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_history_array, mdot_normalized_history)

plt.xlabel("t / M")
plt.ylabel("mdot / mdot_canonical")
plt.title("Normalized mass accretion rate")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_history_array, prdot_history)

plt.xlabel("t / M")
plt.ylabel("Radial momentum rate")
plt.title("Radial momentum accretion rate")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_history_array, pphidot_history)

plt.xlabel("t / M")
plt.ylabel("Angular momentum rate")
plt.title("Angular momentum accretion rate")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_history_array, shock_center_history)

plt.xlabel("t / M")
plt.ylabel("Shock center angle [deg]")
plt.title("Shock cone deflection at r_acc")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_history_array, shock_half_angle_history)

plt.xlabel("t / M")
plt.ylabel("Half-opening angle [deg]")
plt.title("Shock cone opening angle at r_acc")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_history_array, pressure_asymmetry_history)

plt.xlabel("t / M")
plt.ylabel("Relative pressure asymmetry")
plt.title("Inner pressure asymmetry")
plt.grid()
plt.tight_layout()
plt.show()

log_p = np.log10(p_final/pinf)

fig, ax = plt.subplots(figsize=(8, 8))

contour = ax.contourf(X, Y, log_p, levels=100)

plt.colorbar(contour, ax=ax, label="log10(p/p_inf)")

ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, linestyle="--"))

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title("Model 4: pressure")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

contour = ax.contourf(X, Y, W_final, levels=100)

plt.colorbar(contour, ax=ax, label="W")

ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, linestyle="--"))

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title("Model 4: Lorentz factor")

plt.tight_layout()
plt.show()

K_final = p_final/rho_final**Gamma
K_inf = pinf/rhoinf**Gamma

entropy_proxy = np.log10(K_final/K_inf)

fig, ax = plt.subplots(figsize=(8, 8))

contour = ax.contourf(X, Y, entropy_proxy, levels=100)

plt.colorbar(contour, ax=ax, label="log10(K/K_inf)")

ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, linestyle="--"))

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title("Model 4: entropy proxy")

plt.tight_layout()
plt.show()

speed_final = np.sqrt(v_square_final)
mach_local = speed_final/cs_final

fig, ax = plt.subplots(figsize=(8, 8))

contour = ax.contourf(X, Y, mach_local, levels=100)

plt.colorbar(contour, ax=ax, label="v/c_s")

ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, linestyle="--"))

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title("Model 4: local speed / sound speed")

plt.tight_layout()
plt.show()