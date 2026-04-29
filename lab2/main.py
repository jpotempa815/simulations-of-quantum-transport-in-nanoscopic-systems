import numpy as np 
import matplotlib.pyplot as plt

#conv 
a0 = 0.0529 #bohr radius
Ha = 27.211 #hartree 
h_bar = 1.0 #in units of h_bar 
m0 = 1.0 #electron mass in units of m0

#width of AlGaAs
d_algaas = 5/a0 #bohr units
Lx = 15/a0 #bohr units - length of the system
Nx = 101 #number of intervals
Ny = 51
def z_vals(Lx, Nx):
    return np.linspace(0, Lx, Nx), Lx/(Nx-1)
z_val, dz = z_vals(Lx, Nx)

#effective mass of GaAs
m_eff_gaas = 0.063 * m0 
#composition of AlGaAs
x=0.3
m_eff_algaas = (0.063 + 0.083*x) * m0

#effective mass and potential in each interval
m_eff = np.ones(Nx) * m_eff_gaas
U_val = np.zeros(Nx)

#square barrier of width d_algaas in the middle
idx_start = int((Lx - d_algaas)/2 / dz)
idx_end = idx_start + int(d_algaas / dz)

m_eff[idx_start:idx_end] = m_eff_algaas

#barrier height
U = 0.27/Ha #hartrees
U_val[idx_start:idx_end] = U

#energy (skip the zero value)
E_arr = np.linspace(0.001/Ha, 1/Ha, 200)

def calculate_spectra(m_eff_array):
    T_arr = []
    R_arr = []
    for E in E_arr:
        k_val = 1/(h_bar)*np.lib.scimath.sqrt(2*m_eff_array*(E - U_val))
        
        M = np.eye(2, dtype=complex)
        for n in range(0, Nx-1):
            mult1 = (k_val[n+1]*m_eff_array[n])/(k_val[n]*m_eff_array[n+1])
            M_11 = 1/2 * (1 + mult1) * np.exp(1j*(k_val[n+1] - k_val[n])*z_val[n])
            M_12 = 1/2 * (1 - mult1) * np.exp(-1j*(k_val[n+1] + k_val[n])*z_val[n])
            M_21 = 1/2 * (1 - mult1) * np.exp(1j*(k_val[n+1] + k_val[n])*z_val[n])
            M_22 = 1/2 * (1 + mult1) * np.exp(-1j*(k_val[n+1] - k_val[n])*z_val[n])
            M_n = np.array([[M_11, M_12],[M_21, M_22]])
            
            M = M @ M_n
            
        T = (k_val[-1]/k_val[0]) * (m_eff_array[0]/m_eff_array[-1]) * 1/(np.abs(M[0,0]))**2
        R = (np.abs(M[1,0]))**2/(np.abs(M[0,0]))**2
        T_arr.append(np.real(T))
        R_arr.append(np.real(R))
    return T_arr, R_arr

T_arr, R_arr = calculate_spectra(m_eff)

plt.plot(E_arr*Ha, T_arr, label="Transmission")
plt.plot(E_arr*Ha, R_arr, label="Reflection")
plt.xlabel("Energy [eV]")
plt.ylabel("Probability")
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("onebarrier.png", dpi=300)
plt.show()

#m_eff constant
m_eff_constant = np.ones(Nx) * m_eff_gaas
T_arr_const, R_arr_const = calculate_spectra(m_eff_constant)

plt.plot(E_arr*Ha, T_arr_const, label="Transmission")
plt.plot(E_arr*Ha, R_arr_const, label="Reflection")
plt.xlabel("Energy [eV]")
plt.ylabel("Probability")
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("onebarrier_constant.png", dpi=300)
plt.show()

#two barriers 
Lx = 20/a0
z_val, dz = z_vals(Lx, Nx)

#two barriers with width d_algaas
sep_barr = 3/a0 #distance between barriers
idx_start1 = int((Lx/2 - d_algaas - sep_barr/2)/dz)
idx_end1 = int((Lx/2 - sep_barr/2)/dz)
idx_start2 = int((Lx/2 + sep_barr/2)/dz)
idx_end2 = int((Lx/2 + d_algaas + sep_barr/2)/dz)

m_eff = np.ones(Nx) * m_eff_gaas
U_val = np.zeros(Nx)

m_eff[idx_start1:idx_end1] = m_eff_algaas
m_eff[idx_start2:idx_end2] = m_eff_algaas

U_val[idx_start1:idx_end1] = U
U_val[idx_start2:idx_end2] = U

E_arr_high_res = np.linspace(0.001/Ha, 1/Ha, 5000)

E_arr_original = E_arr
E_arr = E_arr_high_res

T_arr, R_arr = calculate_spectra(m_eff)

E_arr = E_arr_original

plt.plot(E_arr_high_res*Ha, T_arr, label="Transmission")
plt.plot(E_arr_high_res*Ha, R_arr, label="Reflection")
plt.xlabel("Energy [eV]")
plt.ylabel("Probability")
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("twobarriers.png", dpi=300)
plt.show()


#current-voltage characteristic 

V_bias_arr = np.arange(0,0.5,0.01)/Ha

mi_s = mi_d = 87 #meV
mi_s /= 1e3*Ha
mi_d /= 1e3*Ha

Temp = 77 #K
kB = 8.617e-5/Ha #hartree/K
e = 1. #atomic units charge

#base potential (zero bias)
U_val_0 = np.copy(U_val)

#tsu-esaki
def tsu_esaki(V_bias_val):
    global U_val
    #voltage drop across
    U_val = U_val_0 - V_bias_val * (z_val / Lx)
    
    #transmission
    T_arr_bias, _ = calculate_spectra(m_eff)
    T_arr_bias = np.array(T_arr_bias)
    

    term1 = np.exp( (mi_s - E_arr) / (kB*Temp))
    term2 = np.exp( (mi_d - e*V_bias_val - E_arr) / (kB*Temp))

    ln_term = np.log((1+term1)/(1+term2))

    prefactor = e * m_eff_gaas * (kB*Temp) / (2 * np.pi**2 * h_bar**2)
    
    integrand = prefactor * T_arr_bias * ln_term
    
    return np.trapezoid(integrand, E_arr)

J_arr = []
for V_b in V_bias_arr:
    J_arr.append(tsu_esaki(V_b))

#reset
U_val = U_val_0

plt.figure(figsize=(8, 5))
plt.plot(V_bias_arr * Ha, J_arr, color='blue', linewidth=2)
plt.xlabel("Bias Voltage [V]")
plt.ylabel("Current Density [a.u.]")
# plt.title("Current-Voltage Characteristic (Tsu-Esaki)")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig("iv_curve.png", dpi=300)
plt.show()

#quantum point contact
#gate voltage
V_g = 4/Ha
epsilon = 13.6
W = 50/a0
L = 100/a0
d = 3/a0

def f(u, v, vg):
    prefactor = e*vg/(2*np.pi*epsilon)
    tan = (u*v/(d*np.sqrt(d**2 + u**2 + v**2)))
    arc = np.arctan(tan)
    return prefactor * arc

def f_inf(u, vg):
    prefactor = e*vg/(2*np.pi*epsilon)
    return prefactor * np.arctan(u/d)

def V(x, y, vg):
    l = 0.3 * L
    r = 0.7 * L
    b = 0.2 * W
    t = 0.8 * W
    
    v_strip = 2 * f_inf(x-l, vg) + 2 * f_inf(r-x, vg)
    v_gap = f(x-l, y-b, vg) + f(x-l, t-y, vg) + f(r-x, y-b, vg) + f(r-x, t-y, vg)
    return v_strip - v_gap

x_val, dx = np.linspace(0, L, Nx, retstep=True)
y_val, dy = np.linspace(0, W, Ny, retstep=True)

def get_En_x(vg):
    En_x = np.zeros((Nx, 5))
    diag_kinetic = 1.0 / (2 * m_eff_gaas * dy**2) * 2
    off_diag_kinetic = -1.0 / (2 * m_eff_gaas * dy**2)
    
    for i, x in enumerate(x_val):
        H = np.zeros((Ny-2, Ny-2))
        for j in range(Ny-2):
            y = y_val[j+1]
            H[j, j] = diag_kinetic + V(x, y, vg)
            if j > 0:
                H[j, j-1] = off_diag_kinetic
            if j < Ny-3:
                H[j, j+1] = off_diag_kinetic
                
        evals = np.linalg.eigvalsh(H)
        En_x[i, :] = evals[:5]
    return En_x

En_x_4V = get_En_x(V_g)

plt.figure(figsize=(8, 5))
for n in range(5):
    plt.plot(x_val*a0, En_x_4V[:, n]*Ha, label=f'n={n+1}')
plt.xlabel("x [nm]")
plt.ylabel("Energy [eV]")
# plt.title("Effective Potential $E_n(x)$")
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig("qpc_effective_potential.png", dpi=300)
plt.show()

def calculate_transmission_qpc(U_profile, E_array):
    T_arr = []
    m_eff_array = np.ones(Nx) * m_eff_gaas
    for E in E_array:
        k_val = 1/(h_bar)*np.lib.scimath.sqrt(2*m_eff_array*(E - U_profile))
        
        M = np.eye(2, dtype=complex)
        for n in range(0, Nx-1):
            if k_val[n] == 0:
                k_val_n = 1e-10 + 1e-10j
            else:
                k_val_n = k_val[n]
                
            if k_val[n+1] == 0:
                k_val_np1 = 1e-10 + 1e-10j
            else:
                k_val_np1 = k_val[n+1]
                
            mult1 = (k_val_np1*m_eff_array[n])/(k_val_n*m_eff_array[n+1])
            M_11 = 1/2 * (1 + mult1) * np.exp(1j*(k_val_np1 - k_val_n)*x_val[n])
            M_12 = 1/2 * (1 - mult1) * np.exp(-1j*(k_val_np1 + k_val_n)*x_val[n])
            M_21 = 1/2 * (1 - mult1) * np.exp(1j*(k_val_np1 + k_val_n)*x_val[n])
            M_22 = 1/2 * (1 + mult1) * np.exp(-1j*(k_val_np1 - k_val_n)*x_val[n])
            M_n = np.array([[M_11, M_12],[M_21, M_22]])
            
            M = M @ M_n
            
        if np.real(k_val[0]) == 0 or np.real(k_val[-1]) == 0 or M[0,0] == 0:
            T = 0
        else:
            T = (np.real(k_val[-1])/np.real(k_val[0])) * (m_eff_array[0]/m_eff_array[-1]) * 1/(np.abs(M[0,0]))**2
        T_arr.append(np.real(T))
    return T_arr

E_qpc_eV = np.linspace(0.001, 0.2, 200)
E_qpc = E_qpc_eV / Ha

G_E = np.zeros_like(E_qpc)

for n in range(5):
    T_n = calculate_transmission_qpc(En_x_4V[:, n], E_qpc)
    G_E += np.array(T_n)

plt.figure(figsize=(8, 5))
plt.plot(E_qpc_eV, G_E, color='purple', linewidth=2)
plt.xlabel("Energy [eV]")
plt.ylabel("Conductance [$2e^2/h$]")
# plt.title("Conductance $G(E)$ vs Energy")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig("qpc_conductance_energy.png", dpi=300)
plt.show()

Vg_arr_V = np.linspace(0, 25, 100)
Vg_arr = Vg_arr_V / Ha

E1 = 50e-3 / Ha
E2 = 100e-3 / Ha

G_Vg_E1 = []
G_Vg_E2 = []

for vg in Vg_arr:
    En_x_vg = get_En_x(vg)
    
    g1 = 0
    g2 = 0
    for n in range(5):
        T_n_E1 = calculate_transmission_qpc(En_x_vg[:, n], [E1])[0]
        T_n_E2 = calculate_transmission_qpc(En_x_vg[:, n], [E2])[0]
        g1 += T_n_E1
        g2 += T_n_E2
        
    G_Vg_E1.append(g1)
    G_Vg_E2.append(g2)

plt.figure(figsize=(8, 5))
plt.plot(Vg_arr_V, G_Vg_E1, label="E = 50 meV", color='blue', linewidth=2)
plt.plot(Vg_arr_V, G_Vg_E2, label="E = 100 meV", color='red', linewidth=2)
plt.xlabel("Gate Voltage $V_g$ [V]")
plt.ylabel("Conductance [$2e^2/h$]")
# plt.title("Conductance $G(V_g)$ vs Gate Voltage")
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig("qpc_conductance_vg.png", dpi=300)
plt.show()

