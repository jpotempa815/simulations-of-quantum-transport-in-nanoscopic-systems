import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import eigh

n=9
N=n*n
d_x = 1
a = d_x *(n-1)/2
h = 1 #h-bar

#conv
hartree_to_mev = 27211.6
bohr_rad = 0.0529 #nm

w_x = 80 / hartree_to_mev
w_y = 200 / hartree_to_mev

m_eff = 0.24

alpha_x = 1 / (m_eff * w_x) * bohr_rad**2
alpha_y = 1 / (m_eff * w_y) * bohr_rad**2

#task one
def xy_val(d_x, k):
    i = k // n
    j = k % n
    x = -a + d_x*i
    y = -a + d_x*j
    return x, y

def array(d_x):
    x_k, y_k = [], []
    for k in range(0, N-1):
        x, y = xy_val(d_x,k)
        x_k.append(x)
        y_k.append(y)
    return x_k, y_k

def gaussian(x, y, d_x, k):
    x_k, y_k = xy_val(d_x,k)
    psi_k1 = (alpha_x*np.pi)**(-1/4)
    psi_k2 = np.exp(-(x - x_k)**2/(2*alpha_x))
    psi_k3 = (alpha_y*np.pi)**(-1/4)
    psi_k4 = np.exp(-(y - y_k)**2/(2*alpha_y))
    return psi_k1 * psi_k2 * psi_k3 * psi_k4

#task one plot
x_vals = np.arange(-a, a, 0.1)
y_vals = np.arange(-a, a, 0.1)
X, Y = np.meshgrid(x_vals, y_vals)

figure, axes = plt.subplots(1, 3, figsize=(12, 5))
for idx, k in enumerate([0, 8, 9]):
    im = axes[idx].imshow(gaussian(X, Y, d_x, k), cmap="plasma", extent=[-a, a, -a, a], origin='lower', vmin=0, vmax=None)
    axes[idx].set_title(f"k = {k}")
    axes[idx].set_xlabel("x [nm]")
    axes[idx].set_ylabel("y [nm]")
    figure.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
plt.tight_layout(w_pad=3.0)
plt.savefig("taskone.png", dpi=300)
plt.show()

#task two
def S(k,l):
    x_k, y_k = xy_val(d_x, k)
    x_l, y_l = xy_val(d_x, l)
    return np.exp(-(x_k-x_l)**2/(4*alpha_x) - (y_k-y_l)**2/(4*alpha_y))

def K(k,l):
    x_k, y_k = xy_val(d_x, k)
    x_l, y_l = xy_val(d_x, l)
    x_k_au, y_k_au = x_k / bohr_rad, y_k / bohr_rad
    x_l_au, y_l_au = x_l / bohr_rad, y_l / bohr_rad
    alpha_x_au = alpha_x / bohr_rad**2
    alpha_y_au = alpha_y / bohr_rad**2
    return -1/(2*m_eff)*(((x_k_au - x_l_au)**2 - 2*alpha_x_au)/(4*alpha_x_au**2)+((y_k_au - y_l_au)**2 - 2*alpha_y_au)/(4*alpha_y_au**2))*S(k,l)

def V(k,l):
    x_k, y_k = xy_val(d_x, k)
    x_l, y_l = xy_val(d_x, l)
    x_k_au, y_k_au = x_k / bohr_rad, y_k / bohr_rad
    x_l_au, y_l_au = x_l / bohr_rad, y_l / bohr_rad
    alpha_x_au = alpha_x / bohr_rad**2
    alpha_y_au = alpha_y / bohr_rad**2
    return 1/2 * m_eff * ((w_x**2)*((x_k_au + x_l_au)**2 + 2*alpha_x_au)/4 + (w_y**2)*((y_k_au + y_l_au)**2 + 2*alpha_y_au)/4)*S(k,l)

def H(k,l):
    return K(k,l) + V(k,l)

#task three
def S_matrix():
    S_matrix = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            S_matrix[k,l] = S(k,l)
    return S_matrix

def H_matrix():
    H_matrix = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            H_matrix[k,l] = H(k,l)
    return H_matrix

def eigenvalues_E(range=1):
    return eigh(H_matrix(), S_matrix())[0][:range] * hartree_to_mev

def c_wavefunction():
    return eigh(H_matrix(), S_matrix())[1]

c = c_wavefunction()
def wavefunction(wave_num):
    Z = np.zeros_like(X)
    for k in range(N):
        Z += c[k, wave_num] * gaussian(X, Y, d_x, k)
    return Z**2

#task three plot - six lowest states
figure2, axes2 = plt.subplots(2, 3, figsize=(12, 12))
axes2 = axes2.flatten()
for i in range(6):
    wav_fun = wavefunction(i)
    im = axes2[i].imshow(wav_fun, cmap="plasma", extent=[-a, a, -a, a], origin='lower', vmin=0, vmax=None)
    axes2[i].set_title(f"State {i+1}")
    axes2[i].set_xlabel("x [nm]")
    axes2[i].set_ylabel("y [nm]")
    figure2.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
plt.tight_layout(w_pad=3.0)
plt.savefig(f"taskthree.png", dpi=300)
plt.show()

def set_omegas(wx_mev, wy_mev):
    global w_x, w_y, c
    w_x = wx_mev / hartree_to_mev
    w_y = wy_mev / hartree_to_mev
    c = c_wavefunction()

#task four
wx_list = np.linspace(10, 500, 20)
wy_mev = 200
energies_list = []
for wx_mev in wx_list:
    set_omegas(wx_mev, wy_mev)
    energies = eigenvalues_E(10)
    energies_list.append(energies)

energies_list = np.array(energies_list)

plt.figure(figsize=(8,6))
for i in range(10):
    plt.plot(wx_list, energies_list[:, i], 'o-', label=f'State {i+1}')

for nx in range(4):
    for ny in range(2):
        E_ana = wx_list * (nx + 0.5) + wy_mev * (ny + 0.5)
        plt.plot(wx_list, E_ana, '--', color='gray')

plt.title(r"Energies of the 10 lowest states vs $\hbar\omega_x$")
plt.xlabel(r"$\hbar\omega_x$ [meV]")
plt.ylabel("Energy [meV]")
plt.grid(True, linestyle='--')
plt.savefig("taskfour.png", dpi=300)
plt.show()

#task five
set_omegas(80, 400) # Increase wy to 400 meV so the lowest 5 states are excited only in x
figure3, axes3 = plt.subplots(2, 3, figsize=(12, 12))
axes3 = axes3.flatten()
for i in range(6):
    wav_fun = wavefunction(i)
    im = axes3[i].imshow(wav_fun, cmap="plasma", extent=[-a, a, -a, a], origin='lower', vmin=0, vmax=None)
    axes3[i].set_title(f"State {i+1}")
    axes3[i].set_xlabel("x [nm]")
    axes3[i].set_ylabel("y [nm]")
    figure3.colorbar(im, ax=axes3[i], fraction=0.046, pad=0.04)
plt.tight_layout(w_pad=3.0)
plt.savefig("taskfive.png", dpi=300)
plt.show()