import numpy as np
import netket as nk
import matplotlib.pyplot as plt
import flax.serialization as serialization
from scipy.ndimage import uniform_filter1d
import os
from tqdm import tqdm

from netket.operator.spin import sigmax, sigmaz

J_ZZ = -1.0
J_XX = 0.0 
h_x = -0.5
h_z = 0.0

save = True
jump_temperatures = False
derivative = False
fontsize = 14

N_list = [4]
temperatures = np.linspace(0.05, 4.0, 200)

def gibbs_thermodynamic_limit(T_array, J, h):
    """
    Calcula observables en el límite termodinámico para el modelo de Ising
    con campo transversal usando la solución analítica.
    """
    Nk = 4000
    k = np.linspace(0, np.pi, Nk)
    dk = np.pi / Nk
    energies_td = []
    sx_td = []
    szz_td = [] 
    for T in T_array:
        beta = 1.0 / T
        eps_k = 2 * np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))
        tanh_term = np.tanh(beta * eps_k / 2)
        # Energía por sitio 
        e = - np.sum((eps_k / 2) * tanh_term) * dk / np.pi
        # <σx>
        mx = np.sum((h - J * np.cos(k)) / eps_k * tanh_term) * dk / np.pi
        # <σz σz>
        corr = np.sum((J - h * np.cos(k))/eps_k* tanh_term) * dk / np.pi
        energies_td.append(e)
        szz_td.append(corr*2)
        sx_td.append(mx*2)   
    return np.array(energies_td), np.array(sx_td), np.array(szz_td)

def canonical_expectation(O, T, eigvals, eigvecs):
    """
    Expectation value <O> en el ensemble canónico a temperatura T.

    Parámetros
    ----------
    O : NetKet operator
    T : float
    eigvals : ndarray
    eigvecs : ndarray

    Returns
    -------
    float
    """

    # convertir operador a matriz densa
    O_matrix = O.to_dense()

    # poblaciones de Boltzmann (estables numéricamente)
    log_pops = -eigvals / T
    log_pops -= np.max(log_pops)

    pops = np.exp(log_pops)
    pops /= pops.sum()

    # <n|O|n>
    O_diag = np.einsum(
        "ij,ji->i",
        np.conj(eigvecs.T),
        O_matrix @ eigvecs,
    )

    # promedio térmico
    return np.sum(pops * O_diag).real

def renyi_exact_weights(T, eigvals, tol=1e-12):
    """
    Devuelve los pesos exactos w_k del MRE (alpha=2).
    """

    from scipy.optimize import brentq

    eigvals = np.array(eigvals, dtype=float)

    def mre_weights(E_perp):
        w = np.maximum(0.0, E_perp - eigvals)
        Z = w.sum()
        if Z < tol:
            return None
        return w / Z

    def f_root(E_perp):
        w = mre_weights(E_perp)
        if w is None:
            return E_perp
        E_bar = np.dot(w, eigvals)
        return E_perp - (2 * T + E_bar)

    E_min = eigvals.min()
    E_max = eigvals.max() + 4 * T

    try:
        E_perp = brentq(f_root, E_min, E_max, xtol=1e-12)
        w = mre_weights(E_perp)

    except ValueError:
        # ground state
        w = np.zeros(len(eigvals))
        w[0] = 1.0

    return w

def renyi_exact_expectation(O, T, eigvals, eigvecs):
    """
    Expectation value <O> en el MRE exacto (alpha=2).
    """

    w = renyi_exact_weights(T, eigvals)

    # operador en base de energía
    O_matrix = O.to_dense()

    O_diag = np.einsum(
        "ij,ji->i",
        np.conj(eigvecs.T),
        O_matrix @ eigvecs,
    ).real

    return float(np.dot(w, O_diag))

def hamiltonian_system_and_extended(J_ZZ, J_XX, h_x, h_z, N, hi_system, hi_extended):
    H_system = 0
    H_extended = 0
    for i in range(N):
        H_system+= h_x * sigmax(hi_system, i)
        H_system+= h_z * sigmaz(hi_system, i)
        H_system += J_ZZ * sigmaz(hi_system, i) @ sigmaz(hi_system, (i + 1) % N)
        H_system += J_XX * sigmax(hi_system, i) @ sigmax(hi_system, (i + 1) % N)
        H_extended += h_x * sigmax(hi_extended, i)
        H_extended += h_z * sigmaz(hi_extended, i)
        H_extended += J_ZZ * sigmaz(hi_extended, i) @ sigmaz(hi_extended, (i + 1) % N)
        H_extended += J_XX * sigmax(hi_extended, i) @ sigmax(hi_extended, (i + 1) % N)
    return H_system, H_extended

def renyi_exact(T, eigvals, tol=1e-12):
    from scipy.optimize import brentq

    eigvals = np.array(eigvals, dtype=float)

    def mre_weights(E_perp):
        w = np.maximum(0.0, E_perp - eigvals)
        Z = w.sum()
        if Z < tol:
            return None
        return w / Z

    def f_root(E_perp):
        w = mre_weights(E_perp)
        if w is None:
            return E_perp
        E_bar = np.dot(w, eigvals)
        return E_perp - (2 * T + E_bar)

    E_min = eigvals.min()
    E_max = eigvals.max() + 4 * T

    try:
        E_perp = brentq(f_root, E_min, E_max, xtol=1e-12)
    except ValueError:
        w = np.zeros(len(eigvals))
        w[0] = 1.0
    else:
        w = mre_weights(E_perp)

    E = np.dot(w, eigvals)
    purity = np.sum(w**2)
    S2 = -np.log(purity)
    F = E - T * S2

    return F, E, S2

print("Calculando límite termodinámico de Gibbs...")
energies_td, sx_td, szz_td = gibbs_thermodynamic_limit(temperatures, -J_ZZ, -h_x)
results = {}

for N in N_list:
    print(f"\nCalculando MRE N = {N}...")
    hi_system = nk.hilbert.Spin(s=1/2, N=N)
    hi_extended = nk.hilbert.Spin(s=1/2, N=N)  
    H_system, H_extended = hamiltonian_system_and_extended(J_ZZ, 0, h_x, 0, N, hi_system, hi_extended)
    
    O_x = sum(sigmax(hi_system, i) for i in range(N)) / N
    O_zz = sum(sigmaz(hi_system, i) @ sigmaz(hi_system, (i+1) % N) 
               for i in range(N)) / N
    
    H_matrix = H_system.to_dense()
    eigvals, eigvecs = np.linalg.eigh(H_matrix)
    
    energies_renyi = []
    sx_renyi = []
    szz_renyi = []
    energies_gibbs = []
    sx_gibbs = []
    szz_gibbs = []
    energy_density = []
    
    for T in tqdm(temperatures, desc=f"Temperaturas N={N}"):
        F, E_R, S2_R = renyi_exact(T, eigvals) 
        energy_density.append(E_R / N)
        
        sx_renyi.append(renyi_exact_expectation(O_x, T, eigvals, eigvecs))
        szz_renyi.append(renyi_exact_expectation(O_zz, T, eigvals, eigvecs))
        
        sx_gibbs.append(canonical_expectation(O_x, T, eigvals, eigvecs))
        szz_gibbs.append(canonical_expectation(O_zz, T, eigvals, eigvecs))
    
    results[N] = {
        'energy_density': np.array(energy_density),
        'sx_renyi': np.array(sx_renyi),
        'szz_renyi': np.array(szz_renyi),
    }

# Crear la figura
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(N_list)))

ax1 = axes[0]
for idx, N in enumerate(N_list):
    color = colors[idx]
    ax1.plot(results[N]['energy_density'], results[N]['sx_renyi'], 
             '-', color=color, linewidth=2, label=f'N={N} (MRE)')
ax1.plot(energies_td, sx_td, 'k-', linewidth=2.5, label=r'Gibbs ($N\to\infty$)')
ax1.set_xlabel(r'$\langle H \rangle / N$')
ax1.set_ylabel(r'$\langle \sigma^x \rangle$')
ax1.set_title(r'(a) $\langle\sigma^x\rangle$')
ax1.legend(fontsize=fontsize-1)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for idx, N in enumerate(N_list):
    color = colors[idx]
    ax2.plot(results[N]['energy_density'], results[N]['szz_renyi'], 
             '-', color=color, linewidth=2, label=f'N={N} (MRE)')
ax2.plot(energies_td, szz_td, 'k-', linewidth=2.5, label=r'Gibbs ($N\to\infty$)')
ax2.set_xlabel(r'$\langle H \rangle / N$')
ax2.set_ylabel(r'$\Gamma^{z,z}$')
ax2.set_title(r'(b) $\Gamma^{z,z}$')
ax2.legend(fontsize=fontsize-1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Guardar la figura (sin mostrar)
if save:
    # Usar ruta absoluta o relativa desde el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir, "..", "plots", "MRE")
    
    print(f"Intentando guardar en: {base}")
    
    try:
        os.makedirs(base, exist_ok=True)
        
        # Guardar como PDF
        pdf_path = os.path.join(base, "convergence_MRE_to_Gibbs_2.pdf")
        plt.savefig(pdf_path, bbox_inches="tight", dpi=300, format='pdf')
        print(f"✓ PDF guardado: {pdf_path}")
        
        # Guardar como PNG
        png_path = os.path.join(base, "convergence_MRE_to_Gibbs_2.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300, format='png')
        print(f"✓ PNG guardado: {png_path}")
        
        # Verificar que los archivos existen
        if os.path.exists(pdf_path):
            print(f"✓ Verificado: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
        else:
            print(f"✗ Error: No se creó {pdf_path}")
            
    except Exception as e:
        print(f"Error al guardar: {e}")
        # Intentar guardar en el directorio actual como fallback
        print("Intentando guardar en directorio actual...")
        plt.savefig("convergence_MRE_to_Gibbs_2.pdf", bbox_inches="tight", dpi=300)
        print("✓ Guardado en directorio actual: convergence_MRE_to_Gibbs_2.pdf")
else:
    plt.show()

# Cerrar la figura para liberar memoria
plt.close(fig)