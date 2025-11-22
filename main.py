import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd

# ============================================================================
# PART 1: SMOOTH KICKED ROTOR MODEL
# ============================================================================

# Parameters
K_default = 5.0      # kick strength
tau = 0.1            # kick width (smoothness parameter)
T_period = 1.0       # period between kicks


def kick_function(t):
    """
    Smooth periodic kick function (Gaussian pulses)
    Instead of delta functions, we use smooth Gaussians
    """
    t_mod = t % T_period  # time within current period
    return np.exp(-t_mod**2 / (2 * tau**2))


def kick_function_derivative(t):
    """Derivative of kick function (needed for some analyses)"""
    t_mod = t % T_period
    return -(t_mod / tau**2) * np.exp(-t_mod**2 / (2 * tau**2))


# ============================================================================
# PART 2: CLASSICAL DYNAMICS WITH ADAPTIVE RK4
# ============================================================================

def classical_hamiltonian_equations(t, state, K):
    """
    Hamilton's equations for smooth kicked rotor:
    ṗ = -∂H/∂θ = K sin(θ) f(t)
    θ̇ = ∂H/∂p = p

    state = [theta, p]
    """
    theta, p = state
    f_t = kick_function(t)

    dtheta_dt = p
    dp_dt = K * np.sin(theta) * f_t

    return np.array([dtheta_dt, dp_dt])


def rk4_step(f, t, y, h, *args):
    """
    Classic RK4 step for system dy/dt = f(t, y)
    """
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)

    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def adaptive_rk4_classical(f, t0, y0, t_end,
                           h0=0.01, tol=1e-6,
                           h_min=1e-6, h_max=0.1, *args):
    """
    Adaptive RK4 integrator using step-doubling for error control
    """
    t = t0
    y = y0.copy()

    ts = [t]
    ys = [y.copy()]
    hs = [h0]

    h = h0
    step_count = 0
    rejected_steps = 0

    print(f"Starting adaptive RK4 integration (classical)...")

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        # One big step
        y_big = rk4_step(f, t, y, h, *args)

        # Two small steps
        y_half1 = rk4_step(f, t, y, h / 2, *args)
        y_small = rk4_step(f, t + h / 2, y_half1, h / 2, *args)

        # Error estimate
        error = np.linalg.norm(y_small - y_big)

        if error < tol or h <= h_min:
            # Accept step
            t += h
            y = y_small  # Use more accurate result

            ts.append(t)
            ys.append(y.copy())
            hs.append(h)
            step_count += 1

            # Adapt step size
            if error > 1e-16:
                h_new = h * min(2.0, max(0.5, 0.9 * (tol / error) ** 0.2))
            else:
                h_new = min(2 * h, h_max)
            h = min(h_new, h_max)

            if step_count % 100 == 0:
                print(f"  [classical] Step {step_count}: t={t:.3f}, h={h:.6f}, error={error:.2e}")
        else:
            # Reject step
            rejected_steps += 1
            h = h * max(0.5, 0.9 * (tol / error) ** 0.2)
            h = max(h, h_min)

    print(f"Classical integration complete: {step_count} steps, {rejected_steps} rejected")
    return np.array(ts), np.array(ys), np.array(hs)


def simulate_classical_trajectory(theta0, p0, K, t_end, h0=0.01):
    """Simulate single classical trajectory"""
    y0 = np.array([theta0, p0])
    # NOTE: pass K as positional *args, NOT as keyword
    ts, ys, hs = adaptive_rk4_classical(
        classical_hamiltonian_equations, 0.0, y0, t_end,
        h0, 1e-6, 1e-6, 0.1, K
    )

    thetas = ys[:, 0]
    ps = ys[:, 1]

    return ts, thetas, ps, hs


def compute_lyapunov_exponent(theta0, p0, K, t_end, delta=1e-8):
    """
    Compute Lyapunov exponent by tracking nearby trajectories
    Uses adaptive RK4 for both trajectories
    """
    print("\nComputing Lyapunov exponent...")

    # Main trajectory
    y0_1 = np.array([theta0, p0])
    ts_1, ys_1, _ = adaptive_rk4_classical(
        classical_hamiltonian_equations, 0.0, y0_1, t_end,
        0.01, 1e-6, 1e-6, 0.1, K
    )

    # Perturbed trajectory
    y0_2 = np.array([theta0 + delta, p0])
    ts_2, ys_2, _ = adaptive_rk4_classical(
        classical_hamiltonian_equations, 0.0, y0_2, t_end,
        0.01, 1e-6, 1e-6, 0.1, K
    )

    # Interpolate to common time grid
    t_common = ts_1
    theta_1 = ys_1[:, 0]
    p_1 = ys_1[:, 1]
    theta_2 = np.interp(t_common, ts_2, ys_2[:, 0])
    p_2 = np.interp(t_common, ts_2, ys_2[:, 1])

    # Compute separations
    separations = np.sqrt((theta_2 - theta_1) ** 2 + (p_2 - p_1) ** 2)

    # Lyapunov exponent (skip transient)
    n_transient = len(t_common) // 4
    log_sep = np.log(separations[n_transient:] / delta)
    lambda_lyap = np.mean(log_sep) / (t_common[-1] - t_common[n_transient])

    return lambda_lyap, t_common, separations


def classical_ensemble_diffusion(K, t_end, n_ensemble=100):
    """
    Compute momentum diffusion for ensemble
    Each trajectory uses adaptive RK4
    """
    print(f"\nRunning ensemble of {n_ensemble} classical trajectories...")

    # Sample times for ensemble average
    t_sample = np.linspace(0, t_end, 100)
    p2_ensemble = np.zeros(len(t_sample))

    for i in range(n_ensemble):
        if (i + 1) % 20 == 0:
            print(f"  [ensemble] Trajectory {i + 1}/{n_ensemble}")

        theta0 = np.random.uniform(0, 2 * np.pi)
        p0 = np.random.uniform(-0.5, 0.5)

        ts, _, ps, _ = simulate_classical_trajectory(theta0, p0, K, t_end, h0=0.01)

        # Interpolate to sample times
        p_interp = np.interp(t_sample, ts, ps)
        p2_ensemble += p_interp ** 2

    p2_ensemble /= n_ensemble
    return t_sample, p2_ensemble


# ============================================================================
# PART 3: QUANTUM DYNAMICS WITH ADAPTIVE RK4
# ============================================================================

def build_hamiltonian_matrix(K, f_t, n_max):
    """
    Build time-dependent Hamiltonian matrix at time t
    H(t) = p²/2 + K cos(θ) f(t)

    In momentum basis:
    H_nm = (n²/2) δ_nm + K f(t) ⟨n|cos(θ)|m⟩

    where ⟨n|cos(θ)|m⟩ = (δ_n,m+1 + δ_n,m-1)/2
    """
    N = 2 * n_max + 1
    H = np.zeros((N, N), dtype=complex)

    n_values = np.arange(-n_max, n_max + 1)

    # Kinetic term: p²/2
    for i, n in enumerate(n_values):
        H[i, i] = n ** 2 / 2.0

    # Potential term: K cos(θ) f(t)
    # cos(θ) connects n to n±1
    for i in range(N):
        if i > 0:  # n to n-1
            H[i, i - 1] += K * f_t / 2.0
        if i < N - 1:  # n to n+1
            H[i, i + 1] += K * f_t / 2.0

    return H


def schrodinger_rhs(t, psi, K, n_max):
    """
    Right-hand side of Schrödinger equation: dψ/dt = -i H(t) ψ
    """
    f_t = kick_function(t)
    H = build_hamiltonian_matrix(K, f_t, n_max)
    return -1j * (H @ psi)


def adaptive_rk4_quantum(t0, psi0, t_end, K, n_max, h0=0.01, tol=1e-6):
    """
    Adaptive RK4 for time-dependent Schrödinger equation
    """
    t = t0
    psi = psi0.copy()

    ts = [t]
    psis = [psi.copy()]
    hs = [h0]

    h = h0
    step_count = 0

    print(f"\nStarting quantum adaptive RK4...")

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        # One big step
        k1 = schrodinger_rhs(t, psi, K, n_max)
        k2 = schrodinger_rhs(t + 0.5 * h, psi + 0.5 * h * k1, K, n_max)
        k3 = schrodinger_rhs(t + 0.5 * h, psi + 0.5 * h * k2, K, n_max)
        k4 = schrodinger_rhs(t + h, psi + h * k3, K, n_max)
        psi_big = psi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Two half steps
        k1 = schrodinger_rhs(t, psi, K, n_max)
        k2 = schrodinger_rhs(t + 0.25 * h, psi + 0.25 * h * k1, K, n_max)
        k3 = schrodinger_rhs(t + 0.25 * h, psi + 0.25 * h * k2, K, n_max)
        k4 = schrodinger_rhs(t + 0.5 * h, psi + 0.5 * h * k3, K, n_max)
        psi_half = psi + (h / 12.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        k1 = schrodinger_rhs(t + 0.5 * h, psi_half, K, n_max)
        k2 = schrodinger_rhs(t + 0.75 * h, psi_half + 0.25 * h * k1, K, n_max)
        k3 = schrodinger_rhs(t + 0.75 * h, psi_half + 0.25 * h * k2, K, n_max)
        k4 = schrodinger_rhs(t + h, psi_half + 0.5 * h * k3, K, n_max)
        psi_small = psi_half + (h / 12.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Error estimate
        error = np.linalg.norm(psi_small - psi_big)

        if error < tol:
            # Accept
            t += h
            psi = psi_small
            psi = psi / np.linalg.norm(psi)  # Renormalize

            ts.append(t)
            psis.append(psi.copy())
            hs.append(h)
            step_count += 1

            # Adapt
            if error > 1e-16:
                h = h * min(2.0, max(0.5, 0.9 * (tol / error) ** 0.2))
            else:
                h = min(2 * h, 0.1)

            if step_count % 50 == 0:
                print(f"  [quantum] Step {step_count}: t={t:.3f}, h={h:.6f}, norm={np.linalg.norm(psi):.10f}")
        else:
            # Reject
            h = h * max(0.5, 0.9 * (tol / error) ** 0.2)
            h = max(h, 1e-6)

    print(f"Quantum integration complete: {step_count} steps")
    return np.array(ts), psis, np.array(hs)


def compute_quantum_observables(ts, psis, n_max):
    """Compute quantum observables from state history"""
    n_values = np.arange(-n_max, n_max + 1)

    p2_mean = np.zeros(len(ts))
    entropy = np.zeros(len(ts))
    participation = np.zeros(len(ts))

    for i, psi in enumerate(psis):
        P = np.abs(psi) ** 2

        # <p²>
        p2_mean[i] = np.sum(n_values ** 2 * P)

        # Entropy (Shannon in momentum basis)
        P_safe = P[P > 1e-16]
        entropy[i] = -np.sum(P_safe * np.log(P_safe))

        # Participation ratio
        participation[i] = 1.0 / np.sum(P ** 2)

    return p2_mean, entropy, participation


# ============================================================================
# PART 4: SVD ANALYSIS
# ============================================================================

def svd_analysis(psis):
    """SVD of snapshot matrix"""
    X = np.array(psis).T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    d_eff = (np.sum(S) ** 2) / np.sum(S ** 2)
    return U, S, Vt, d_eff


# ============================================================================
# PART 5: MAIN SIMULATION
# ============================================================================

def run_full_simulation(K=5.0, t_end=10.0, n_max=30):
    """
    Run complete simulation with ADAPTIVE RK4 for both classical and quantum
    """
    print("=" * 70)
    print("SMOOTH KICKED ROTOR WITH ADAPTIVE RK4")
    print(f"K = {K}, t_end = {t_end}, n_max = {n_max}")
    print("=" * 70)

    # Classical single trajectory
    print("\n[1/5] Classical single trajectory...")
    ts_c, thetas_c, ps_c, hs_c = simulate_classical_trajectory(0.1, 0.0, K, t_end)

    # Lyapunov
    print("\n[2/5] Lyapunov exponent...")
    lambda_lyap, ts_lyap, seps = compute_lyapunov_exponent(0.1, 0.0, K, t_end)
    print(f"  λ = {lambda_lyap:.4f}")

    # Classical ensemble
    print("\n[3/5] Classical ensemble diffusion...")
    ts_ens, p2_classical = classical_ensemble_diffusion(K, t_end, n_ensemble=50)
    D_classical = np.polyfit(ts_ens[10:50], p2_classical[10:50], 1)[0]
    print(f"  D_classical = {D_classical:.4f}")

    # Quantum evolution
    print("\n[4/5] Quantum evolution with adaptive RK4...")
    N = 2 * n_max + 1
    psi0 = np.zeros(N, dtype=complex)
    psi0[n_max] = 1.0  # Start at n=0

    ts_q, psis_q, hs_q = adaptive_rk4_quantum(0.0, psi0, t_end, K, n_max)
    p2_quantum, entropy, participation = compute_quantum_observables(ts_q, psis_q, n_max)

    print(f"  Final <p²>_quantum = {p2_quantum[-1]:.4f}")

    # SVD
    print("\n[5/5] SVD analysis...")
    U_svd, S, Vt, d_eff = svd_analysis(psis_q)
    print(f"  d_eff = {d_eff:.2f}")

    # Save data
    print("\nSaving data...")

    pd.DataFrame({
        't': ts_ens,
        'p2_classical': p2_classical
    }).to_csv('classical_diffusion.csv', index=False)

    pd.DataFrame({
        't': ts_q,
        'p2_quantum': p2_quantum,
        'entropy': entropy,
        'participation': participation
    }).to_csv('quantum_observables.csv', index=False)

    pd.DataFrame({
        'mode': np.arange(len(S)),
        'singular_value': S
    }).to_csv('svd_spectrum.csv', index=False)

    pd.DataFrame({
        't': ts_c,
        'h_classical': hs_c
    }).to_csv('adaptive_stepsize_classical.csv', index=False)

    pd.DataFrame({
        't': ts_q,
        'h_quantum': hs_q
    }).to_csv('adaptive_stepsize_quantum.csv', index=False)

    print("✓ All data saved!")

    return {
        'classical': (ts_c, thetas_c, ps_c, hs_c, ts_ens, p2_classical),
        'quantum': (ts_q, psis_q, p2_quantum, entropy, participation, hs_q),
        'lyapunov': (lambda_lyap, ts_lyap, seps),
        'svd': (U_svd, S, Vt, d_eff),
        'params': (K, t_end, n_max)
    }


# ============================================================================
# PART 6: BUNDLE HELPERS (CLASSICAL AND QUANTUM)
# ============================================================================

def classical_trajectory_bundle(theta0, p0, K, t_end, n_traj=5, delta_theta=1e-2):
    """
    Generate a bundle of nearby classical trajectories to visualise divergence.
    All trajectories share the same p0 and have slightly shifted initial angles:
        theta0 + k * delta_theta,  k = 0, 1, ..., n_traj-1

    Returns:
        t_common: 1D array, common time grid
        P_bundle: 2D array shape (n_traj, len(t_common)) with p(t) for each traj
    """
    trajectories = []
    times_list = []

    for k in range(n_traj):
        th0 = theta0 + k * delta_theta
        ts, thetas, ps, hs = simulate_classical_trajectory(th0, p0, K, t_end, h0=0.01)
        trajectories.append(ps)
        times_list.append(ts)

    # Use the first trajectory's times as common grid and interpolate others
    t_common = times_list[0]
    P_bundle = np.zeros((n_traj, len(t_common)))

    for k in range(n_traj):
        P_bundle[k, :] = np.interp(t_common, times_list[k], trajectories[k])

    return t_common, P_bundle


def quantum_trajectory_bundle(K, t_end, n_max, initial_indices=None, h0=0.01, tol=1e-6):
    """
    Evolve several different initial quantum states (momentum eigenstates)
    and compute <p^2>(t) for each, to show localisation vs initial condition.

    initial_indices: list of momentum indices n (in [-n_max, n_max]) for which
                     we take |psi(0)> = |n>.

    Returns:
        t_common: 1D array, common time grid
        p2_bundle: 2D array shape (n_traj, len(t_common)) with <p^2>(t) curves
        initial_indices: list of n-values used
    """
    if initial_indices is None:
        initial_indices = [0, 1, 2]

    N = 2 * n_max + 1
    idx0 = n_max  # index corresponding to n = 0

    p2_list = []
    t_list = []

    for n0 in initial_indices:
        psi0 = np.zeros(N, dtype=complex)
        psi0[idx0 + n0] = 1.0  # |n0> in our indexing

        ts, psis, hs = adaptive_rk4_quantum(0.0, psi0, t_end, K, n_max, h0=h0, tol=tol)
        p2, entropy, participation = compute_quantum_observables(ts, psis, n_max)

        p2_list.append(p2)
        t_list.append(ts)

    # Use first time grid as reference and interpolate others
    t_common = t_list[0]
    n_traj = len(initial_indices)
    p2_bundle = np.zeros((n_traj, len(t_common)))

    for k in range(n_traj):
        p2_bundle[k, :] = np.interp(t_common, t_list[k], p2_list[k])

    return t_common, p2_bundle, initial_indices


# ============================================================================
# PART 7: PLOTTING – SEPARATE PUBLICATION FIGURES
# ============================================================================

def create_publication_plots(results, K=5.0, t_end=10.0, n_max=30):
    """
    Create separate, publication-style plots and save them as individual PDFs.
    """
    classical = results['classical']
    quantum = results['quantum']
    lyap_data = results['lyapunov']
    svd_data = results['svd']

    ts_c, thetas_c, ps_c, hs_c, ts_ens, p2_class = classical
    ts_q, psis_q, p2_quant, entropy, participation, hs_q = quantum
    lambda_lyap, ts_lyap, seps = lyap_data
    U_svd, S, Vt, d_eff = svd_data

    # 1) Diffusion: classical vs quantum
    plt.figure(figsize=(6, 4))
    plt.plot(ts_ens, p2_class, linewidth=2.0, label='Classical')
    plt.plot(ts_q, p2_quant, linewidth=2.0, label='Quantum')
    plt.xlabel('Time')
    plt.ylabel(r'$\langle p^2 \rangle$')
    plt.title('Momentum Diffusion: Classical vs Quantum')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_diffusion_classical_quantum.pdf', dpi=300)
    plt.close()
    print("saved fig_diffusion_classical_quantum.pdf")

    # 2) Classical phase space
    plt.figure(figsize=(6, 4))
    plt.plot(thetas_c, ps_c, '.', markersize=1.0, alpha=0.4)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')
    plt.title('Classical Phase Space ($K=%.1f$)' % K)
    plt.tight_layout()
    plt.savefig('fig_classical_phase_space.pdf', dpi=300)
    plt.close()
    print("saved fig_classical_phase_space.pdf")

    # 3) Lyapunov separation
    plt.figure(figsize=(6, 4))
    plt.semilogy(ts_lyap, seps, linewidth=2.0)
    plt.xlabel('Time')
    plt.ylabel('Separation')
    plt.title(r'Lyapunov Separation ($\lambda \approx %.2f$)' % lambda_lyap)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_lyapunov_separation.pdf', dpi=300)
    plt.close()
    print("saved fig_lyapunov_separation.pdf")

    # 4) Entropy
    plt.figure(figsize=(6, 4))
    plt.plot(ts_q, entropy, linewidth=2.0)
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title('Entropy of Momentum Distribution')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_entropy.pdf', dpi=300)
    plt.close()
    print("saved fig_entropy.pdf")

    # 5) Participation ratio
    plt.figure(figsize=(6, 4))
    plt.plot(ts_q, participation, linewidth=2.0)
    plt.xlabel('Time')
    plt.ylabel('Participation ratio')
    plt.title('Participation Ratio')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_participation.pdf', dpi=300)
    plt.close()
    print("saved fig_participation.pdf")

    # 6) SVD spectrum
    plt.figure(figsize=(6, 4))
    plt.semilogy(S / S[0], 'o-', linewidth=2.0, markersize=4)
    plt.xlabel('Mode index')
    plt.ylabel(r'Normalised singular value $\sigma_i / \sigma_1$')
    plt.title(r'SVD Spectrum ($d_{\mathrm{eff}} \approx %.1f$)' % d_eff)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_svd_spectrum.pdf', dpi=300)
    plt.close()
    print("saved fig_svd_spectrum.pdf")

    # 7) Kick function
    t_plot = np.linspace(0, 3, 1000)
    f_plot = [kick_function(t) for t in t_plot]
    plt.figure(figsize=(6, 4))
    plt.plot(t_plot, f_plot, linewidth=2.0)
    plt.xlabel('Time')
    plt.ylabel(r'$f(t)$')
    plt.title('Smooth Kick Function')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_kick_function.pdf', dpi=300)
    plt.close()
    print("saved fig_kick_function.pdf")

    # 8) Adaptive step sizes (classical)
    plt.figure(figsize=(6, 4))
    plt.plot(ts_c, hs_c, linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Step size $h$')
    plt.title('Adaptive RK4 Step Size (Classical)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_stepsizes_classical.pdf', dpi=300)
    plt.close()
    print("saved fig_stepsizes_classical.pdf")

    # 9) Adaptive step sizes (quantum)
    plt.figure(figsize=(6, 4))
    plt.plot(ts_q, hs_q, linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Step size $h$')
    plt.title('Adaptive RK4 Step Size (Quantum)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_stepsizes_quantum.pdf', dpi=300)
    plt.close()
    print("saved fig_stepsizes_quantum.pdf")


def create_classical_bundle_plot(K=5.0, t_end=20.0, n_traj=6, delta_theta=1e-2):
    """
    Plot a bundle of classical trajectories p(t) with slightly different
    initial angles, to show chaotic divergence.
    """
    t_bundle, P_bundle = classical_trajectory_bundle(
        theta0=0.1, p0=0.0, K=K,
        t_end=t_end, n_traj=n_traj, delta_theta=delta_theta
    )

    plt.figure(figsize=(6, 4))
    for k in range(P_bundle.shape[0]):
        plt.plot(t_bundle, P_bundle[k, :],
                 linewidth=1.5,
                 label=fr'Traj {k}: $\theta_0 + {k}\,\Delta\theta$')
    plt.xlabel('Time')
    plt.ylabel(r'$p(t)$')
    plt.title('Bundle of Classical Trajectories (Chaotic Divergence)')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig_classical_bundle.pdf', dpi=300)
    plt.close()
    print("saved fig_classical_bundle.pdf")


def create_quantum_bundle_plot(K=5.0, t_end=20.0, n_max=30,
                               initial_indices=None):
    """
    Plot <p^2>(t) for several different initial momentum eigenstates
    to show that all quantum evolutions remain localised (saturate).
    """
    if initial_indices is None:
        initial_indices = [0, 1, 2, -1]  # for example

    t_qbundle, p2_bundle, initial_indices = quantum_trajectory_bundle(
        K=K, t_end=t_end, n_max=n_max,
        initial_indices=initial_indices
    )

    plt.figure(figsize=(6, 4))
    for k, n0 in enumerate(initial_indices):
        plt.plot(t_qbundle, p2_bundle[k, :],
                 linewidth=1.8,
                 label=fr'$|n={n0}\rangle$')
    plt.xlabel('Time')
    plt.ylabel(r'$\langle p^2 \rangle(t)$')
    # Avoid LaTeX math in title to prevent mathtext parsing errors
    plt.title('Quantum Evolutions: Localised <p^2>')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig_quantum_bundle_p2.pdf', dpi=300)
    plt.close()
    print("saved fig_quantum_bundle_p2.pdf")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SMOOTH KICKED ROTOR: ADAPTIVE RK4 DEMONSTRATION")
    print("=" * 70 + "\n")

    # Use slightly longer time to see divergence / saturation clearly
    K = K_default
    t_end = 20.0
    n_max = 30

    results = run_full_simulation(K=K, t_end=t_end, n_max=n_max)

    # Separate publication-ready figures (diffusion, SVD, etc.)
    create_publication_plots(results, K=K, t_end=t_end, n_max=n_max)

    # Explicit “bundle” plots for chaos vs localisation
    create_classical_bundle_plot(K=K, t_end=t_end, n_traj=6, delta_theta=1e-2)
    create_quantum_bundle_plot(K=K, t_end=t_end, n_max=n_max,
                               initial_indices=[0, 1, 2, -1])

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print("\nKey Results:")
    print(f"  - Lyapunov exponent: {results['lyapunov'][0]:.4f}")
    print(f"  - SVD effective dimension: {results['svd'][3]:.2f}")
    print(f"  - Adaptive RK4 used for BOTH classical and quantum!")
    print("=" * 70 + "\n")
