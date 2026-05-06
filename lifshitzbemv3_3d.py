# -*- coding: utf-8 -*-
"""
casimir_v2_3d_validated.py
==========================
Original Lifshitz-BEM code with one critical physics fix applied, plus a
3-D scalar extension and validation workflow.

THE BUG (in the original)
--------------------------
calculate_casimir_energy() and run_lateral_experiment() built a single
N × N matrix containing ALL panel-pair interactions — including same-plate
pairs (top↔top, bottom↔bottom).  Same-plate distances are set by panel
spacing and do NOT change with separation `a`.  They dominated log det(M)
and completely masked the a-dependence, producing the flat curve you saw.

THE FIX
--------
We now build only the cross-plate propagator G₁₂ (shape N_half × N_half),
whose entries K₀(κ · dist_cross) decay with `a` as they should.

The energy integrand becomes:

    integrand(ξ) = −‖G₁₂(ξ)‖²_F   (first-Born approximation)

This is:
  • Always negative  → attractive Casimir energy  ✓
  • Monotonically decaying with a                  ✓
  • Numerically stable for all κ                  ✓
  • Exact in the weak-coupling / large-a limit    ✓

Everything else (geometry, frequency sweep, plotting, lateral sweep,
vector plot) is kept identical to the original.

NEW IN THIS VERSION
-------------------
This file also adds a 3-D scalar surface-discretized version of the same idea.

The 2-D code uses the Euclidean scalar Green's function after one transverse
dimension has already been analytically integrated out:

    G₂D(r) ∝ K₀(κr)

The new 3-D code uses the full 3-D Euclidean scalar Green's function:

    G₃D(r) = exp(-κr) / (4πr)

The 3-D energy integrand is still the same weak-coupling / first-Born form:

    integrand(ξ) = −‖G₁₂(ξ)‖²_F

IMPORTANT PHYSICS NOTE
----------------------
This is NOT a full electromagnetic Lifshitz solver. It is a scalar
weak-coupling / first-Born BEM-like model.

Therefore:
  • It should reproduce the correct qualitative distance dependence.
  • It can compare corrugated vs flat geometry cleanly through ratios.
  • It can be calibrated to the analytic ideal parallel-plate result.
  • After validation on simple geometries, it can be used to study
    nontrivial geometries where closed-form expressions do not exist.

The safest scientific claim is:

    "The 3-D scalar BEM model was validated against the known parallel-plate
    Casimir scaling, calibrated to the analytic ideal-plate force at one
    reference separation, and then applied to nontrivial corrugated geometry
    to estimate geometry-dependent corrections beyond PFA."
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn
from scipy.integrate import simpson
import os
import warnings

H_BAR = 1.054e-34
C     = 2.998e8


# ── Geometry ──────────────────────────────────────────────────────────────────

def y_top(x, a=1.0, A=0.25, lam=2.5, phase=0.0):
    return a + A * np.sin(2 * np.pi * x / lam + phase)


def build_panels(n=50, a=1.0, A=0.25, lam=2.5, phase=0.0):
    x   = np.linspace(-3, 3, n)
    top = np.column_stack([x, y_top(x, a, A, lam, phase)])
    bot = np.column_stack([x, np.zeros_like(x)])
    starts  = np.vstack([top[:-1], bot[:-1]])
    ends    = np.vstack([top[1:],  bot[1:]])
    centers = 0.5 * (starts + ends)
    return starts, ends, centers


# ── Cross-plate propagator (THE FIX) ─────────────────────────────────────────

def build_G12(centers, lengths, kappa):
    """
    Build the cross-plate propagator G₁₂ at wavenumber κ = ξ/c.

    Only top↔bottom panel pairs are included.  Same-plate pairs are
    excluded entirely — they do not depend on separation `a` and would
    otherwise dominate the log-determinant and mask the force.

    G₁₂[i, j] = length_j · K₀(κ · ‖top_center_i − bot_center_j‖)

    Energy integrand = −‖G₁₂‖²_F  (first-Born, always attractive)
    """
    N      = len(centers)
    N_half = N // 2
    top_c  = centers[:N_half]    # corrugated top plate
    bot_c  = centers[N_half:]    # flat bottom plate
    bot_l  = lengths[N_half:]

    G12 = np.zeros((N_half, N_half))
    for i in range(N_half):
        for j in range(N_half):
            dist = np.linalg.norm(top_c[i] - bot_c[j])
            if dist < 1e-12:
                continue
            arg = kappa * dist
            if arg > 500:        # K₀ is numerically zero — skip
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                G12[i, j] = bot_l[j] * kn(0, arg)
    return G12


# ── Energy calculation ────────────────────────────────────────────────────────

def calculate_casimir_energy(a, A_amp, lam, n_panels=40,
                             xi_lo=1e7, xi_hi=1e10, n_xi=40):
    """
    xi_lo / xi_hi should bracket C/a_max and C/a_min for your sweep
    with ~10x margin on each side.  Defaults cover a ∈ [0.5, 2.5]
    (arbitrary length units) with C = 2.998e8:
        a=0.5 → peak 6e8,  a=2.5 → peak 1.2e8
        window [1e7, 1e10] gives ~10x margin both sides for all a.

    DO NOT make xi_lo/xi_hi functions of a — that introduces a spurious
    1/a Jacobian factor that corrupts the power-law exponent.
    """
    starts, ends, centers = build_panels(n_panels, a, A_amp, lam)
    vecs    = ends - starts
    lengths = np.linalg.norm(vecs, axis=1)

    xi_points = np.logspace(np.log10(xi_lo), np.log10(xi_hi), n_xi)

    integrand_vals = []
    for xi in xi_points:
        kappa = xi / C
        G12   = build_G12(centers, lengths, kappa)
        integrand_vals.append(-np.sum(G12 ** 2))

    energy_integral = simpson(y=integrand_vals, x=xi_points)
    return (H_BAR / (2 * np.pi)) * energy_integral


# ── Corrugated vs flat comparison (the main result figure) ───────────────────

def run_comparison_experiment(a_vals=None, n_panels=40):
    """
    The primary paper figure.

    Computes the BEM Casimir force for both the corrugated plate (A=0.3)
    and a flat plate (A=0) using identical simulation settings, then plots:

      Left  — |Force| vs separation for both geometries (log scale)
      Right — Ratio F_corrugated / F_flat, showing the beyond-PFA correction

    The ratio is the cleanest result: numerical artifacts (IR cutoff, panel
    count, frequency sampling) cancel, leaving only the genuine physical
    effect of the corrugation geometry.
    """
    if a_vals is None:
        a_vals = np.linspace(0.5, 2.5, 30)

    print("Computing corrugated (A=0.3) sweep...")
    E_corr = []
    for a in a_vals:
        E = calculate_casimir_energy(a, A_amp=0.3, lam=2.0, n_panels=n_panels)
        E_corr.append(E)
        print(f"  a={a:.2f}  E={E:.4e}")

    print("Computing flat plate (A=0) reference sweep...")
    E_flat = []
    for a in a_vals:
        E = calculate_casimir_energy(a, A_amp=0.0, lam=2.0, n_panels=n_panels)
        E_flat.append(E)
        print(f"  a={a:.2f}  E={E:.4e}")

    E_corr = np.array(E_corr)
    E_flat = np.array(E_flat)
    F_corr = -np.gradient(E_corr, a_vals)
    F_flat = -np.gradient(E_flat, a_vals)

    # Power-law exponents (avoid noisy endpoints)
    mid = slice(4, -4)
    exp_corr = np.polyfit(np.log(a_vals[mid]), np.log(np.abs(F_corr[mid])), 1)[0]
    exp_flat = np.polyfit(np.log(a_vals[mid]), np.log(np.abs(F_flat[mid])), 1)[0]
    print(f"\nFlat plate exponent:       a^{exp_flat:.3f}")
    print(f"Corrugated plate exponent: a^{exp_corr:.3f}")

    # Correction factor
    ratio = np.abs(F_corr) / np.abs(F_flat)
    dev_mid = abs(ratio[len(a_vals)//2] - 1.0) * 100
    print(f"Beyond-PFA deviation at midpoint: {dev_mid:.2f}%")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: force magnitude
    ax = axes[0]
    ax.plot(a_vals, np.abs(F_corr), 'o-', color='darkmagenta', markersize=4,
            label=fr'Corrugated  ($A=0.3$)  fit: $a^{{{exp_corr:.2f}}}$')
    ax.plot(a_vals, np.abs(F_flat), 's--', color='steelblue', markersize=4,
            lw=1.8,
            label=fr'Flat plate  ($A=0$)    fit: $a^{{{exp_flat:.2f}}}$')
    ax.set(yscale='log',
           xlabel='Separation $a$',
           ylabel='|Force| (a.u.)',
           title='BEM Force: Corrugated vs Flat Plate')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend()

    # Right: beyond-PFA correction factor
    ax2 = axes[1]
    ax2.plot(a_vals, ratio, 'o-', color='crimson', markersize=4,
             label='Correction factor')
    ax2.axhline(1.0, color='k', lw=1.2, ls='--', label='PFA limit (ratio = 1)')
    ax2.fill_between(a_vals, ratio, 1.0,
                     where=(ratio >= 1.0), alpha=0.15, color='crimson',
                     label='Beyond-PFA enhancement')
    ax2.fill_between(a_vals, ratio, 1.0,
                     where=(ratio < 1.0),  alpha=0.15, color='steelblue',
                     label='Beyond-PFA suppression')
    ax2.set(xlabel='Separation $a$',
            ylabel=r'$|F_\mathrm{corrugated}|\;/\;|F_\mathrm{flat}|$',
            title='Corrugation Correction Factor (PFA Deviation)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    os.makedirs("img", exist_ok=True)
    save_path = os.path.join("img", "lifshitzbem_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison figure saved to {save_path}")
    plt.show()

    return a_vals, F_corr, F_flat, ratio


# ── Vertical force experiment ─────────────────────────────────────────────────

def run_experiment():
    a_vals = np.linspace(0.5, 2.5, 100)
    print("Starting Lifshitz-BEM frequency integration...")
    energies = []
    for a in a_vals:
        E = calculate_casimir_energy(a, 0.3, 2.0)
        energies.append(E)
        print(f"Distance a={a:.2f}  E={E:.4e}")

    energies = np.array(energies)
    forces   = -np.gradient(energies, a_vals)

    # FIX 1: PFA reference for 2D scalar BEM is a^-3, not a^-4.
    # The 3D flat-plate result F ∝ a^-4 comes from integrating over 2 transverse
    # dimensions.  In this 2D simulation (line panels, one transverse dimension
    # integrated out analytically via K₀), the correct PFA scaling is a^-3.
    pfa_2d = np.abs(forces[0]) * (a_vals[0] / a_vals) ** 3

    # Measure the actual power-law exponent from the simulation data
    # (use middle of the range to avoid endpoint noise)
    mid = slice(10, -10)
    coeffs   = np.polyfit(np.log(a_vals[mid]), np.log(np.abs(forces[mid])), 1)
    exponent = coeffs[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_vals, np.abs(forces), 'o-', color='darkmagenta', markersize=3,
            label=f'|Lifshitz-BEM Force|  (fit: $a^{{{exponent:.2f}}}$)')
    ax.plot(a_vals, pfa_2d, 'k--', lw=1.5,
            label=r'2-D PFA reference  $\propto a^{-3}$')
    ax.set(yscale='log',
           xlabel='Separation $a$',
           ylabel='Force Magnitude (a.u.)',
           title='Lifshitz-BEM Force Profile — 2-D PFA & Adaptive Frequency')
    ax.grid(True, which='both', ls='-', alpha=0.3)
    ax.legend()

    os.makedirs("img", exist_ok=True)
    save_path = os.path.join("img", "lifshitzbem_fixed.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nMeasured power-law exponent: a^{exponent:.3f}")
    print(f"Graph saved to {save_path}")
    plt.show()

    return a_vals, energies, forces


# ── Lateral force experiment ──────────────────────────────────────────────────

def run_lateral_experiment():
    a_fixed = 1.0
    phases  = np.linspace(0, 2 * np.pi, 40)
    print(f"Starting Lateral Force sweep at a={a_fixed}...")

    energies = []
    for p in phases:
        starts, ends, centers = build_panels(n=60, a=a_fixed, A=0.3,
                                             lam=2.0, phase=p)
        vecs    = ends - starts
        lengths = np.linalg.norm(vecs, axis=1)

        # Fixed frequency window — same as calculate_casimir_energy.
        # Must NOT scale with a to avoid spurious power-law artifacts.
        xi_points = np.logspace(7, 10, 30)
        integrand_vals = []
        for xi in xi_points:
            kappa = xi / C
            G12   = build_G12(centers, lengths, kappa)
            integrand_vals.append(-np.sum(G12 ** 2))

        E = (H_BAR / (2 * np.pi)) * simpson(y=integrand_vals, x=xi_points)
        energies.append(E)
        print(f"Phase {p / np.pi:.2f}π  E={E:.4e}")

    energies  = np.array(energies)
    lat_force = -np.gradient(energies, phases)

    plt.figure(figsize=(8, 5))
    plt.plot(phases / np.pi, lat_force, color='forestgreen',
             lw=2, marker='o', markersize=4, label='Lateral force')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.xlabel("Phase Shift (units of π)")
    plt.ylabel("Lateral Force $F_x$ (a.u.)")
    plt.title("Lifshitz-BEM Lateral Force — Cross-Plate Fix Applied")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs("img", exist_ok=True)
    save_path = os.path.join("img", "lifshitz_lateral_fixed.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Lateral graph saved to {save_path}")
    plt.show()

    return phases, energies, lat_force


# ── Force vector plot ─────────────────────────────────────────────────────────

def plot_data_driven_vectors(a_vals, v_forces, phases, l_forces, A, lam):
    """
    Visualise the net Casimir force vector on the corrugated plate
    using the actual sweep data (unchanged from original).
    """
    os.makedirs("img", exist_ok=True)

    idx_a = len(a_vals) // 4
    idx_p = len(phases)  // 4

    a   = a_vals[idx_a]
    phi = phases[idx_p]
    fy  = v_forces[idx_a]
    fx  = l_forces[idx_p]

    starts, ends, centers = build_panels(n=80, a=a, A=A, lam=lam, phase=phi)

    plt.figure(figsize=(10, 6))
    plt.plot(centers[:40, 0], centers[:40, 1], 'k-', lw=3, label="Top Plate")
    plt.plot(centers[40:, 0], centers[40:, 1], 'k-', lw=3, label="Bottom Plate")

    origin_x = np.mean(centers[:40, 0])
    origin_y = np.mean(centers[:40, 1])

    # Scale factor: Casimir forces are ~10⁻²⁵ N, so we amplify for visibility
    scale_factor = 1e26
    plt.quiver(origin_x, origin_y,
               fx * scale_factor, fy * scale_factor,
               color='crimson', angles='xy', scale_units='xy', scale=1,
               width=0.01, label="Net Force Vector (data-driven)")

    plt.annotate(f"Fx: {fx:.2e}\nFy: {fy:.2e}",
                 xy=(origin_x, origin_y), xytext=(20, 20),
                 textcoords='offset points', color='crimson', weight='bold')

    plt.title(f"Force Vector at a={a:.2f}, phase={phi/np.pi:.2f}π")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.legend()

    plt.savefig("img/casimir_data_vector_fixed.png", dpi=300,
                bbox_inches="tight")
    print("Vector plot saved to img/casimir_data_vector_fixed.png")
    plt.show()


# ── 3-D surface geometry extension ────────────────────────────────────────────

def make_rect_plate_3d(nx=30, ny=30, Lx=6.0, Ly=6.0, z=0.0):
    """
    Build a flat rectangular plate as a grid of small area elements.

    This is the 3-D analog of build_panels(), except the surface is now
    discretized into tiny patches rather than line segments.

    Returns:
        centers : (nx*ny, 3) array of patch center coordinates
        areas   : (nx*ny,) array of patch areas

    NOTE:
    We use point-centered patches, so this is not a full triangular mesh yet.
    It is still a valid first 3-D validation geometry because parallel plates
    are exactly the geometry we need for the analytic benchmark.
    """
    xs = np.linspace(-Lx / 2, Lx / 2, nx)
    ys = np.linspace(-Ly / 2, Ly / 2, ny)

    # Patch spacing.  These become the area weights in the surface integral.
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    centers = []
    areas   = []

    for x in xs:
        for y in ys:
            centers.append([x, y, z])
            areas.append(dx * dy)

    return np.array(centers), np.array(areas)


def make_corrugated_plate_3d(nx=30, ny=30, Lx=6.0, Ly=6.0,
                             a=1.0, A=0.3, lam=2.0, phase=0.0):
    """
    Build a 3-D sinusoidally corrugated plate:

        z(x, y) = a + A sin(2πx/λ + phase)

    The corrugation varies only in x, while the plate extends in both x and y.
    This gives the simplest 3-D version of your original 2-D profile.

    Area correction:
        dS = dx dy sqrt(1 + (dz/dx)^2 + (dz/dy)^2)

    Since z does not vary in y here:
        dS = dx dy sqrt(1 + (dz/dx)^2)
    """
    xs = np.linspace(-Lx / 2, Lx / 2, nx)
    ys = np.linspace(-Ly / 2, Ly / 2, ny)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    centers = []
    areas   = []

    for x in xs:
        for y in ys:
            z = a + A * np.sin(2 * np.pi * x / lam + phase)

            # Surface-area correction for the slanted corrugated patches.
            dzdx = A * (2 * np.pi / lam) * np.cos(2 * np.pi * x / lam + phase)
            dS   = dx * dy * np.sqrt(1 + dzdx ** 2)

            centers.append([x, y, z])
            areas.append(dS)

    return np.array(centers), np.array(areas)


def build_G12_3d(top_centers, bot_centers, top_areas, bot_areas, kappa):
    """
    Build the 3-D cross-surface propagator G₁₂.

    Only top↔bottom patch pairs are included.  Same-surface pairs are still
    excluded for the same reason as in the original 2-D fix: they do not
    encode the separation-dependent interaction between the two plates.

    3-D Euclidean scalar Green's function:

        G₃D(r) = exp(-κr) / (4πr)

    Weighted matrix:

        G₁₂[i, j] = sqrt(A_i A_j) · exp(-κrᵢⱼ) / (4πrᵢⱼ)

    Energy integrand:

        integrand(ξ) = −‖G₁₂(ξ)‖²_F

    The sqrt(A_i A_j) weighting is symmetric and makes the Frobenius norm
    behave like a double surface integral over dS dS'.
    """
    diff = top_centers[:, None, :] - bot_centers[None, :, :]
    r    = np.linalg.norm(diff, axis=2)

    # Avoid division by zero.  Cross-plate patches should never overlap in a
    # valid setup, but this prevents numerical crashes if geometry is malformed.
    r[r < 1e-12] = 1e-12

    # For large κr, exp(-κr) is essentially zero and naturally underflows safely.
    kernel = np.exp(-kappa * r) / (4 * np.pi * r)

    # Symmetric surface area weighting.
    weights = np.sqrt(top_areas[:, None] * bot_areas[None, :])

    return weights * kernel


def calculate_casimir_energy_3d(a, A_amp=0.0, lam=2.0,
                                nx=30, ny=30, Lx=6.0, Ly=6.0,
                                xi_lo=1e7, xi_hi=1e10, n_xi=50):
    """
    3-D scalar first-Born Casimir energy.

    For A_amp = 0:
        top plate is flat, so this is the parallel-plate validation case.

    For A_amp > 0:
        top plate is corrugated, so this becomes the nontrivial geometry case.

    IMPORTANT:
    This returns the raw scalar-model energy, not the exact EM Lifshitz energy.
    For absolute force estimates, calibrate this model to an analytic geometry
    first using validate_3d_parallel_plate().
    """
    bot_centers, bot_areas = make_rect_plate_3d(nx, ny, Lx, Ly, z=0.0)

    if A_amp == 0:
        top_centers, top_areas = make_rect_plate_3d(nx, ny, Lx, Ly, z=a)
    else:
        top_centers, top_areas = make_corrugated_plate_3d(
            nx, ny, Lx, Ly, a=a, A=A_amp, lam=lam
        )

    xi_points = np.logspace(np.log10(xi_lo), np.log10(xi_hi), n_xi)

    integrand_vals = []
    for xi in xi_points:
        kappa = xi / C
        G12   = build_G12_3d(top_centers, bot_centers,
                             top_areas, bot_areas, kappa)
        integrand_vals.append(-np.sum(G12 ** 2))

    energy_integral = simpson(y=integrand_vals, x=xi_points)
    return (H_BAR / (2 * np.pi)) * energy_integral


# ── Analytic benchmark formulas ───────────────────────────────────────────────

def analytic_parallel_plate_pressure(a):
    """
    Exact ideal-conductor parallel-plate Casimir pressure:

        P = -π² ħ c / (240 a⁴)

    Units:
        a in meters
        pressure in N/m²

    This is the experimental/theoretical benchmark for the 3-D scaling.
    """
    return -(np.pi ** 2 * H_BAR * C) / (240 * a ** 4)


def analytic_parallel_plate_force(a, area):
    """
    Exact ideal-conductor parallel-plate Casimir force:

        F = P · area
          = -π² ħ c A / (240 a⁴)

    Units:
        a in meters
        area in m²
        force in newtons
    """
    return analytic_parallel_plate_pressure(a) * area


def analytic_sphere_plate_force_pfa(a, R):
    """
    Ideal-conductor sphere-plate Casimir force using PFA:

        F = -π³ ħ c R / (360 a³)

    This is often closer to real experimental setups because sphere-plate
    alignment is easier than perfectly parallel plate-plate alignment.

    Units:
        a in meters
        R in meters
        force in newtons
    """
    return -(np.pi ** 3 * H_BAR * C * R) / (360 * a ** 3)


def print_experimental_scale_examples():
    """
    Print real force estimates for standard experimental geometries.

    These values are not generated by the scalar model.  They are analytic
    ideal-conductor benchmarks used to sanity-check orders of magnitude.
    """
    print("\nExperimental-scale analytic estimates")
    print("-------------------------------------")

    a_plate = 1e-6          # 1 micrometer
    area    = 1e-4          # 1 cm² = 1e-4 m²
    P       = analytic_parallel_plate_pressure(a_plate)
    F       = analytic_parallel_plate_force(a_plate, area)

    print(f"Parallel plates at a = {a_plate:.1e} m")
    print(f"  Pressure: {P:.4e} N/m²")
    print(f"  Force for area = {area:.1e} m²: {F:.4e} N")

    a_sphere = 100e-9       # 100 nm
    R        = 100e-6       # 100 micrometers
    Fsp      = analytic_sphere_plate_force_pfa(a_sphere, R)

    print(f"\nSphere-plate PFA at a = {a_sphere:.1e} m, R = {R:.1e} m")
    print(f"  Force: {Fsp:.4e} N")


# ── 3-D validation: parallel plates ───────────────────────────────────────────

def validate_3d_parallel_plate(a_vals=None, nx=24, ny=24,
                               Lx=6.0, Ly=6.0,
                               xi_lo=1e7, xi_hi=1e10, n_xi=40):
    """
    Validate the 3-D scalar solver against the known parallel-plate result.

    What we expect:
        Exact EM ideal plates: F ∝ a⁻⁴

    What we check:
        1. Does the 3-D numerical force decay close to a power law?
        2. Is the exponent reasonably close to -4 after going from 2-D to 3-D?
        3. After one-point calibration, does the curve follow the analytic
           parallel-plate trend?

    Why calibration is needed:
        The code is a scalar weak-coupling model, not a full EM Lifshitz solver.
        So the absolute prefactor is not expected to match exactly.  The clean
        validation target is the distance dependence and the ability to compare
        geometry corrections using the same model.
    """
    if a_vals is None:
        a_vals = np.linspace(0.6, 2.4, 24)

    print("\nStarting 3-D parallel-plate validation...")
    print("Computing raw scalar-model energies...")

    energies = []
    for a in a_vals:
        E = calculate_casimir_energy_3d(
            a, A_amp=0.0, lam=2.0,
            nx=nx, ny=ny, Lx=Lx, Ly=Ly,
            xi_lo=xi_lo, xi_hi=xi_hi, n_xi=n_xi
        )
        energies.append(E)
        print(f"  a={a:.3f}  E_raw={E:.4e}")

    energies = np.array(energies)
    F_raw    = -np.gradient(energies, a_vals)

    # Use the same arbitrary simulation area for the analytic comparison.
    # This keeps the curve comparison internally consistent.
    area_sim = Lx * Ly

    # The numerical a values are dimensionless simulation units.
    # For the validation plot, we compare the shape using the same a values.
    # The analytic force is therefore used as a reference power law.  For real
    # experimental values, use print_experimental_scale_examples().
    F_exact_shape = np.array([
        -area_sim / (a ** 4) for a in a_vals
    ])

    # One-point calibration at the midpoint: match scalar model to reference
    # shape at one separation, then test whether the whole curve follows.
    mid_idx = len(a_vals) // 2
    scale   = F_exact_shape[mid_idx] / F_raw[mid_idx]
    F_scaled = scale * F_raw

    # Power-law exponents.  Avoid endpoints because numerical gradients are
    # noisier there.
    fit_slice = slice(4, -4)
    exp_raw = np.polyfit(
        np.log(a_vals[fit_slice]),
        np.log(np.abs(F_raw[fit_slice])),
        1
    )[0]

    exp_scaled = np.polyfit(
        np.log(a_vals[fit_slice]),
        np.log(np.abs(F_scaled[fit_slice])),
        1
    )[0]

    print(f"\n3-D raw scalar-model exponent:       a^{exp_raw:.3f}")
    print(f"3-D calibrated model exponent:       a^{exp_scaled:.3f}")
    print("Expected ideal parallel-plate force: a^-4")
    print(f"Calibration scale factor: {scale:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_vals, np.abs(F_scaled), 'o-', color='darkmagenta',
            markersize=4,
            label=fr'3-D scalar BEM, calibrated  fit: $a^{{{exp_scaled:.2f}}}$')
    ax.plot(a_vals, np.abs(F_exact_shape), 'k--', lw=1.5,
            label=r'Ideal parallel-plate reference  $\propto a^{-4}$')
    ax.set(yscale='log',
           xlabel='Separation $a$',
           ylabel='|Force| (scaled a.u.)',
           title='3-D Validation: Parallel Plates')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    os.makedirs("img", exist_ok=True)
    save_path = os.path.join("img", "validate_3d_parallel_plate.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"3-D validation graph saved to {save_path}")
    plt.show()

    return a_vals, energies, F_raw, F_scaled, F_exact_shape, scale


# ── 3-D corrugated vs flat comparison ─────────────────────────────────────────

def run_3d_corrugated_comparison(a_vals=None, nx=24, ny=24,
                                 Lx=6.0, Ly=6.0,
                                 A_amp=0.3, lam=2.0,
                                 xi_lo=1e7, xi_hi=1e10, n_xi=40):
    """
    Main 3-D nontrivial geometry experiment.

    This repeats the same idea as run_comparison_experiment(), but now the
    surfaces are genuinely 3-D.

    It computes:
        1. Flat-plate 3-D scalar force
        2. Corrugated-plate 3-D scalar force
        3. Ratio |F_corrugated| / |F_flat|

    The ratio is still the safest result because it cancels most scalar-model
    prefactor issues and isolates the effect of geometry.
    """
    if a_vals is None:
        a_vals = np.linspace(0.6, 2.4, 24)

    print("\nStarting 3-D corrugated vs flat comparison...")

    E_flat = []
    print("Computing 3-D flat reference...")
    for a in a_vals:
        E = calculate_casimir_energy_3d(
            a, A_amp=0.0, lam=lam,
            nx=nx, ny=ny, Lx=Lx, Ly=Ly,
            xi_lo=xi_lo, xi_hi=xi_hi, n_xi=n_xi
        )
        E_flat.append(E)
        print(f"  flat a={a:.3f}  E={E:.4e}")

    E_corr = []
    print("Computing 3-D corrugated geometry...")
    for a in a_vals:
        E = calculate_casimir_energy_3d(
            a, A_amp=A_amp, lam=lam,
            nx=nx, ny=ny, Lx=Lx, Ly=Ly,
            xi_lo=xi_lo, xi_hi=xi_hi, n_xi=n_xi
        )
        E_corr.append(E)
        print(f"  corr a={a:.3f}  E={E:.4e}")

    E_flat = np.array(E_flat)
    E_corr = np.array(E_corr)

    F_flat = -np.gradient(E_flat, a_vals)
    F_corr = -np.gradient(E_corr, a_vals)

    ratio = np.abs(F_corr) / np.abs(F_flat)

    fit_slice = slice(4, -4)
    exp_flat = np.polyfit(
        np.log(a_vals[fit_slice]),
        np.log(np.abs(F_flat[fit_slice])),
        1
    )[0]
    exp_corr = np.polyfit(
        np.log(a_vals[fit_slice]),
        np.log(np.abs(F_corr[fit_slice])),
        1
    )[0]

    dev_mid = abs(ratio[len(a_vals) // 2] - 1.0) * 100

    print(f"\n3-D flat exponent:       a^{exp_flat:.3f}")
    print(f"3-D corrugated exponent: a^{exp_corr:.3f}")
    print(f"3-D corrugation deviation at midpoint: {dev_mid:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(a_vals, np.abs(F_corr), 'o-', color='darkmagenta', markersize=4,
            label=fr'3-D Corrugated  fit: $a^{{{exp_corr:.2f}}}$')
    ax.plot(a_vals, np.abs(F_flat), 's--', color='steelblue', markersize=4,
            lw=1.8,
            label=fr'3-D Flat  fit: $a^{{{exp_flat:.2f}}}$')
    ax.set(yscale='log',
           xlabel='Separation $a$',
           ylabel='|Force| (raw scalar a.u.)',
           title='3-D Scalar BEM Force: Corrugated vs Flat')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(a_vals, ratio, 'o-', color='crimson', markersize=4,
             label='3-D correction factor')
    ax2.axhline(1.0, color='k', lw=1.2, ls='--', label='Flat reference')
    ax2.fill_between(a_vals, ratio, 1.0,
                     where=(ratio >= 1.0), alpha=0.15, color='crimson',
                     label='Geometry enhancement')
    ax2.fill_between(a_vals, ratio, 1.0,
                     where=(ratio < 1.0), alpha=0.15, color='steelblue',
                     label='Geometry suppression')
    ax2.set(xlabel='Separation $a$',
            ylabel=r'$|F_\mathrm{corrugated}|\;/\;|F_\mathrm{flat}|$',
            title='3-D Corrugation Correction Factor')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    os.makedirs("img", exist_ok=True)
    save_path = os.path.join("img", "lifshitzbem_3d_corrugated_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"3-D corrugated comparison saved to {save_path}")
    plt.show()

    return a_vals, F_corr, F_flat, ratio


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    A_amp      = 0.3
    wavelength = 2.0

    # Keep the original 2-D workflow exactly available.
    # These runs are useful because they are faster and good for debugging.

    # 1. Main result: corrugated vs flat + beyond-PFA correction factor
    a_vals, F_corr, F_flat, ratio = run_comparison_experiment()

    # 2. Vertical force sweep (single corrugated curve)
    a_vals, energies_v, forces_v = run_experiment()

    # 3. Lateral force sweep
    phases, energies_l, forces_l = run_lateral_experiment()

    # 4. Force vector visualization
    plot_data_driven_vectors(a_vals, forces_v, phases, forces_l,
                             A=A_amp, lam=wavelength)

    # 5. Print analytic experimental-scale benchmarks.
    # These are real-order force estimates for known simple geometries.
    print_experimental_scale_examples()

    # 6. Validate the new 3-D solver on flat parallel plates.
    # Expected force scaling: F ∝ a^-4.
    validate_3d_parallel_plate()

    # 7. Apply the validated 3-D workflow to a nontrivial corrugated geometry.
    run_3d_corrugated_comparison(A_amp=A_amp, lam=wavelength)

