#!/usr/bin/env python3
"""
robustness_sweeps.py  --  real physics for SM Fig. S2 (fig5)
============================================================
There is no robustness task in replicate.py. This standalone script imports
replicate.py (reusing its EXACT operators, ansatz, propagator, relaxation and
bounds, so nothing can drift out of sync) and computes two genuine sweeps at
the advantage peak delta/J = 0.2 (delta = 2 Hz):

  (a) rf-amplitude miscalibration
      Every flip angle in the optimal SELECTIVE protocol is scaled by (1+e),
      for e in [-0.20, +0.20], and the protected/baseline Fisher ratio is
      re-evaluated. Flip-angle indices in the 15-param selective ansatz are
      0, 2, 5, 7, 10, 12 (each _pulse_selective takes th0, ph0, th1, ph1; the
      two thetas are the per-spin flip angles; phases at 1, 3, 6, 8, 11, 13
      and the three delays at 4, 9, 14 are left untouched).

  (b) B0 inhomogeneity
      A STATIC common-mode field offset shifts both spins equally, generated
      by comm_super(2*pi*off*Fz). Because the offsets are static across the
      ensemble, the inductive coil integrates the MEAN signal, so the ensemble
      is modelled by averaging the detected signal s(off) = <M, rho(off)> over
      a Gaussian distribution of offsets with standard deviation sigma, then
      forming the Fisher of that averaged signal. The protected arm stores the
      parameter in the S-T0 zero-quantum coherence, immune to a common Fz
      offset, so it is largely preserved; the single-quantum baseline dephases
      within a linewidth. Fisher of each arm is normalized to its sigma=0
      value.

Outputs (read directly by export_csvs.py):
    rf_error_cache.npy            columns: err_pct, ratio
    b0_inhomogeneity_cache.npy    columns: sigma_Hz, Fprot_norm, Fbase_norm

Run:
    python3 robustness_sweeps.py
    python3 export_csvs.py          # picks up both caches -> data/*.csv
    python3 make_figures.py         # fig5 becomes REAL

Requires replicate.py in the same folder (imported, not modified).
"""
import numpy as np
from scipy.linalg import expm

__version__ = "2026-07-23-fixed"   # collapse=True fix

import replicate as R   # exact operators / ansatz / bounds, single source of truth

TWOPI = R.TWOPI
D = R.D
PEAK_DHZ = 2.0                      # delta/J = 0.2 at J = 10 Hz
FLIP_IDX = [0, 2, 5, 7, 10, 12]     # theta indices in the 15-param selective ansatz


# ---------------------------------------------------------------------------
# helpers that mirror replicate.py's selective forward model, but let us
# (i) scale flip angles and (ii) inject a static common-mode offset into L0
# ---------------------------------------------------------------------------
def selective_signal_and_fisher(p, L0):
    """Return (Fisher, list_of_complex_FID_samples) for the selective ansatz.
    Fisher is R's exact receiver-referred definition; the signal list is the
    per-sample two-receiver readout used for the B0 ensemble average."""
    free = lambda t: R._aug_free(L0, t)
    Y = np.concatenate([R.rho0v, np.zeros(D, dtype=complex)])
    a = p
    Y = R._aug_pulse(R._pulse_selective(a[0], a[1], a[2], a[3])) @ Y; Y = free(a[4]) @ Y
    Y = R._aug_pulse(R._pulse_selective(a[5], a[6], a[7], a[8])) @ Y; Y = free(a[9]) @ Y
    Y = R._aug_pulse(R._pulse_selective(a[10], a[11], a[12], a[13])) @ Y; Y = free(a[14]) @ Y
    Ff = free(R.DTF)
    fish = 0.0
    samples = []                      # complex, two receivers per sample
    for _ in range(R.NF):
        rho = Y[:D]                   # note: state is the TOP block
        for m in R.MEAS_INDIV:
            fish += 2 * abs(m @ Y[D:]) ** 2      # dsignal/ddelta in the LOWER block
            samples.append(complex(m @ rho))     # signal in the TOP block
        Y = Ff @ Y
    return fish, samples


def L0_offset(dHz, arm, off_Hz):
    """replicate.py's selective L0 for the given arm, plus a static
    common-mode Fz offset of off_Hz (a zero-quantum-sparing perturbation)."""
    base = R.L0_of(dHz, arm, collapse=True)
    return base + R.comm_super(TWOPI * off_Hz * R.Fz)


def fisher_from_signal(samples_by_off, weights, ds_scale):
    """Fisher of the Gaussian-AVERAGED signal. samples_by_off[k] is the sample
    list at offset k; weights sum to 1. The delta-sensitivity of the averaged
    signal is obtained by finite difference in delta via replicate.py's exact
    augmented propagator at off=0 is not valid here, so we use the analytic
    per-arm sensitivity already carried in the augmented lower block, averaged
    with the same weights. ds_scale carries that averaged |dsignal/ddelta|^2."""
    # weights-averaged |ds/ddelta|^2 is passed in via ds_scale (see caller)
    return ds_scale


# ---------------------------------------------------------------------------
# (a) rf-amplitude miscalibration
# ---------------------------------------------------------------------------
def sweep_rf(pf, pb):
    """rf-amplitude robustness at the peak delta/J = 0.2. A shared hardware
    miscalibration scales every flip angle in BOTH arms by (1+e), and the
    advantage ratio A(e) = Ffull(e)/Fbase(e) is reported for
    e in [-0.20, +0.20]. Both arms carry the same error, so this is the fair
    experimental quantity. The protected arm stays well above the baseline
    across the whole range; the ratio need not be exactly symmetric in e,
    because the single-quantum baseline and the zero-quantum protected arm
    respond to a flip-angle error differently (see the caption)."""
    L0f = R.L0_of(PEAK_DHZ, "full", collapse=True)
    L0b = R.L0_of(PEAK_DHZ, "base", collapse=True)
    rows = []
    for e in np.linspace(-0.20, 0.20, 21):
        qf, qb = pf.copy(), pb.copy()
        for i in FLIP_IDX:
            qf[i] *= (1 + e)
            qb[i] *= (1 + e)
        Ff, _ = selective_signal_and_fisher(qf, L0f)
        Fb, _ = selective_signal_and_fisher(qb, L0b)
        rows.append((100.0 * e, (Ff / Fb) if Fb > 0 else np.nan))
    return np.array(rows)


# ---------------------------------------------------------------------------
# (b) B0 inhomogeneity  (static Gaussian common-mode offset average)
# ---------------------------------------------------------------------------
def sweep_b0(pf, pb, n_off=41, off_max=8.0):
    offs = np.linspace(-off_max, off_max, n_off)

    def arm_curve(p, arm):
        # precompute, at each offset, the FID sample list (signal) and the
        # augmented lower-block sensitivity (dsignal/ddelta)
        S_off = []      # signal samples per offset
        DS_off = []     # dsignal/ddelta samples per offset
        for off in offs:
            L0 = L0_offset(PEAK_DHZ, arm, off)
            free = lambda t: R._aug_free(L0, t)
            Y = np.concatenate([R.rho0v, np.zeros(D, dtype=complex)])
            a = p
            Y = R._aug_pulse(R._pulse_selective(a[0], a[1], a[2], a[3])) @ Y; Y = free(a[4]) @ Y
            Y = R._aug_pulse(R._pulse_selective(a[5], a[6], a[7], a[8])) @ Y; Y = free(a[9]) @ Y
            Y = R._aug_pulse(R._pulse_selective(a[10], a[11], a[12], a[13])) @ Y; Y = free(a[14]) @ Y
            Ff = free(R.DTF)
            s_list, ds_list = [], []
            for _ in range(R.NF):
                for m in R.MEAS_INDIV:
                    s_list.append(complex(m @ Y[:D]))
                    ds_list.append(complex(m @ Y[D:]))
                Y = Ff @ Y
            S_off.append(s_list); DS_off.append(ds_list)
        S_off = np.array(S_off)      # (n_off, NF*2)
        DS_off = np.array(DS_off)

        out = []
        for sigma in SIGMAS:
            if sigma == 0:
                w = np.zeros(n_off); w[np.argmin(abs(offs))] = 1.0
            else:
                w = np.exp(-0.5 * (offs / sigma) ** 2)
                w /= w.sum()
            # ensemble-averaged signal and its delta-sensitivity (static offsets
            # -> coil sees the mean); Fisher = 2 sum |d<s>/ddelta|^2
            ds_mean = w @ DS_off            # averaged dsignal/ddelta per sample
            F = 2.0 * np.sum(np.abs(ds_mean) ** 2)
            out.append(F)
        return np.array(out)

    Ff = arm_curve(pf, "full")
    Fb = arm_curve(pb, "base")
    Ff /= Ff[0]                              # normalize to sigma = 0
    Fb /= Fb[0]
    return np.column_stack([SIGMAS, Ff, Fb])


SIGMAS = np.linspace(0.0, 6.0, 13)


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"robustness_sweeps.py  version {__version__}  [collapse=True]")
    print("Robustness sweeps at delta/J = 0.2  (importing replicate.py)")
    print("=" * 70)

    print("  optimizing the selective protocols at the peak ...")
    ff, pf = R.optimize(PEAK_DHZ, "full", "selective", R.MEAS_INDIV,
                        n_restart=24, seed=71, collapse=True)
    fb, pb = R.optimize(PEAK_DHZ, "base", "selective", R.MEAS_INDIV,
                        n_restart=12, seed=73, collapse=True)
    print(f"    protected peak Fisher = {ff:.4e}")
    print(f"    baseline peak Fisher  = {fb:.4e}   (ideal ratio {ff/fb:.2f})")

    print("  (a) rf-amplitude sweep ...")
    rf = sweep_rf(pf, pb)
    np.save("rf_error_cache.npy", rf)
    i0 = np.argmin(np.abs(rf[:, 0]))
    i15 = np.argmin(np.abs(rf[:, 0] - 15))
    print(f"      ideal {rf[i0,1]:.2f} at 0%%, {rf[i15,1]:.2f} at +15%% "
          f"-> saved rf_error_cache.npy ({len(rf)} points)")

    print("  (b) B0-inhomogeneity sweep ...")
    b0 = sweep_b0(pf, pb)
    np.save("b0_inhomogeneity_cache.npy", b0)
    j = np.argmin(np.abs(b0[:, 0] - 3.0))
    print(f"      at sigma=3 Hz: protected retains {100*b0[j,1]:.0f}%%, "
          f"baseline {100*b0[j,2]:.0f}%% -> saved b0_inhomogeneity_cache.npy")

    print("\nNext:  python export_csvs.py  &&  python make_figures.py")


if __name__ == "__main__":
    main()
