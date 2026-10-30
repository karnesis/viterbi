"""Simulated LISA data: noise model, Gaussian noise and MBHB injections in A, E, T.

All sources are generated in the frequency domain with BBHx (v1.2.x API, i.e.
``force_backend`` instead of the old ``use_gpu``), on the rfft grid of a
time series of ``n_samples`` samples with cadence ``dt``.

Conventions
-----------
* Frequency series follow the continuous Fourier transform convention used by
  BBHx, ``h(f) = int h(t) exp(-2 pi i f t) dt``, so that the DFT of the sampled
  series is ``h(f)/dt``.
* One-sided PSD ``S(f)`` such that ``<|n(f)|^2> = S(f) T / 2``.

NK, NG 2026
"""
import numpy as np
import scipy.fft as sft

try:
    from bbhx.waveformbuild import BBHWaveformFD
    from bbhx.utils.constants import PC_SI, YRSID_SI
except ImportError:  # keep the noise utilities usable without BBHx
    BBHWaveformFD = None
    PC_SI = 3.0856775814913674e16
    YRSID_SI = 31558149.763545603

CLIGHT = 299792458.0
LISA_L = 2.5e9  # arm length [m]

# Nominal SciRD noise levels (log10)
SA_NOMINAL = np.log10((3.e-15)**2)   # m^2/sec^4/Hz
SI_NOMINAL = np.log10((15.e-12)**2)  # m^2/Hz


# ------------------------------------------------------------------ noise ---- #

def noise_psd_ae(f, Sa=SA_NOMINAL, Si=SI_NOMINAL):
    """Analytic one-sided PSD of the A and E TDI channels (equal-arm, TDI1).

    Same model as ``statutils.noise_model``. The value at ``f=0`` is set to inf so
    that whitening by it zeroes the DC bin.
    """
    f = np.asarray(f, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 2.0 * np.pi * (LISA_L / CLIGHT) * f
        Spm = 10**Sa * (1.0 + (0.4e-3/f)**2) * (1.0 + (f/8e-3)**4) \
            * (2.0*np.pi*f)**(-4.) * (2.0*np.pi*f/CLIGHT)**2
        Sop = 10**Si * (1.0 + (2.e-3/f)**4) * (2.0*np.pi*f/CLIGHT)**2
        S = 8.0 * np.sin(x)**2 * (2.0*Spm*(3.0 + 2.0*np.cos(x) + np.cos(2.*x)) + Sop*(2.0 + np.cos(x)))
    S = np.where(f > 0, S, np.inf)
    return S


def gaussian_noise_fd(psd, df, rng):
    """Complex Gaussian noise with one-sided PSD ``psd`` on a grid of spacing ``df``."""
    psd = np.where(np.isfinite(psd), psd, 0.0)
    sigma = np.sqrt(psd / (4.0 * df))
    return rng.normal(scale=sigma) + 1j * rng.normal(scale=sigma)


def fd_to_td(hf, n_samples, dt):
    """Frequency series (continuous-FT units) to a real time series."""
    return sft.irfft(hf, n=n_samples) / dt


def compute_snr(hf, psd, df):
    """Optimal SNR, sqrt(4 df sum |h|^2/S), of a frequency series."""
    good = np.isfinite(psd) & (psd > 0)
    return np.sqrt(4.0 * df * np.sum(np.abs(hf[good])**2 / psd[good]))


# -------------------------------------------------------------- MBHB model --- #

def chirp_mass(m1, m2):
    return (m1 * m2)**0.6 / (m1 + m2)**0.2


AU_SI = 149597870700.0


def lisa_arm_vectors(t, kappa=0.0, lambda0=0.0):
    """Unit arm vectors n_ij(t) of the analytic equal-arm LISA orbit (first order in e).

    Returns an array of shape (3, 3, nt, 3): ``n[i, j]`` points from spacecraft
    i to spacecraft j (zero on the diagonal).
    """
    t = np.atleast_1d(t)
    alpha = 2.0 * np.pi * t / YRSID_SI + kappa
    ecc = LISA_L / (2.0 * np.sqrt(3.0) * AU_SI)
    pos = []
    for k in range(3):
        bk = 2.0 * np.pi * k / 3.0 + lambda0
        x = AU_SI*np.cos(alpha) + 0.5*ecc*AU_SI*(np.cos(2*alpha - bk) - 3*np.cos(bk))
        y = AU_SI*np.sin(alpha) + 0.5*ecc*AU_SI*(np.sin(2*alpha - bk) - 3*np.sin(bk))
        z = -np.sqrt(3.0)*ecc*AU_SI*np.cos(alpha - bk)
        pos.append(np.stack([x, y, z], axis=-1))
    n = np.zeros((3, 3, t.size, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                d = pos[j] - pos[i]
                n[i, j] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return n


def polarization_tensors(lam, beta, psi):
    """Plus and cross polarization tensors for a source at ecliptic (lam, beta)."""
    sl, cl, sb, cb, sp, cp = np.sin(lam), np.cos(lam), np.sin(beta), np.cos(beta), np.sin(psi), np.cos(psi)
    p = np.array([sl*cp - cl*sb*sp, -cl*cp - sl*sb*sp, cb*sp])
    q = np.array([-sl*sp - cl*sb*cp, cl*sp - sl*sb*cp, cb*cp])
    return np.outer(p, p) - np.outer(q, q), np.outer(p, q) + np.outer(q, p)


def antenna_patterns_ae(t, lam, beta, psi):
    """Long-wavelength antenna patterns (F+, Fx) of A and E along time ``t``.

    Uses the three Michelson-like combinations X, Y, Z built from the arm
    vectors and A = (Z - X)/sqrt2, E = (X - 2Y + Z)/sqrt6. Output shape (2, 2, nt):
    [channel A/E, polarization +/x].
    """
    n = lisa_arm_vectors(t)
    ep, ec = polarization_tensors(lam, beta, psi)

    def arm(i, j, e):
        return np.einsum('ta,ab,tb->t', n[i, j], e, n[i, j])

    F = {}
    for name, e in (('p', ep), ('c', ec)):
        X = 0.5 * (arm(0, 1, e) - arm(0, 2, e))
        Y = 0.5 * (arm(1, 2, e) - arm(1, 0, e))
        Z = 0.5 * (arm(2, 0, e) - arm(2, 1, e))
        F[name] = ((Z - X)/np.sqrt(2.0), (X - 2.0*Y + Z)/np.sqrt(6.0))
    return np.array([[F['p'][0], F['c'][0]], [F['p'][1], F['c'][1]]])


class MBHBGenerator:
    """MBHB signals in A, E on the rfft grid of a time series.

    PhenomD/HM amplitude, phase and t(f) come from BBHx (``PhenomHMAmpPhase``,
    evaluated directly on the grid). The LISA response is the long-wavelength
    approximation with the rotating-constellation antenna patterns evaluated at
    t(f) (stationary phase) and a TDI transfer ~ x sin(x), x = 2 pi f L / c.
    Overall normalisation is arbitrary: use ``target_snr`` in ``simulate_dataset``.

    NOTE: BBHx's full response (``BBHWaveformFD``) would be preferable, but it does
    not run with lisaanalysistools 2.x (orbit ABI mismatch). Swap it in here when
    a matched bbhx/lisatools install is available.

    Parameters
    ----------
    n_samples, dt : length and cadence of the time series.
    modes : harmonics passed to BBHx (PhenomHM if more than the (2,2)).
    """

    def __init__(self, n_samples, dt, modes=((2, 2),), force_backend='cpu'):
        if BBHWaveformFD is None:
            raise ImportError("BBHx is needed to generate MBHB signals.")
        from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
        self.n_samples = n_samples
        self.dt = dt
        self.tobs = n_samples * dt
        self.freqs = sft.rfftfreq(n_samples, dt)
        self.df = self.freqs[1]
        self.modes = list(modes)
        self._ampphase = PhenomHMAmpPhase(run_phenomd=(self.modes == [(2, 2)]),
                                          force_backend=force_backend)

    def __call__(self, m1, m2, chi1, chi2, dist_mpc, phi_ref, inc, lam, beta, psi, t_merger):
        """Return an array of shape (3, n_freqs) with the A, E, T frequency series.

        ``t_merger`` is the merger time in seconds since the start of the data.
        T is returned as zeros (it carries no signal in the long-wavelength limit).
        """
        if m2 > m1:
            m1, m2, chi1, chi2 = m2, m1, chi2, chi1
        f = self.freqs[1:]
        self._ampphase.run_wave(m1, m2, chi1, chi2, dist_mpc * 1e6 * PC_SI, phi_ref, 0.0,
                                t_merger, f.size, freqs=f, modes=self.modes)
        amp = np.asarray(self._ampphase.amp)[0]      # (n_modes, nf)
        phase = np.asarray(self._ampphase.phase)[0]
        tf = np.asarray(self._ampphase.tf)[0]

        out = np.zeros((3, self.freqs.size), dtype=complex)
        x = 2.0 * np.pi * f * LISA_L / CLIGHT
        transfer = x * np.sin(x)
        ci = np.cos(inc)
        for (ell, mm), a, ph, t in zip(self.modes, amp, phase, tf):
            # only the (2,2) angular factor is implemented
            hp = 0.5 * (1.0 + ci**2) * a * np.exp(-1j * ph)
            hc = -1j * ci * a * np.exp(-1j * ph)
            inside = (t >= 0.0) & (t <= self.tobs)
            F = antenna_patterns_ae(np.clip(t, 0.0, self.tobs), lam, beta, psi)
            for c in range(2):
                out[c, 1:] += np.where(inside, transfer * (F[c, 0]*hp + F[c, 1]*hc), 0.0)
        return out


DEFAULT_SOURCE = dict(m1=1.0e6, m2=8.0e5, chi1=0.2, chi2=0.1, dist_mpc=3.0e4,
                      phi_ref=0.64, inc=1.0, lam=4.05, beta=-1.08, psi=1.23,
                      t_merger=None)


def _source_in_window(p, n_samples, dt, modes, pre=86400.0, post=86400.0):
    """A, E, T frequency series of one source as seen in the window [0, n_samples dt].

    The source is generated on a longer grid that starts ``pre`` seconds before the
    window and extends ``post`` seconds past the merger (if it is later than the
    window end), transformed to the time domain and cut to the window. This is how
    real data see a source merging after the end of the data: a chirp that is
    simply truncated in time, without frequency-domain wrap-around.
    """
    tobs = n_samples * dt
    n_pre = int(round(pre / dt))
    t_end = max(tobs, p['t_merger'] + post)
    n_gen = sft.next_fast_len(n_pre + int(np.ceil(t_end / dt)), real=True)
    gen = MBHBGenerator(n_gen, dt, modes=modes)
    q = dict(p, t_merger=p['t_merger'] + n_pre * dt)
    h_td = sft.irfft(gen(**q), n=n_gen, axis=-1) / dt
    return sft.rfft(h_td[:, n_pre:n_pre + n_samples], axis=-1) * dt


def simulate_dataset(sources, n_samples, dt, rng=None, add_noise=True, target_snr=None,
                     Sa=SA_NOMINAL, Si=SI_NOMINAL, modes=((2, 2),)):
    """Simulate A, E (and T) data with any number of MBHBs.

    Parameters
    ----------
    sources : list of dicts with the keys of ``DEFAULT_SOURCE`` (missing keys
        take the defaults). ``t_merger`` in seconds since the start of the data;
        it may be later than the end of the data (pre-merger tracking).
    target_snr : optional list (same length as ``sources``); if given, each
        source's distance is rescaled so that its A+E SNR *within the window*
        equals that value.

    Returns
    -------
    dict with
        ``freqs``, ``psd``, ``dt``, ``tobs``,
        ``data_fd`` (3, nf) : signal + noise in A, E, T,
        ``signals_fd`` (nsrc, 3, nf) : each injection separately,
        ``snr`` (nsrc,) : A+E optimal SNR of each injection in the window,
        ``sources`` : the parameters actually used.
    """
    rng = np.random.default_rng() if rng is None else rng
    freqs = sft.rfftfreq(n_samples, dt)
    df = freqs[1]
    psd = noise_psd_ae(freqs, Sa, Si)

    signals, snrs, used = [], [], []
    for i, src in enumerate(sources):
        p = {**DEFAULT_SOURCE, **src}
        if p['t_merger'] is None:
            p['t_merger'] = 0.9 * n_samples * dt
        h = _source_in_window(p, n_samples, dt, modes)
        snr = np.hypot(compute_snr(h[0], psd, df), compute_snr(h[1], psd, df))
        if target_snr is not None:
            h *= target_snr[i] / snr
            p['dist_mpc'] *= snr / target_snr[i]
            snr = target_snr[i]
        signals.append(h)
        snrs.append(snr)
        used.append(p)

    signals = np.array(signals) if signals else np.zeros((0, 3, freqs.size), complex)
    data = signals.sum(axis=0) if len(signals) else np.zeros((3, freqs.size), complex)
    if add_noise:
        # only A and E are used downstream; T gets the same noise model as a placeholder
        data = data + np.array([gaussian_noise_fd(psd, df, rng) for _ in range(3)])

    return dict(freqs=freqs, psd=psd, dt=dt, tobs=n_samples * dt, data_fd=data,
                signals_fd=signals, snr=np.array(snrs), sources=used)


def gen_timeseries_from_psd(S, fs, rng=None):
    """Time series from a *one-sided* PSD via iFFT (chi2 amplitudes, random phases).

    Kept from the original script (``genTimeSeriesFromPSD2``).
    """
    rng = np.random.default_rng() if rng is None else rng
    N = 2 * (len(S) - 1)
    Ak = np.sqrt(N * S / 2 * fs)
    Ak = np.append(Ak, Ak[1:-1][::-1])
    rphi = np.zeros(N)
    rphi[1:N//2-1] = 2*np.pi*rng.uniform(0, 1, N//2-2)
    rphi[N//2] = 2*np.pi*round(rng.uniform(0, 1))
    rphi[N//2+1:N-1] = -rphi[1:N//2-1][::-1]
    X = Ak * np.sqrt(rng.chisquare(2, size=N)/2) * np.exp(1j*rphi)
    return np.fft.irfft(X, n=N)
