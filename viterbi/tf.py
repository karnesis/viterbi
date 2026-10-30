"""Time-frequency representations of (whitened) LISA data and channel combination.

Every transform returns ``(t, f, W)`` with ``W`` complex of shape (nt, nf)
(time first, as the tracker expects). ``power_map`` turns one or more such
maps into the per-pixel statistic the Viterbi tracker sums along a path.

NK, NG 2026
"""
import numpy as np
import scipy.fft as sft
from scipy import ndimage


# -------------------------------------------------------------- whitening --- #

def whiten_fd(data_fd, psd, n_samples, dt):
    """Whiten a frequency series with an (analytic) one-sided PSD.

    Returns the whitened *time series*, normalised to unit variance per sample
    for stationary Gaussian noise with that PSD.
    """
    tobs = n_samples * dt
    with np.errstate(divide='ignore', invalid='ignore'):
        w = data_fd * np.sqrt(2.0 * n_samples / (psd * tobs))
    w = np.where(np.isfinite(w), w, 0.0)
    return sft.irfft(w, n=n_samples)


def whiten_td(data, dt, psd_func):
    """Whiten a time series with a PSD model ``psd_func(f)`` (FFT based)."""
    n = data.size
    f = sft.rfftfreq(n, dt)
    return whiten_fd(sft.rfft(data) * dt, psd_func(f), n, dt)


def whiten_numeric(x, v, norm=1/0.7023319615912207, order=3, wsize=100, dopolyfit=True):
    """Running-median (+ polynomial trend) normalisation of a spectrum ``v(x)``.

    Kept from the original script.
    """
    vs = ndimage.median_filter(v, size=wsize) * norm
    if dopolyfit:
        coeff = np.polyfit(x, np.log(vs), order)
        vs = np.exp(np.polyval(coeff, x))
    return v / vs


# ------------------------------------------------------------- transforms --- #

def log_freq_grid(fmin, fmax, nf):
    return np.geomspace(fmin, fmax, nf)


def stft(x, dt, nperseg, noverlap=None, window='hann', flims=None):
    """Short-time Fourier transform of a whitened series.

    Unlike the original ``produce_spectrogram`` it supports windows and overlap.
    """
    from scipy.signal import ShortTimeFFT, get_window
    noverlap = nperseg // 2 if noverlap is None else noverlap
    win = get_window(window, nperseg)
    win = win / np.sqrt(np.sum(win**2))  # unit noise power per pixel
    sft_ = ShortTimeFFT(win, hop=nperseg - noverlap, fs=1.0/dt, scale_to=None)
    S = sft_.stft(x)                      # (nf, nt)
    t = sft_.t(x.size)
    f = sft_.f
    keep_t = (t >= 0) & (t <= x.size * dt)
    S, t = S[:, keep_t], t[keep_t]
    if flims is not None:
        keep_f = (f >= flims[0]) & (f <= flims[1])
        S, f = S[keep_f], f[keep_f]
    return t, f, S.T


def morlet_cwt(x, dt, freqs, n_cycles=12.0, tstep=None):
    """Constant-Q (Morlet/Gabor) wavelet transform on an arbitrary frequency grid.

    The wavelet at frequency ``f0`` has a Gaussian spectral window of width
    ``sigma_f = f0 / n_cycles`` (time width ``n_cycles / (2 pi f0)``), so the
    frequency resolution is constant in log f, which suits chirps that span
    decades in frequency. Analytic (positive-frequency) wavelets, computed by FFT.

    If ``tstep`` (seconds) is given, the *power* is block-averaged to that time
    step and the complex coefficient is returned at the block centre with its
    magnitude replaced by sqrt(mean power), so ``|W|^2`` is the tile energy.

    Returns ``(t, freqs, W)`` with ``W`` of shape (nt, nf) normalised so that
    unit-variance white noise gives ``<|W|^2> = 1``.
    """
    n = x.size
    # zero padding by the longest wavelet (~ 6 sigma_t) avoids circular wrap-around
    npad = sft.next_fast_len(n + int(6 * n_cycles / (2 * np.pi * min(freqs)) / dt))
    X = sft.fft(x, n=npad)
    nu = sft.fftfreq(npad, dt)
    step = 1 if tstep is None else max(1, int(round(tstep / dt)))
    nblk = n // step
    W = np.empty((nblk, len(freqs)), dtype=complex)
    for i, f0 in enumerate(freqs):
        sig = f0 / n_cycles
        psi = np.exp(-0.5 * ((nu - f0) / sig)**2) * (nu > 0)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) / npad)   # unit noise power
        w = sft.ifft(X * psi)[:n]
        if step > 1:
            w = w[:nblk*step].reshape(nblk, step)
            pw = np.mean(np.abs(w)**2, axis=1)
            mid = w[:, step//2]
            W[:, i] = np.sqrt(pw) * np.exp(1j * np.angle(mid))
        else:
            W[:, i] = w
    t = (np.arange(nblk) + 0.5) * step * dt
    return t, np.asarray(freqs), W


def cwt_dof(freqs, n_cycles, tstep):
    """Independent t-f cells per pixel of a (block-averaged) Morlet map.

    This is the Jacobian between pixels and Heisenberg cells. The noise power of
    a Gaussian wavelet of time width sigma_t = n_cycles/(2 pi f) has
    autocorrelation exp(-tau^2 / (2 sigma_t^2)), i.e. an integrated correlation
    time sqrt(2 pi) sigma_t, so a tile of length ``tstep`` holds
    n_eff = tstep sqrt(2 pi) f / n_cycles independent samples: << 1 at low f
    (consecutive pixels repeat the same information), > 1 at high f.
    Weighting the per-pixel log-likelihood by n_eff stops tracks from
    collecting correlated low-frequency noise.
    """
    return tstep * np.sqrt(2.0 * np.pi) * np.asarray(freqs) / n_cycles


def cwt_halfwidth_bins(freqs, n_cycles, nsigma=3.0):
    """Half width (in bins of a log grid) of the wavelet's spectral footprint."""
    dlnf = np.log(freqs[1] / freqs[0])
    return int(np.ceil(nsigma / n_cycles / dlnf))


def block_average(W, step):
    """Average the power of a full-resolution complex map (nt, nf) over blocks of
    ``step`` samples; the phase is taken at the block centre."""
    if step <= 1:
        return W
    nblk = W.shape[0] // step
    w = W[:nblk*step].reshape(nblk, step, -1)
    return np.sqrt(np.mean(np.abs(w)**2, axis=1)) * np.exp(1j * np.angle(w[:, step//2]))


def superlet(x, dt, freqs, c1=3, order=(4, 12), tstep=None):
    """Adaptive (fractional) complex superlet transform, Moca et al. (2021).

    Uses ``superletcx`` from https://github.com/TransylvanianInstituteOfNeuroscience/Superlets
    (its ``python/`` folder must be on the path). At each frequency the wavelets
    have c1, 2 c1, ..., order c1 cycles, with ``order`` spread linearly over the
    frequency *indices* from order[0] to order[1]; the magnitude is their geometric
    mean (time resolution of the short wavelets, frequency resolution of the
    long ones) and the phase their average, so the output is complex and can be
    used with every channel combination.

    Returns ``(t, f, W)`` with ``W`` (nt, nf) complex; the power is block-averaged
    to ``tstep`` like in ``morlet_cwt``. Row normalisation is left to ``power_map``.
    """
    try:
        import superletcx
    except ImportError as e:
        raise ImportError("Superlets not found: clone TransylvanianInstituteOfNeuroscience/Superlets "
                          "and add its python/ folder to the path, or use morlet_cwt.") from e
    W = superletcx.superlets(x, 1.0 / dt, np.asarray(freqs), c1, tuple(order)).T  # (n, nf)
    step = 1 if tstep is None else max(1, int(round(tstep / dt)))
    W = block_average(W, step)
    t = (np.arange(W.shape[0]) + 0.5) * step * dt
    return t, np.asarray(freqs), W


_NOISE_CALIBRATION = {}


def noise_calibration(dt, method, kw, duration=10 * 86400.0, seed=12345, rho_min=0.011):
    """Pixel weights and mask width of a transform, measured on white noise.

    Whitened data are unit-variance white noise plus signal, so the noise
    correlations of the map depend only on the transform. They are measured
    once (cached) on a white-noise realisation at full time resolution:

    * ``dof`` (nf,): independent cells per pixel of length ``tstep``,
      tstep / tau(f), with tau the integrated autocorrelation time of the noise
      power in each row (the pixel -> Heisenberg-cell Jacobian, cf. ``cwt_dof``);
    * ``halfwidth``: first frequency-bin lag at which the noise power
      correlation between rows drops below ``rho_min`` (exp(-4.5), i.e. 3 sigma
      for a Gaussian kernel), used to mask a track's spectral footprint;
    * ``level`` (nf,): mean noise power per row, to normalise the data maps
      without estimating the noise level from the data themselves;
    * ``footprint`` (nf, halfwidth+1): relative power of a tone centred on each
      row vs bin offset, used to subtract found tracks (``viterbi.extract_tracks``).
    """
    key = (dt, method, repr(sorted(kw.items())), duration)
    if key in _NOISE_CALIBRATION:
        return _NOISE_CALIBRATION[key]
    n = int(duration / dt)
    x = np.random.default_rng(seed).normal(size=n)
    full = dict(kw, tstep=None)
    _, f, W = transform(x, dt, method=method, **full)
    # discard the edges (one longest-wavelet length on each side is plenty)
    cut = n // 5
    P = np.abs(W[cut:-cut])**2
    level = P.mean(axis=0)
    P = P / level[None, :] - 1.0
    m = P.shape[0]
    acf = sft.irfft(np.abs(sft.rfft(P, n=2*m, axis=0))**2, axis=0)[:m]
    acf /= acf[0][None, :]
    tau = np.empty(P.shape[1])
    for k in range(P.shape[1]):
        neg = np.nonzero(acf[:, k] <= 0)[0]
        L = neg[0] if neg.size else m
        tau[k] = 1.0 + 2.0 * np.sum(acf[1:L, k])
    tstep = kw.get('tstep') or dt
    dof = (tstep / dt) / tau
    # row-to-row noise power correlation rho_k(lag), per row (the resolution may
    # change along the grid, e.g. superlet orders), averaged over +-lag
    nf = P.shape[1]
    Z = (P - P.mean(axis=0)) / P.std(axis=0)
    maxlag = nf // 2
    rho = np.zeros((nf, maxlag + 1))
    rho[:, 0] = 1.0
    for lag in range(1, maxlag + 1):
        c = np.mean(Z[:, :-lag] * Z[:, lag:], axis=0)          # corr(k, k+lag), (nf-lag,)
        up = np.concatenate([c, np.full(lag, np.nan)])
        dn = np.concatenate([np.full(lag, np.nan), c])
        rho[:, lag] = np.nanmean(np.stack([up, dn]), axis=0)
        if np.nanmedian(rho[:, lag]) < rho_min:
            rho = rho[:, :lag + 1]
            break
    below = np.nonzero(np.nanmedian(rho, axis=0) < rho_min)[0]
    halfwidth = int(below[0]) if below.size else rho.shape[1] - 1
    # spectral footprint of a tone (nf, halfwidth+1): for Gaussian-like kernels the
    # tone's power profile across rows is the square of the noise power correlation
    footprint = np.clip(np.nan_to_num(rho[:, :halfwidth + 1]), 0.0, 1.0)**2
    _NOISE_CALIBRATION[key] = dict(dof=dof, halfwidth=halfwidth, level=level, footprint=footprint)
    return _NOISE_CALIBRATION[key]


def transform(x, dt, method='cwt', **kw):
    """Dispatch to one of the transforms above."""
    method = method.lower()
    if method in ('cwt', 'morlet', 'wavelet'):
        return morlet_cwt(x, dt, **kw)
    if method in ('stft', 'fft', 'spectrogram'):
        return stft(x, dt, **kw)
    if method in ('superlet', 'superlets', 'super'):
        return superlet(x, dt, **kw)
    raise ValueError(f"Unknown transform '{method}'")


# ---------------------------------------------------- channel combination --- #

def normalise_rows(P, robust=True):
    """Divide every frequency row by its noise level estimated from the map itself.

    Robust version uses the median of an exponential (chi2_2/2) distribution,
    median = ln 2 * mean, which is barely affected by a few loud signal pixels.
    It fails for long-lived, slowly chirping signals that occupy the same rows
    for most of the data (the median *is* the signal there): when the PSD is
    known, normalise with the transform's white-noise response instead
    (``noise_calibration``), as ``viterbi.compute_maps`` does.
    """
    level = np.median(P, axis=0) / np.log(2.0) if robust else np.mean(P, axis=0)
    return P / level[None, :]


def _unit_noise(maps, normalise):
    if not normalise:
        return [np.asarray(m) for m in maps]
    return [m / np.sqrt(np.median(np.abs(m)**2, axis=0) / np.log(2.0))[None, :] for m in maps]


def power_map(maps, mode='sum', direction=None, csd=None, normalise=True):
    """Per-pixel detection statistic from one or more complex t-f maps.

    Parameters
    ----------
    maps : list of complex arrays (nt, nf), one per TDI channel (e.g. [A, E]),
        each already whitened by its own PSD.
    normalise : if True, estimate each row's noise level from the map (median,
        see ``normalise_rows``); set False when the maps already have unit noise
        power (e.g. scaled with ``noise_calibration``). Not used by 'csd'.
    mode :
        'single'  - |W_0|^2 (first channel only, what the original code did)
        'sum'     - sum_c |W_c|^2 / sigma_c^2. Log-likelihood ratio (maximised
                    over an unknown complex amplitude per channel) for
                    independent channels. Invariant to any unitary re-mixing of
                    the channels (e.g. the A/E rotation with polarisation).
        'coherent' - |u^H W|^2 for a fixed unit vector ``direction`` u in
                    channel space (rank-1 signal model: E = r A with r constant).
                    Halves the noise degrees of freedom per pixel.
        'csd'     - w^H C^{-1} w with a per-frequency 2x2 cross-spectral matrix
                    ``csd`` of shape (nf, nc, nc) (correlated / unequal noise).
                    Reduces to 'sum' for C = diag(1, 1); invariant to *any*
                    invertible linear transformation of the channels.

    Returns the statistic normalised so that noise has unit mean per channel
    degree of freedom, i.e. mean = number of channels combined (1 or 2).
    """
    if mode == 'csd':
        maps = [np.asarray(m) for m in maps]
        w = np.stack(maps, axis=-1)                 # (nt, nf, nc)
        Ci = np.linalg.inv(csd)                     # (nf, nc, nc)
        return np.real(np.einsum('tfi,fij,tfj->tf', np.conj(w), Ci, w))
    maps = _unit_noise(maps, normalise)
    if mode == 'single':
        return np.abs(maps[0])**2
    if mode == 'sum':
        return sum(np.abs(m)**2 for m in maps)
    if mode == 'coherent':
        # projection weights are in noise units since the maps have unit noise
        u = np.asarray(direction, dtype=complex)
        u = u / np.linalg.norm(u)
        return np.abs(sum(np.conj(uc) * m for uc, m in zip(u, maps)))**2
    raise ValueError(f"Unknown combination mode '{mode}'")


def estimate_direction(maps, it, ik, normalise=True):
    """Best fixed channel combination along a track: principal eigenvector of
    sum_track w w^H, with w the noise-normalised channel coefficients."""
    normed = _unit_noise(maps, normalise)
    w = np.stack([m[it, ik] for m in normed])            # (nc, n_track)
    _, vecs = np.linalg.eigh(w @ np.conj(w.T))
    return vecs[:, -1]


def estimate_csd(maps):
    """Robust per-frequency channel covariance from the t-f coefficients themselves."""
    w = np.stack(maps, axis=-1)                     # (nt, nf, nc)
    C = np.einsum('tfi,tfj->fij', w, np.conj(w)) / w.shape[0]
    return C
