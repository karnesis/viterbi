"""Viterbi tracking of MBHB chirps in LISA time-frequency maps.

Pipeline: ``generate_data`` (simulation) -> ``tf`` (whitening, t-f transform,
channel combination) -> this module (tracking, multi-track extraction, plots).

Trackers
--------
``viterbi_first_order``
    The classic SOAP-like tracker: state = frequency bin, allowed jumps given
    explicitly (e.g. -1/0/+1). Optional *local* (free start / free end) mode.
``viterbi_chirp``
    Chirp-aware tracker. State = (frequency bin, chirp-mass bin). The allowed
    frequency jump is predicted from the leading-order chirp law
    ``d f^{-8/3}/dt = -a(Mc)``, i.e. through the Jacobian of the map
    f -> u = f^{-8/3} that makes every inspiral track a straight line of slope
    a(Mc). Because the chirp mass is (almost) conserved along a track, two
    crossing tracks with different Mc keep their own slopes through the crossing.

``viterbi_tc_mc``
    Pre-merger tracker in the conserved chirp parameters. State = (merger time,
    chirp-mass bin); the frequency at every time follows exactly from the
    Newtonian chirp, so slow (sub-bin per step) drifts are tracked without
    accumulating rounding errors, and each track predicts its merger time.

Multiple tracks
---------------
``extract_tracks`` peels tracks one at a time: find the best local path,
neutralise (set to zero, not -inf) the pixels it used, repeat until the best
path score falls below a threshold. Neutral pixels let a later track pass
through an earlier one, which is what happens at a crossing.

NK, NG 2026
"""
import numpy as np
import matplotlib.pyplot as plt

import tf as tfr

G_SI = 6.67430e-11
MSUN_SI = 1.988409870698051e30
CLIGHT = 299792458.0
MTSUN_SI = G_SI * MSUN_SI / CLIGHT**3


# ---------------------------------------------------------------- scoring --- #

def score_map(P, noise_mean=1.0, penalty=1.0, weight=None):
    """Per-pixel additive score: power minus (1 + penalty) x its noise mean.

    With penalty > 0, pure-noise paths drift negative so that local (free
    start / end) tracking only keeps stretches with excess power.
    ``weight`` (nf,) is the number of independent cells per pixel (see
    ``tf.cwt_dof``); the score is then the weak-signal log-likelihood ratio
    n_eff (P/mean - 1 - penalty), comparable across frequencies.
    """
    S = P / noise_mean - (1.0 + penalty)
    return S if weight is None else S * np.asarray(weight)[None, :]


# ------------------------------------------------------- first-order Viterbi --- #

def viterbi_first_order(score, jumps=(-1, 0, 1), jump_cost=None, local=True):
    """Classic Viterbi on a (nt, nf) score map.

    Parameters
    ----------
    jumps : allowed frequency-bin jumps per time step (k_t - k_{t-1}).
    jump_cost : cost (subtracted log prior) of each jump, default 0.
    local : if True a path may start and end at any time (score clipped at 0,
        Smith-Waterman style). If False the path spans the whole map (the
        original behaviour).

    Returns
    -------
    dict(t=..., k=..., score=..., V=..., back=...) for the best path.
    """
    nt, nf = score.shape
    jumps = np.asarray(jumps, dtype=int)
    jump_cost = np.zeros(jumps.size) if jump_cost is None else np.asarray(jump_cost, float)
    V = np.empty((nt, nf))
    back = np.full((nt, nf), -1, dtype=np.int32)    # -1 = path starts here
    V[0] = score[0]
    k = np.arange(nf)
    for t in range(1, nt):
        best = np.full(nf, -np.inf)
        arg = np.full(nf, -1, dtype=np.int32)
        for j, c in zip(jumps, jump_cost):
            kp = k - j
            ok = (kp >= 0) & (kp < nf)
            cand = np.full(nf, -np.inf)
            cand[ok] = V[t-1, kp[ok]] - c
            better = cand > best
            best[better], arg[better] = cand[better], kp[better]
        if local:
            restart = best < 0.0
            best[restart], arg[restart] = 0.0, -1
        V[t] = score[t] + best
        back[t] = arg
    return _backtrack_first_order(V, back, local)


def _backtrack_first_order(V, back, local, end=None):
    nt = V.shape[0]
    if end is None:
        end = np.unravel_index(np.argmax(V), V.shape) if local else (nt - 1, int(np.argmax(V[-1])))
    t, k = end
    ts, ks = [t], [k]
    while t > 0 and back[t, k] >= 0:
        k = back[t, k]
        t -= 1
        ts.append(t)
        ks.append(k)
    return dict(t=np.array(ts[::-1]), k=np.array(ks[::-1]), score=V[end], V=V, back=back)


# ------------------------------------------------------------ chirp Viterbi --- #

def chirp_rate(mc_msun):
    """a(Mc) in d(f^{-8/3})/dt = -a, leading (Newtonian) order, Mc in solar masses."""
    return (256.0 / 5.0) * np.pi**(8.0/3.0) * (mc_msun * MTSUN_SI)**(5.0/3.0)


def chirp_previous_bin(f, mc_grid, dt):
    """Predicted (fractional) previous frequency bin for each (bin, Mc).

    Inverts one time step of the chirp: u_prev = u + a dt with u = f^{-8/3}. On
    the bin index this is the Jacobian dk/du = (dk/dlnf)(dlnf/du).
    """
    lnf = np.log(f)
    u = f[:, None]**(-8.0/3.0) + chirp_rate(np.asarray(mc_grid))[None, :] * dt
    lnf_prev = -3.0/8.0 * np.log(u)
    return np.interp(lnf_prev, lnf, np.arange(f.size), left=-np.inf)


def viterbi_chirp(score, t, f, mc_grid, tol=2, tol_cost=3.0, mc_step=1, mc_cost=3.0,
                  local=True):
    """Chirp-aware Viterbi with state (frequency bin, chirp-mass bin).

    Parameters
    ----------
    score : (nt, nf) additive score (see ``score_map``).
    t, f : time stamps (uniform) and frequency grid (any monotonic grid; log is best).
    mc_grid : chirp-mass bins (detector frame, solar masses), log-spaced.
    tol : allowed deviation (bins) from the predicted frequency jump; absorbs the
        t-f resolution, PN corrections and the merger.
    tol_cost : cost per bin of deviation.
    mc_step : max change of the chirp-mass bin per time step.
    mc_cost : cost per chirp-mass bin change (Mc is conserved along a track).

    Returns the best path as dict(t, k, m, mc, score, V).
    """
    nt, nf = score.shape
    nm = len(mc_grid)
    dt = t[1] - t[0]
    kprev = np.round(chirp_previous_bin(f, mc_grid, dt))       # (nf, nm), may be -inf
    kk = np.arange(nf)[:, None]
    mm = np.broadcast_to(np.arange(nm)[None, :], (nf, nm))

    V = np.empty((nt, nf, nm))
    bk = np.full((nt, nf, nm), -1, dtype=np.int32)              # previous k (-1 = start)
    bm = np.zeros((nt, nf, nm), dtype=np.int16)                 # previous m
    V[0] = score[0][:, None]
    for it in range(1, nt):
        Vp = V[it-1]
        # 1) best previous chirp-mass bin for each (k, m)
        Wm = np.full((nf, nm), -np.inf)
        am = np.zeros((nf, nm), dtype=np.int16)
        for d in range(-mc_step, mc_step + 1):
            shifted = np.full((nf, nm), -np.inf)
            src = np.arange(nm) + d
            ok = (src >= 0) & (src < nm)
            shifted[:, ok] = Vp[:, src[ok]] - mc_cost * abs(d)
            better = shifted > Wm
            Wm[better] = shifted[better]
            am[better] = np.broadcast_to(src[None, :], (nf, nm))[better]
        # 2) best previous frequency bin around the chirp prediction
        best = np.full((nf, nm), -np.inf)
        ak = np.full((nf, nm), -1, dtype=np.int32)
        am_best = np.zeros((nf, nm), dtype=np.int16)
        for o in range(-tol, tol + 1):
            kp = kprev + o
            ok = np.isfinite(kp) & (kp >= 0) & (kp <= kk)
            kpi = np.where(ok, kp, 0).astype(int)
            cand = np.where(ok, Wm[kpi, mm] - tol_cost * abs(o), -np.inf)
            better = cand > best
            best[better] = cand[better]
            ak[better] = kpi[better]
            am_best[better] = am[kpi, mm][better]
        if local:
            restart = best < 0.0
            best[restart] = 0.0
            ak[restart] = -1
        V[it] = score[it][:, None] + best
        bk[it], bm[it] = ak, am_best

    if local:
        it, k, m = np.unravel_index(np.argmax(V), V.shape)
    else:
        it = nt - 1
        k, m = np.unravel_index(np.argmax(V[-1]), V[-1].shape)
    end_score = V[it, k, m]
    ts, ks, ms = [it], [k], [m]
    while it > 0 and bk[it, k, m] >= 0:
        k, m = bk[it, k, m], bm[it, k, m]
        it -= 1
        ts.append(it); ks.append(k); ms.append(m)
    ms = np.array(ms[::-1])
    return dict(t=np.array(ts[::-1]), k=np.array(ks[::-1]), m=ms,
                mc=np.asarray(mc_grid)[ms], score=end_score, V=V)


def chirp_bins(t, f, mc_grid, tc_grid):
    """Fractional frequency-bin index of the Newtonian track f(t; tc, Mc) for every
    time and every (tc, Mc) state, shape (nt, ntc, nm). NaN after coalescence or
    outside the frequency grid."""
    lnf = np.log(f)
    tau = np.asarray(tc_grid)[None, :, None] - np.asarray(t)[:, None, None]      # (nt, ntc, 1)
    a = chirp_rate(np.asarray(mc_grid))[None, None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        lnf_track = -3.0/8.0 * np.log(a * tau)
    lnf_track = np.where(np.isfinite(lnf_track), lnf_track, -np.inf)
    k = np.interp(lnf_track, lnf, np.arange(f.size, dtype=float), left=np.nan, right=np.nan)
    return k


def viterbi_tc_mc(score, t, f, mc_grid, tc_grid, tc_step=1, tc_cost=1.0, mc_step=1, mc_cost=1.0,
                  local=True):
    """Viterbi in the space of the conserved chirp parameters (tc, Mc).

    For a state (tc, Mc) the frequency at every time is given exactly by the
    Newtonian chirp f(t) = [a(Mc) (tc - t)]^{-3/8}, read from the map at the
    nearest bin. The state may drift by ``tc_step`` / ``mc_step`` grid points per
    time step at cost ``tc_cost`` / ``mc_cost`` per point, which absorbs PN
    corrections and the t-f resolution. Unlike ``viterbi_chirp`` the sub-bin
    frequency drift of slow chirps does not accumulate rounding errors, and the
    track comes with an estimate of the merger time ``tc``. Works on any
    monotonic frequency grid; meant for pre-merger tracks (tc after the data).

    Returns the best path as dict(t, k, mc, tc, score).
    """
    nt, nf = score.shape
    mc_grid, tc_grid = np.asarray(mc_grid), np.asarray(tc_grid)
    ntc, nm = tc_grid.size, mc_grid.size
    kf = chirp_bins(t, f, mc_grid, tc_grid)
    valid = np.isfinite(kf)
    K = np.where(valid, np.rint(np.nan_to_num(kf)), 0).astype(np.int32)

    V = np.empty((ntc, nm))
    bj = np.zeros((nt, ntc, nm), dtype=np.int8)       # tc shift to the previous state
    bm = np.zeros((nt, ntc, nm), dtype=np.int8)       # Mc shift to the previous state
    start = np.zeros((nt, ntc, nm), dtype=bool)       # path starts here
    best_val, best_at = -np.inf, None
    for it in range(nt):
        s = np.where(valid[it], score[it][K[it]], -np.inf)
        if it == 0:
            V = s.copy()
            start[0] = True
        else:
            best = np.full((ntc, nm), -np.inf)
            aj = np.zeros((ntc, nm), dtype=np.int8)
            am = np.zeros((ntc, nm), dtype=np.int8)
            for dj in range(-tc_step, tc_step + 1):
                for dm in range(-mc_step, mc_step + 1):
                    cand = np.full((ntc, nm), -np.inf)
                    js = slice(max(0, -dj), ntc - max(0, dj))
                    ms = slice(max(0, -dm), nm - max(0, dm))
                    jp = slice(max(0, dj), ntc - max(0, -dj))
                    mp = slice(max(0, dm), nm - max(0, -dm))
                    cand[js, ms] = V[jp, mp] - tc_cost * abs(dj) - mc_cost * abs(dm)
                    better = cand > best
                    best[better], aj[better], am[better] = cand[better], dj, dm
            if local:
                restart = best < 0.0
                best[restart] = 0.0
                start[it] = restart
            V = s + best
            bj[it], bm[it] = aj, am
        i = np.argmax(V)
        if V.flat[i] > best_val:
            best_val, best_at = V.flat[i], (it,) + np.unravel_index(i, V.shape)
    if not local:
        best_at = (nt - 1,) + np.unravel_index(np.argmax(V), V.shape)
        best_val = V[best_at[1:]]

    it, j, m = best_at
    ts, js, ms = [it], [j], [m]
    while it > 0 and not start[it, j, m]:
        j, m = j + bj[it, j, m], m + bm[it, j, m]
        it -= 1
        ts.append(it); js.append(j); ms.append(m)
    ts, js, ms = np.array(ts[::-1]), np.array(js[::-1]), np.array(ms[::-1])
    return dict(t=ts, k=K[ts, js, ms], mc=mc_grid[ms], tc=tc_grid[js], score=best_val)


def default_tc_grid(tobs, tmin=1800.0, tmax=2 * 3.15576e7, rel=0.03):
    """Merger times after the end of the data, log-spaced in (tc - tobs) with relative
    spacing ``rel`` (a tc step then moves the track by << 1 bin)."""
    n = int(np.ceil(np.log(tmax / tmin) / rel)) + 1
    return tobs + np.geomspace(tmin, tmax, n)


# ------------------------------------------------------------ multi-track --- #

def extract_tracks(score, tracker, max_tracks=5, min_score=0.0, mask_halfwidth=2,
                   post_merger_steps=2, min_length=3, removal='mask', footprint=None,
                   weight=None, penalty=1.0, **tracker_kw):
    """Iteratively extract tracks, removing each one from the score map before
    searching for the next.

    removal :
        'mask'     - the pixels within ``mask_halfwidth`` bins of the track are set
                     to 0 (neutral): the next search cannot reuse them but can
                     cross them. Destroys a second source that runs inside the
                     first one's footprint for long.
        'subtract' - CLEAN-like: at every time step the track's excess power
                     (estimated at its pixel) times the transform's spectral
                     ``footprint`` (relative power vs bin offset, per centre row
                     (nf, H+1) or common (H+1,); footprint[..., 0] = 1)
                     is subtracted, so the pixels return to noise level and a
                     second track running through them keeps its own power.
                     Needs ``weight`` and ``penalty`` of the score map
                     (score = weight (P/mean - 1 - penalty)).
    ``post_merger_steps`` steps around the track end are also neutralised above
    the end frequency (merger/ringdown leftovers, if the merger is in the data).

    Returns a list of track dicts (with an added ``"score"``), best first.
    """
    s = score.copy()
    nt, nf = s.shape
    w = np.ones(nf) if weight is None else np.asarray(weight)
    tracks = []
    for _ in range(max_tracks):
        tr = tracker(s, **tracker_kw)
        tr.pop('V', None)
        tr.pop('back', None)
        if tr['score'] <= min_score:
            break
        if removal == 'subtract':
            g = np.atleast_2d(footprint)
            g = np.broadcast_to(g, (nf, g.shape[1])) if g.shape[0] == 1 else g
            H = g.shape[1] - 1
            for ti, ki in zip(tr['t'], tr['k']):
                excess = max(s[ti, ki] / w[ki] + penalty, 0.0)
                lo, hi = max(0, ki - H), min(nf, ki + H + 1)
                kk = np.arange(lo, hi)
                s[ti, lo:hi] -= w[lo:hi] * excess * g[ki, np.abs(kk - ki)]
        else:
            # mask the whole span swept since the previous step (near-vertical ridges at merger)
            kprev = np.concatenate([[tr['k'][0]], tr['k'][:-1]])
            for ti, k0, k1 in zip(tr['t'], kprev, tr['k']):
                lo, hi = min(k0, k1), max(k0, k1)
                s[ti, max(0, lo - mask_halfwidth):hi + mask_halfwidth + 1] = 0.0
        # merger/ringdown is broadband: neutralise everything above the track around its end
        te, ke = tr['t'][-1], tr['k'][-1]
        if te < nt - 1:
            for ti in range(max(0, te - post_merger_steps), min(nt, te + 1 + post_merger_steps)):
                s[ti, max(0, ke - mask_halfwidth):] = np.minimum(s[ti, max(0, ke - mask_halfwidth):], 0.0)
        if tr['t'].size >= min_length:
            tracks.append(tr)
    return tracks


def backtrack_from_ends(V, back, n_ends=5, min_sep=5):
    """What the original code effectively did: backtrack several end points of ONE
    global Viterbi lattice. Kept for demonstration: the survivor paths merge, so
    all such 'tracks' collapse onto the single best one.
    """
    last = V[-1]
    order = np.argsort(last)[::-1]
    ends = []
    for k in order:
        if all(abs(k - e) >= min_sep for e in ends):
            ends.append(k)
        if len(ends) == n_ends:
            break
    return [_backtrack_first_order(V, back, local=False, end=(V.shape[0]-1, k)) for k in ends]


# ------------------------------------------------------------- evaluation --- #

def true_ridge(P_signal, min_power=1.0):
    """Ground-truth ridge of a noiseless (normalised) map: argmax per time step,
    flagged invalid where the signal power is below ``min_power``."""
    k = np.argmax(P_signal, axis=1)
    valid = P_signal[np.arange(P_signal.shape[0]), k] >= min_power
    return k, valid


def match_track(track, ridge_k, ridge_valid, tol=2):
    """Precision: fraction of the track's pixels within ``tol`` bins of the ridge.
    Recall: fraction of valid ridge steps that the track follows within ``tol``."""
    on = ridge_valid[track['t']] & (np.abs(track['k'] - ridge_k[track['t']]) <= tol)
    precision = on.mean() if track['t'].size else 0.0
    recall = on.sum() / max(1, ridge_valid.sum())
    return precision, recall


def assign_tracks(tracks, ridges, tol=2):
    """Greedy one-to-one assignment of tracks to true sources by recall.

    Returns a (n_sources,) list with (track index, precision, recall) or None.
    """
    scores = np.array([[match_track(tr, k, v, tol) for (k, v) in ridges] for tr in tracks]) \
        if tracks else np.zeros((0, len(ridges), 2))
    out = [None] * len(ridges)
    used = set()
    pairs = sorted(((scores[i, j, 1], i, j) for i in range(len(tracks)) for j in range(len(ridges))),
                   reverse=True)
    for rec, i, j in pairs:
        if rec > 0 and i not in used and out[j] is None:
            out[j] = (i, scores[i, j, 0], rec)
            used.add(i)
    return out


# --------------------------------------------------------------- plotting --- #

TRACK_COLOURS = ['white', 'cyan', 'lime', 'magenta', 'orange', 'pink']


def plot_tracks(t, f, P, tracks=(), ridges=(), ax=None, title=None, vmax=None, cmap='plasma'):
    """Map in (days, log10 f) with recovered tracks (markers) and true ridges (lines)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    days = t / 86400.0
    cf = ax.pcolormesh(days, np.log10(f), P.T, shading='nearest', cmap=cmap,
                       vmin=0, vmax=np.percentile(P, 99.5) if vmax is None else vmax,
                       rasterized=True)
    for (k, v) in ridges:
        kk = np.where(v, k, -1).astype(float)
        kk[kk < 0] = np.nan
        ax.plot(days, np.log10(np.interp(kk, np.arange(f.size), f, left=np.nan)), 'k--', lw=1)
    for i, tr in enumerate(tracks):
        ax.plot(days[tr['t']], np.log10(f[tr['k']]), '.', ms=2.5,
                color=TRACK_COLOURS[i % len(TRACK_COLOURS)], label=f"track {i}")
    ax.set_xlabel('Time [days]')
    ax.set_ylabel(r'$\log_{10} f$ [Hz]')
    if title:
        ax.set_title(title)
    plt.colorbar(cf, ax=ax)
    return ax


# --------------------------------------------------------------- pipeline --- #

def compute_maps(data, channels=(0, 1), transform='cwt', transform_kw=None):
    """Whiten and transform each channel (maps scaled to unit noise power using the
    known PSD); also return the pixel weights and mask width.

    ``data`` is the dict returned by ``generate_data.simulate_dataset`` (or any
    dict with ``data_fd``, ``psd``, ``dt`` and ``tobs``). For the Morlet CWT the
    weights/mask follow analytically from the resolution; for other transforms
    they are measured on white noise (``tf.noise_calibration``).
    """
    transform_kw = {} if transform_kw is None else dict(transform_kw)
    n = int(round(data['tobs'] / data['dt']))
    maps = []
    for c in channels:
        x = tfr.whiten_fd(data['data_fd'][c], data['psd'], n, data['dt'])
        t, f, W = tfr.transform(x, data['dt'], method=transform, **transform_kw)
        maps.append(W)
    # Morlet and STFT have unit noise power by construction (whitened input);
    # superlets are scaled with their measured white-noise response
    weight, halfwidth, footprint = None, 2, None
    if transform in ('cwt', 'morlet', 'wavelet'):
        n_cycles = transform_kw.get('n_cycles', 12.0)
        weight = tfr.cwt_dof(f, n_cycles, t[1] - t[0])
        cal = tfr.noise_calibration(data['dt'], 'cwt', transform_kw)
        halfwidth, footprint = cal['halfwidth'], cal['footprint']
    elif transform in ('superlet', 'superlets', 'super'):
        cal = tfr.noise_calibration(data['dt'], 'superlet', transform_kw)
        weight, halfwidth, footprint = cal['dof'], cal['halfwidth'], cal['footprint']
        maps = [W / np.sqrt(cal['level'])[None, :] for W in maps]
    return dict(t=t, f=f, maps=maps, weight=weight, mask_halfwidth=halfwidth, footprint=footprint)


def track_maps(m, combine='sum', direction=None, tracker='chirp', tracker_kw=None, penalty=1.0,
               max_tracks=5, min_score=0.0, mask_halfwidth=None, post_merger_steps=3,
               removal=None):
    """Combine the channel maps of ``compute_maps`` and extract tracks.

    ``removal``: 'mask' (default) or 'subtract' (see ``extract_tracks``; at high
    SNR the footprint model errors leave residuals far above the noise, which
    produce duplicate tracks, so subtraction is not the default)."""
    tracker_kw = {} if tracker_kw is None else dict(tracker_kw)
    t, f, maps, weight = m['t'], m['f'], m['maps'], m['weight']
    if removal is None:
        removal = 'mask'
    common = dict(min_score=min_score, post_merger_steps=post_merger_steps, removal=removal,
                  footprint=m.get('footprint'), weight=weight, penalty=penalty,
                  mask_halfwidth=m['mask_halfwidth'] if mask_halfwidth is None else mask_halfwidth)

    def run(P, nchan, ntracks):
        S = score_map(P, noise_mean=nchan, penalty=penalty, weight=weight)
        if tracker == 'chirp':
            return S, extract_tracks(S, viterbi_chirp, t=t, f=f, max_tracks=ntracks,
                                     **common, **tracker_kw)
        if tracker == 'tc_mc':
            kw = dict(tracker_kw)
            kw.setdefault('tc_grid', default_tc_grid(t[-1]))
            return S, extract_tracks(S, viterbi_tc_mc, t=t, f=f, max_tracks=ntracks, **common, **kw)
        return S, extract_tracks(S, viterbi_first_order, max_tracks=ntracks, **common, **tracker_kw)

    if combine == 'coherent' and direction is None:
        # two-stage: find the loudest track with the A+E sum, estimate the channel
        # direction from the data along it, then track with the coherent statistic
        _, first = run(tfr.power_map(maps, mode='sum', normalise=False), len(maps), 1)
        direction = tfr.estimate_direction(maps, first[0]['t'], first[0]['k'], normalise=False) \
            if first else np.ones(len(maps))
    P = tfr.power_map(maps, mode=combine, direction=direction, normalise=False)
    nchan = len(maps) if combine in ('sum', 'csd') else 1
    S, tracks = run(P, nchan, max_tracks)
    return dict(t=t, f=f, P=P, score=S, maps=maps, tracks=tracks, weight=weight,
                direction=direction)


def analyse(data, channels=(0, 1), transform='cwt', transform_kw=None, **kw):
    """Whiten, transform, combine channels and extract tracks (``compute_maps`` +
    ``track_maps``; keyword arguments go to ``track_maps``)."""
    return track_maps(compute_maps(data, channels, transform, transform_kw), **kw)


def analyse_channels(data, names=('A', 'E'), transform='cwt', transform_kw=None, **kw):
    """Track each channel on its own and the combination of both.

    The two channels are transformed once. Returns {'A': ..., 'E': ..., 'A+E': ...}
    (``kw`` go to ``track_maps``; ``combine`` applies to the combined case only).
    """
    combine = kw.pop('combine', 'sum')
    m = compute_maps(data, (0, 1), transform, transform_kw)
    out = {}
    for i, name in enumerate(names):
        out[name] = track_maps(dict(m, maps=[m['maps'][i]]), combine='single', **kw)
    out['+'.join(names)] = track_maps(m, combine=combine, **kw)
    return out


def plot_channels(results, ridges=(), xlim=None, ylim=None, title_extra='', vmax=None):
    """One panel per entry of ``analyse_channels``: map, recovered and true tracks.

    ``vmax`` caps the colour scale (in units of the noise mean per channel), useful
    when one loud source hides the others."""
    fig, ax = plt.subplots(1, len(results), figsize=(7 * len(results), 4.5))
    for a, (name, r) in zip(np.atleast_1d(ax), results.items()):
        scores = ', '.join(f"{tr['score']:.0f}" for tr in r['tracks']) or 'none'
        nchan = len(r['maps']) if name.count('+') else 1
        plot_tracks(r['t'], r['f'], r['P'], r['tracks'], ridges, ax=a,
                    vmax=None if vmax is None else vmax * nchan,
                    title=f"{name}{title_extra} (track scores: {scores})")
        if xlim is not None:
            a.set_xlim(*xlim)
        if ylim is not None:
            a.set_ylim(*ylim)
    plt.tight_layout()
    return fig, ax


def source_ridges(data, transform='cwt', transform_kw=None, min_power=2.0):
    """True ridges of each injection: argmax per time of its noiseless A+E map,
    computed with the same transform and noise normalisation as the data."""
    ridges = []
    for h in data['signals_fd']:
        m = compute_maps(dict(data, data_fd=h), (0, 1), transform, transform_kw)
        ridges.append(true_ridge(sum(np.abs(W)**2 for W in m['maps']), min_power=min_power))
    return ridges


def summarise_tracks(results, ridges=None, labels=None, tol=4):
    """Print, for every result and track, its time span, Mc and (if the true
    ridges are given) the fraction of its steps on each source's ridge."""
    for name, r in results.items():
        print(name)
        if not r['tracks']:
            print('  no track above threshold')
        for i, tr in enumerate(r['tracks']):
            line = (f"  track {i}: score {tr['score']:.0f}, days "
                    f"{r['t'][tr['t'][0]] / 86400:.2f}-{r['t'][tr['t'][-1]] / 86400:.2f}")
            if 'mc' in tr:
                line += f", median Mc {np.median(tr['mc']):.3g}"
            if 'tc' in tr:
                line += f", predicted merger day {tr['tc'][-1] / 86400:.1f}"
            if ridges:
                labels = labels or [f'source {j}' for j in range(len(ridges))]
                on = [np.mean(v[tr['t']] & (np.abs(tr['k'] - k[tr['t']]) <= tol)) for (k, v) in ridges]
                line += '; on ridge: ' + ', '.join(f"{lab} {100 * o:.0f}%" for lab, o in zip(labels, on))
            print(line)


if __name__ == '__main__':
    import generate_data as gd

    dt = 10.0
    n = int(30 * 86400 / dt)
    srcs = [dict(m1=2.0e6, m2=1.5e6, t_merger=14 * 86400.0),
            dict(m1=1.5e5, m2=1.0e5, t_merger=27 * 86400.0, lam=1.0, beta=0.5)]
    data = gd.simulate_dataset(srcs, n, dt, rng=np.random.default_rng(0), target_snr=[150, 150])
    f = tfr.log_freq_grid(5e-5, 2e-2, 160)
    mc = np.geomspace(1e4, 3e7, 48)
    res = analyse(data, transform_kw=dict(freqs=f, n_cycles=12, tstep=3600.0),
                  tracker_kw=dict(mc_grid=mc), min_score=50.0)
    for i, tr in enumerate(res['tracks']):
        print(f"track {i}: score {tr['score']:.1f}, days {res['t'][tr['t'][0]]/86400:.1f}-"
              f"{res['t'][tr['t'][-1]]/86400:.1f}, median Mc {np.median(tr['mc']):.3g}")
    plot_tracks(res['t'], res['f'], res['P'], res['tracks'])
    plt.savefig('viterbi_demo.png', dpi=150, bbox_inches='tight')
