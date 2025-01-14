import numpy as np
import matplotlib.pyplot as plt
from statutils import smbhb, ucb, noise_model
import scipy.fft as sft
from scipy.stats import chi2
import matplotlib.colors as colors
from itertools import cycle
from scipy import ndimage
from tqdm import tqdm
from scipy.signal import find_peaks
import time

colourlist = cycle(['pink', 'white', 'gray', 'magenta'])

YRSID_SI = 3.154e7

import sys
sys.path.append('/Users/nikos/work/Git/Software/Superlets/python/')
from superlet import superlets

Sa = np.log10((3.e-15)**2)  # m^2/sec^4/Hz
Si = np.log10((15.e-12)**2) # m^2/Hz

# Test if there is cupy / GPU available
try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice
    gpu_available = True
    print("\n > Found CUPY, I'll try to use it!\n")
except ModuleNotFoundError:
    import numpy as xp
    gpu_available = False

def whiten_numeric(x, v, norm=1/0.7023319615912207, order=3, wsize=100, dopolyfit=True):
    vs = ndimage.median_filter(v, size=wsize)*norm
    if dopolyfit:
        coeff = np.polyfit(x, np.log(vs), order)
        trnd = np.exp( np.polyval(coeff, x) )
        vs = trnd
    return v/vs

def track(input, T=np.zeros(3), maxt=1, thres=2, 
            detrend=True, f=None, whiten=True, 
                offset=50, peak_dist=50, winsize=100, 
                    nperseg=None): 
    
    # T - transition probabilities (up, horizontal, down)

    c = np.swapaxes(input, 0, 1).copy()
    x = np.arange(c.shape[1]) if f is None else np.log(f)
    V = np.zeros(c.shape) # V array
    A = np.zeros(c.shape, dtype=int) # index array

    if whiten:
        for kk in tqdm(range(0, c.shape[0]), ncols=60, desc="Preprocessing"):
            c[kk,:] = whiten_numeric(x, c[kk, :], wsize=winsize)

    # plt.figure()
    # nnn = np.linspace(start=0, stop=c.shape[0]-1, num=15).astype(int)
    # for kk in nnn: 
    #     plt.loglog(f, c[kk, :], label=f"j={kk}")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel("$c_{j,k}$")
    # plt.savefig('../figures/ckj_7.pdf', bbox_inches='tight')
    # plt.close('all')

    # V[0, :] = c[0, :] # initiate first row
    # baseline_Start = thres * np.median(V[0, offset:]) # Get number of potential tracks
    # peaks_start, _ = find_peaks(V[0, offset:], distance=peak_dist, height=baseline_Start, prominence=baseline_Start)
    # peaks_start+=offset

    # plt.figure()
    # plt.loglog(f, V[0, :], label=f"j={0}")
    # plt.plot(f[peaks_start], V[0, peaks_start], "x")
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel("$V_{0,k}$")
    # plt.show()
    # plt.savefig('../figures/Vkj_peaks_start.pdf', bbox_inches='tight')
    # plt.close('all')
    # num_tracks = maxt if len(peaks_start) > maxt else len(peaks_start)

    if nperseg is None:
        nperseg = c.shape[0]
    nsegs = np.floor(c.shape[0]/nperseg)

    for seg in range(int(nsegs)):
        V[seg*nperseg, :] = c[seg*nperseg, :] # initiate first row
        for j in tqdm(range(1, nperseg), ncols=70, desc=f"Viterbi segment {seg}/{int(nsegs)}"):
            # init for first freq bin, can't be for down transition
            k = 0 # k - freq bin, j - time bin
            idx = np.argmax(T[1:3] + V[seg*nperseg + j-1, k:k+2]) + 1
            V[seg*nperseg + j, k] = c[seg*nperseg + j, k] + T[idx] + V[seg*nperseg + j-1, k-1+idx]
            A[seg*nperseg + j, k] = idx - 1
            
            for k in range(1, c.shape[1]-1): # main for loop. cases k=0 and k=last are outside the loop because some steps are out-
                idx = np.argmax(T + V[seg*nperseg + j-1, k-1:k+2])        # -of bounds
                V[seg*nperseg + j, k] = c[seg*nperseg+j, k] + T[idx] + V[seg*nperseg + j-1, k-1+idx]
                A[seg*nperseg + j, k] = idx - 1

            k = c.shape[1] - 1
            idx = np.argmax(T[0:2] + V[seg*nperseg + j-1, k-1:k+1])
            V[seg*nperseg + j, k] = c[seg*nperseg + j, k] + T[idx] + V[seg*nperseg + j-1, k-1+idx]
            A[seg*nperseg + j, k] = idx - 1

        # nnn = np.array_split(np.arange(c.shape[0]), c.shape[0]/100)
        # for nn in list(nnn[:5]) + list(nnn[-5:]): 
        #     plt.figure()
        #     for kk in nn:
        #         lbl = f"j={kk}" if kk==nn[0] or kk==nn[-1] else None
        #         plt.loglog(f, V[kk, :], label=lbl)
        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #     plt.xlabel('Frequency [Hz]')
        #     plt.ylabel("$V_{j,k}$")
        #     plt.savefig(f'../figures/batch_super/sparce_t_v[{nn[0]}:{nn[-1]},j].pdf', bbox_inches='tight')
        #     plt.close('all')

        # for nn in list(nnn[:5]) + list(nnn[-5:]): 
        #     plt.figure()
        #     for kk in nn:
        #         lbl = f"j={kk}" if kk==nn[0] or kk==nn[-1] else None
        #         plt.loglog(f, c[kk, :], label=lbl)
        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #     plt.xlabel('Frequency [Hz]')
        #     plt.ylabel("$c_{j,k}$")
        #     plt.savefig(f'../figures/batch_super/sparce_t_c[{nn[0]}:{nn[-1]},j].pdf', bbox_inches='tight')
        #     plt.close('all')
        # breakpoint()
        # plt.figure()
        # cc = c.copy()
        # x = np.arange(cc.shape[1]) if f is None else np.log(f)
        # for kk in nnn: 
        #     get_trend = np.polyfit(x, np.log(cc[kk,:]), 3)
        #     predicted = np.exp( np.polyval(get_trend, x) )
        #     plt.loglog(c[kk, :], label=f"j={kk}")
        #     plt.loglog(predicted, 'k--')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel("$c_{j,k}$")
        # plt.savefig('../figures/vkj_4.pdf', bbox_inches='tight')
        # plt.close('all')

        # plt.figure()
        # for kk in nnn:      
        #     norm=1/0.7023319615912207
        #     smthd = ndimage.median_filter(c[kk, :], size=1000) * norm
        #     qq=c[kk, :]/smthd
        #     get_trend = np.polyfit(x, np.log10(qq), 3)
        #     predicted = 10**np.polyval(get_trend, x)
        #     plt.loglog(qq, label=f"j={kk}")
        #     plt.loglog(predicted, color='gray', linestyle='--')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel("$c_{j,k}$")
        # plt.savefig('../figures/vkj_5.pdf', bbox_inches='tight')
        # plt.close('all')

        # plt.figure()
        # for kk in nnn:      
        #     norm=1/0.7023319615912207
        #     smthd = ndimage.median_filter(c[kk, :], size=1000) * norm
        #     qq=c[kk, :]/smthd
        #     get_trend = np.polyfit(x, np.log10(qq), 3)
        #     predicted = 10**np.polyval(get_trend, x)
        #     plt.loglog(qq/predicted, label=f"j={kk}")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel("$c_{j,k}$")
        # plt.savefig('../figures/vkj_6.pdf', bbox_inches='tight')
        # plt.close('all')
        # np.savetxt('vjk.txt', x)
        # breakpoint()
        # nmaxindcs = np.argsort(V[-1,:])[-num_tracks:]
        
        # Idea: simplify data (create zeros below threshold for example,
        # and then search for peaks - make our lives easier)
        baseline_End = thres * np.median(c[seg*nperseg, offset:])
        peaks_end, _ = find_peaks(c[seg*nperseg, offset:], height=baseline_End, prominence=baseline_End)
        peaks_end+=offset

        # plt.figure()
        # plt.loglog(f, V[-1, :], label="j=end")
        # plt.plot(f[peaks_end], V[-1, peaks_end], "x")
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel("$V_{\mathrm{end},k}$")
        # plt.savefig('../figures/Vkj_peaks_end.pdf', bbox_inches='tight')
        # plt.close('all')
        # plt.figure()
        # plt.loglog(f, c[-1, :], label="j=end")
        # plt.plot(f[peaks_end], c[-1, peaks_end], "x")
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel("$V_{\mathrm{end},k}$")
        # plt.show()
        # plt.savefig('../figures/ckj_peaks_end.pdf', bbox_inches='tight')
        # plt.close('all')

        # choose peaks
        peaks_found = np.sort(peaks_end)

        if 'num_tracks' not in locals():
            num_tracks = maxt if len(peaks_found) > maxt else len(peaks_found)

        if 'sol' not in locals():
            sol = np.zeros((num_tracks, c.shape[0]), dtype=int) # solution array
            # sol.fill(np.nan)
            # sol[:] = np.nan
        # breakpoint()
        for t in range(num_tracks):
            # breakpoint()
            # fig, ax = plt.subplots(figsize=(10, 8))
            # ax.pcolormesh(np.arange(c.shape[0]), np.log10(f), c.T, shading='nearest', cmap='YlOrBr', snap=True)
            # breakpoint()
            sol[t, int(-(nsegs - (seg+1) )*nperseg)-1] = peaks_found[t] # np.argmax(V[-1, :]) # find maximum likelihood and backtrack solution
            for i in range(1, nperseg):
                idx = sol[t, int(-i -(nsegs - (seg+1) )*nperseg)]
                sol[t, int(-i -1 -(nsegs - (seg+1) )*nperseg)] = idx + A[int(-i -(nsegs - (seg+1) )*nperseg), idx]
            # ax.plot(np.arange(c.shape[0]), np.log10(f[sol[t]]), 'pink')
            # plt.show()
            # breakpoint()
            # plt.close('all')
    return sol


def viterbi_multiple_tracks(data, f=None, flims=None, Tobs=None, dt=None, T=np.zeros(3), 
                nperseg=1000, signal=None, method='superlets', order=8, const=20, offset=20,
                    clrmap="plasma", maxini_tracks=5, thres=25, extrasmooth=False, peak_dist=20,
                        winsize=100, doplot=True, figname='viterbi.pdf', marker='.'):

    """ 
    A function to plot the time series signal and signal + noise, the whitened spectrogram and the most probable path according to the viterbi algorithm.

    Inputs
    ---------
    Tobs : total observation time
    dt   : sampling interval
    noise_spectrum : lambda function of frequency that produces the noise spectrum
    signal : lambda function of time that produces the injected signal
    T : transition matrix for the viterbi algorithm jumps. Shape is (3,) . Elements 0, 1, 2 are the log probabilities for the DCU transitions respectively. Default is 0 (no preference) 
    nperseg : number of data points per segment for the Short Time Fourier Transform. Default is 1000
    overlap :  number of data points overlap between segments. Default is 0 (no overlap)

    Outputs
    ---------
    Plot 1 : signal and signal+noise time series
    Plot 2 : whitened spectrogram of the data
    Plot 3 : most probable frequency path of the source accoring to the viterbi algorithm
    
    """
    if Tobs is None or dt is None:
        raise Exception("Ahh, I need Tobs and dt!")
    if method.lower() in ['superlets', 'superlet', 'super'] and f is None:
        raise Exception("Ahh, I need a frequency vector if the method chosen is 'superlets'!")  

    time_vec = np.linspace(0, Tobs, data.size, endpoint=True)

    print("Computing time-frequency representation of the data...")
    startt = time.time()
    if method.lower() in ['superlets', 'superlet', 'super']:
        whitened_ts = whiten(data, dt)
        f = f[1:]
        wspec = superlets(whitened_ts, 1/dt, f, const, [order])
        t = np.linspace(0,Tobs, num=(wspec.shape[1]))
    else:
        t, f, wspec = produce_spectrogram(Tobs, dt, data, flims, nperseg=nperseg, whiten=True)    
    print(f"Elapsed {time.time() - startt}")

    print("Preparing Viterbi algorithm.")
    sol = track(wspec, T=np.zeros(3), maxt=maxini_tracks, offset=offset, winsize=winsize,
                    thres=thres, f=f, whiten=extrasmooth, peak_dist=peak_dist, nperseg=nperseg)
    
    if doplot:
        print("Generating plots")
        if signal is not None:
            fig, ax = plt.subplots(1, 3)
            fig.set_figwidth(22)
            ax[0].plot(time_vec, data)
            ax[0].plot(time_vec, signal, 'crimson', alpha=.6)
            ax[0].set_xlabel('Time [sec]')
            ax[0].set_ylabel('Strain amplitude')
            if method.lower() in ['superlets', 'superlet', 'super']:
                whitened_sig_ts = whiten(signal, dt)
                print("Computing superlet transform for the signal-only part")
                startt = time.time()
                wspecs = superlets(whitened_sig_ts, 1/dt, f, const, [order])
                # ax[1].imshow(wspecs, cmap=clrmap, interpolation="none", origin="lower", aspect="auto", 
                #                     extent=[0, len(data)*dt, np.log10(f[1]), np.log10(f[-1])])
                print(f"Elapsed {time.time() - startt}")
                ax[1].pcolorfast(t, np.log10(f), wspecs[1:, 1:], cmap=clrmap)
                ax[1].set_ylabel('log-Frequency [Hz]')
            else:
                _, _, tf_signal = produce_spectrogram(Tobs, dt, signal, flims, nperseg=nperseg, whiten=True)
                ax[1].pcolormesh(t, np.log10(f), tf_signal, shading='nearest', cmap=clrmap, snap=True)
                ax[1].set_ylabel('log-Frequency [Hz]')

            ax[1].set_xlabel('Time [sec]')

            if method.lower() in ['superlets', 'superlet', 'super']:
                # cf = ax[2].imshow(wspec, cmap=clrmap, interpolation="none", origin="lower", aspect="auto", 
                #                 extent=[0, len(data)*dt, np.log10(f[1]), np.log10(f[-1])]) 
                cf = ax[2].pcolorfast(t, np.log10(f), wspec[1:, 1:], cmap=clrmap)               
            else:
                cf = ax[2].pcolormesh(t, np.log10(f), wspec, shading='nearest', cmap=clrmap, snap=True)

            fig.colorbar(cf, ax=ax[2])
            path = np.zeros((sol.shape[0], t.shape[0]))
            for tr in range(sol.shape[0]):
                for i, id in enumerate(sol[tr,:]):
                    path[tr, i] = f[id]

            if method.lower() in ['superlets', 'superlet', 'super']:
                for tr in range(sol.shape[0]):
                    ax[2].plot(t, np.log10(path[tr]), markersize=1, marker=marker, 
                                            linestyle='None', markerfacecolor='None', color=next(colourlist))
            else:
                for tr in range(sol.shape[0]):
                    ax[2].plot(t, np.log10(path[tr]), markersize=1, marker=marker, 
                                            linestyle='None', markerfacecolor='None', color=next(colourlist))

            ax[2].set_xlabel('Time [sec]')
            plt.tight_layout()
            
        else:
            fig, ax = plt.subplots(1, 2)
            fig.set_figwidth(14)
            ax[0].plot(time_vec, data)
            ax[0].set_xlabel('Time [sec]')
            ax[0].set_ylabel('Strain amplitude')

            if method.lower() in ['superlets', 'superlet', 'super']:
                cf = ax[1].imshow(wspec, cmap=clrmap, interpolation="none", origin="lower", aspect="auto", 
                                extent=[0, len(data)*dt, f[1], f[-1]]) 
            else:
                cf = ax[1].pcolormesh(t, np.log10(f), wspec, shading='nearest', cmap=clrmap, snap=True)
            fig.colorbar(cf, ax=ax[1])
            path = np.zeros((sol.shape[0], t.shape[0]))
            for tr in range(sol.shape[0]):
                for i, id in enumerate(sol[tr,:]):
                    path[tr, i] = f[id]
            if method.lower() in ['superlets', 'superlet', 'super']:
                for tr in range(sol.shape[0]):
                    ax[1].plot(t, path[tr], markersize=1, marker=marker, 
                                            linestyle='None', markerfacecolor='None', color=next(colourlist))
                ax[1].set_ylabel('Frequency [Hz]')
            else:
                for tr in range(sol.shape[0]):
                    ax[1].plot(t, np.log10(path[tr]), markersize=1, marker=marker, 
                                            linestyle='None', markerfacecolor='None', color=next(colourlist))
                ax[1].set_ylabel('log-Frequency [Hz]')

            ax[1].set_xlabel('Time [sec]')
            plt.tight_layout()
        plt.savefig(figname, bbox_inches='tight')
        plt.show()

    return t, f, wspec, sol


def viterbi(data, f=None, flims=None, Tobs=None, dt=None, T=np.zeros(3), 
                nperseg=1000, signal=None, method='superlets', order=8, const=20,
                    clrmap="plasma"):

    """ 
    A function to plot the time series signal and signal + noise, the whitened spectrogram and the most probable path according to the viterbi algorithm.

    Inputs
    ---------
    Tobs : total observation time
    dt   : sampling interval
    noise_spectrum : lambda function of frequency that produces the noise spectrum
    signal : lambda function of time that produces the injected signal
    T : transition matrix for the viterbi algorithm jumps. Shape is (3,) . Elements 0, 1, 2 are the log probabilities for the DCU transitions respectively. Default is 0 (no preference) 
    nperseg : number of data points per segment for the Short Time Fourier Transform. Default is 1000
    overlap :  number of data points overlap between segments. Default is 0 (no overlap)

    Outputs
    ---------
    Plot 1 : signal and signal+noise time series
    Plot 2 : whitened spectrogram of the data
    Plot 3 : most probable frequency path of the source accoring to the viterbi algorithm
    
    """
    if Tobs is None or dt is None:
        raise Exception("Ahh, I need Tobs and dt!")
    if method.lower() in ['superlets', 'superlet', 'super'] and f is None:
        raise Exception("Ahh, I a frequency vector if the method chosen is 'superlets'!")  

    time = np.linspace(0, Tobs, data.size, endpoint=True)

    print("Computing time-frequency representation of the data...")
    if method.lower() in ['superlets', 'superlet', 'super']:
        whitened_ts = whiten(data, dt)
        wspec = superlets(whitened_ts, 1/dt, f[1:], const, [order])
        t = np.linspace(0,Tobs, num=(wspec.shape[1]))
    else:
        t, f, wspec = produce_spectrogram(Tobs, dt, data, flims, nperseg=nperseg, whiten=True)    
    
    print("Starting Viterbi algorithm.")
    sol = track(wspec, T=np.zeros(3), thres=25)

    print("Generating plots")
    if signal is not None:
        
        fig, ax = plt.subplots(1, 3)
        fig.set_figwidth(22)

        ax[0].plot(time, data)
        if method.lower() in ['superlets', 'superlet', 'super']:
            whitened_sig_ts = whiten(signal, dt)
            wspecs = superlets(whitened_sig_ts, 1/dt, f[1:], const, [order])**2
            ax[1].pcolormesh(t, np.log10(f), wspecs, shading='nearest', cmap=clrmap)
        else:
            _, _, tf_signal = produce_spectrogram(Tobs, dt, signal, flims, nperseg=nperseg, whiten=True)
            ax[1].pcolormesh(t, np.log10(f), tf_signal, shading='nearest', cmap=clrmap)
            ax[1].set_ylabel('log-Frequency [Hz]')

        ax[1].set_xlabel('Time [sec]')

        if method.lower() in ['superlets', 'superlet', 'super']:
            cf = ax[2].imshow(wspec, cmap=clrmap, interpolation="none", origin="lower", aspect="auto", 
                            extent=[0, len(data)*dt, f[1], f[-1]]) 
        else:
            cf = ax[2].pcolormesh(t, np.log10(f), wspec, shading='nearest', cmap=clrmap)

        fig.colorbar(cf, ax=ax[2])
        path = np.zeros_like(t)
        for i, id in enumerate(sol):
            path[i] = f[id]

        if method.lower() in ['superlets', 'superlet', 'super']:
            ax[2].plot(t, path, linestyle='--', color='pink')
            ax[2].set_ylabel('Frequency [Hz]')
        else:
            ax[2].plot(t, np.log10(path), linestyle='--', color='pink')
            ax[2].set_ylabel('log-Frequency [Hz]')

        ax[2].set_xlabel('Time [sec]')
        plt.tight_layout()
        
    else:
        fig, ax = plt.subplots(1, 2)
        fig.set_figwidth(14)

        ax[0].plot(time, data)

        if method.lower() in ['superlets', 'superlet', 'super']:
            cf = ax[1].imshow(wspec, cmap=clrmap, interpolation="none", origin="lower", aspect="auto", 
                            extent=[0, len(data)*dt, f[1], f[-1]]) 
        else:
            cf = ax[1].pcolormesh(t, np.log10(f), wspec, shading='nearest', cmap=clrmap)
        fig.colorbar(cf, ax=ax[1])
        path = np.zeros_like(t)
        for i, id in enumerate(sol):
            path[i] = np.log10(f[id])
        if method.lower() in ['superlets', 'superlet', 'super']:
            ax[1].plot(t, path, linestyle='--', color='pink')
            ax[1].set_ylabel('Frequency [Hz]')
        else:
            ax[1].plot(t, np.log10(path), linestyle='--', color='pink')
            ax[1].set_ylabel('log-Frequency [Hz]')

        ax[1].set_xlabel('Time [sec]')
        plt.tight_layout()

    return t, f, wspec


def produce_spectrogram(Tobs, dt, data, flims, nperseg=1000, whiten=False, plot=False):
    
    Np = int(Tobs/dt + 1)
    f_min = flims[0]
    f_max = flims[1]
    
    time = np.linspace(0, Tobs, data.size, endpoint=True)

    Ns = nperseg
    t = np.zeros(Np//Ns)
    f = sft.rfftfreq(Ns, dt)
    g = np.zeros((f.size, t.size))
    

    for i in range(g.shape[1]):
        t[i] = time[i*Ns+Ns//2]
        fft = sft.rfft(data[i*Ns:(i+1)*Ns])
        g[:, i] = np.real(fft*np.conjugate(fft))/Ns

    filt = np.logical_and(f>=f_min, f<=f_max)
    g = g[filt, :]
    f = f[filt]
    
    if whiten:
        lisa_noise_model = noise_model(f)
        Sn = lisa_noise_model([Sa, Si]).squeeze()

        for k in range(g.shape[0]):
            g[k, :] = g[k, :]/Sn[k]

    if plot:
        fig, ax = plt.subplots(1, 2)
        fig.set_figwidth(14)

        ax[0].plot(time, data)
        
        ax[1].pcolormesh(t, np.log10(f), g, shading='nearest')
        ax[1].set_ylabel('Frequency [Hz]')
        ax[1].set_xlabel('Time [sec]')
        plt.tight_layout()

    return t, f, g


def genTimeSeriesFromPSD2(S,fs):
    """ genTimeSeriesFromPSD (S,fs)

    A simple function to generate time series from a given *one-sided* power spectrum
    via iFFT. It draws the amplitudes from a chi2 distribution, while it asigns
    a random phase between [-2*pi, 2*pi].

    NK 2019
    """

    N    = 2*(len(S)-1)
    Ak   = np.sqrt(N*S/2*fs)             # Take the sqrt of the amplitude
    Ak   = np.append(Ak, Ak[1:-1][::-1]) # make two-sided
    rphi = np.zeros(N)

    rphi[1:int(N/2)-1]   = 2*np.pi*np.random.uniform(0, 1, int(N/2)-2)  # First half
    rphi[int(N/2)]       = 2*np.pi*round(np.random.uniform(0, 1))       # mid point
    rphi[int(N/2+1):N-1] = -rphi[1:int(N/2)-1][::-1]                    # reflected half

    X = Ak*np.sqrt(chi2.rvs(2, size=N)/2) * np.exp(1j * rphi)

    return np.fft.irfft(X,n=int(N)) # return Inverse FFT



def ComputeSNR(f, d, S, fmin=None, fmax=None):

    imax = len(f)-1
    df   = f[1] - f[0]
    if (fmax != None):
        imax = np.argwhere(f >=fmax)[0][0]
    imin = 1
    if (fmin != None):
        imin = np.argwhere(f >=fmin)[0][0]

    SNR = 2 * np.sqrt(df * np.sum(np.real(d[imin:imax]*np.conjugate(d[imin:imax])/S[imin:imax])))
    return SNR

def produce_gbs_signal(DOPLOT=False, t_remain=2, rseed=42, dt=1,  flims=[1e-5, 1.5e-2],
                      params = [-20.37729133, 2.61301, -16.53675992, 3.12184095, 0.05407993, 0.80372815, 1.76872, 0.1012303],
                      t_obs = 1/12,
                      ):
    params = np.array(params)
    n_binaries = params.ndim
    params = np.atleast_2d(params)
    df = 1/(t_obs*YRSID_SI)
    ndata = int(t_obs*YRSID_SI/dt)
    # Carefully select the length of the frequency vector
    if (ndata % 2)==0:              # Get the number of requencies
        nfft = int((ndata/2)+1)
    else:
        nfft = int((ndata+1)/2)
    F    = df*nfft                 # make the positive frequency vector
    fvec = np.arange(0, F, df)        

    # init
    As = xp.zeros_like(fvec).astype(xp.complex128)
    Es = xp.zeros_like(fvec).astype(xp.complex128)

    for i in range(n_binaries):
        p0 = np.array([list(params[i])])

        # Get the waveform settings
        f0 = p0.squeeze()[1]
        f0_lims = np.array([f0-1e-3, f0+1e-3]) * 1e-3 # <

        buffer = 200 # N-points as a buffer
        fmin = f0_lims[0] - buffer * df
        fmax = f0_lims[1] + buffer * df
        start_freq_ind = int(fmin / df)
        end_freq_ind = int(fmax / df)

        # Evaluate the model - If GPU is found we'll get cuda arrays
        template = ucb(f_lims=f0_lims, f_inds=[start_freq_ind, end_freq_ind + 1], \
                    gpu=gpu_available, tobs=t_obs*YRSID_SI, dt=dt)
        Asi, Esi, _ = template.eval(p0)

        # Add them up
        As[start_freq_ind:end_freq_ind+1] += Asi.squeeze()
        Es[start_freq_ind:end_freq_ind+1] += Esi.squeeze()

    return fvec, As, Es

def produce_smbhb_signal(DOPLOT=False, t_remain=2, rseed=42, dt=1,  flims=[1e-5, 1.5e-2],
                    m1   = 4.956676e+06,
                    m2   = 4.067167e+06,
                    c1   = -0.523733,
                    c2   = -0.117412, 
                    dist = 6109, # 61097.116076 
                    inc  = 1.420048,
                    beta = -1.081082,
                    lam  = 4.052963,
                    psi  = 1.228444,
                    phi0 = 0.641716,
                    t_obs = 1/12,
                    ):

    #------------------------------------- Run Settings Below ----------------------------- #

    # WF parameters
   

    # Noise parameters: Nominal values for the SciRD noise
    Sa = np.log10((3.e-15)**2)  # m^2/sec^4/Hz
    Si = np.log10((15.e-12)**2) # m^2/Hz

     # 1.0 # YRS
    npoints = None # int(2e3) # If set to None, it will generate the correct f-vector
    modes = [(2,2)]
    # modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

    #------------------------------------- Run Settings Above ----------------------------- #

    
    # make the plots look a bit nicer with some defaults

    # ----------------------------------------- Case 1: ------------------------------------- #
    #

    df = 1/(t_obs*YRSID_SI) # Get the frequency resolution
    if npoints is not None:
        fvec = np.logspace(np.log10(3e-5), np.log10(1e-1), num=npoints)  # make the positive frequency vector
    else:
        ndata = int(t_obs*YRSID_SI/dt)
        # Carefully select the length of the frequency vector
        if (ndata % 2)==0:              # Get the number of requencies
            nfft = int((ndata/2)+1)
        else:
            nfft = int((ndata+1)/2)
        F    = df*nfft                 # make the positive frequency vector
        fvec = np.arange(0, F, df)        

    # Get the PSD of the noise using the model above
    lisa_noise_model = noise_model(fvec)
    Sn = lisa_noise_model([Sa, Si])
    Sn[0] = 0.0

    # ------------------------------ Generate some SMBHB signal --------------------------- #      
    #

    hr = 60*60 # hours in sec
    t_ref = t_obs * YRSID_SI - t_remain * hr  # merger time that is a parameter of the mbh
    t_ref_sample = t_ref / hr
    t_obs_start = 1 / 12 * YRSID_SI  # dial backward from merger the start of the waveform
    t_obs_end = 0.0 # include the merger

    # Compute sampling quantities

    if m2>m1: # make sure that m1>m2
        tmp = m2
        m2 = m1
        m1 = tmp
        # now swap spins
        tmp = c2
        c2 = c1
        c1 = tmp
    
    eta  = m1*m2/(m1+m2)**2
    q    = m1/m2
    Mc   = (m1+m2)*eta**0.6
    logD = np.log10(dist)
    ci   = np.cos(inc)
    sb   = np.sin(beta)

    # Get the true parameter values here. Needs to be 2 dim array
    lam  = lam % (2. * np.pi) # should be in (0,2*pi)
    psi  = psi % (1. * np.pi) # should be in (0,pi)
    phi0 = phi0 % (2. * np.pi) # should be in (0,2*pi)
    p0 = np.array([[Mc, q, c1, c2, logD, t_ref_sample, ci, sb, lam, psi, phi0]]) 

    # Get the default test-parameter values

    # Evaluate the model - If GPU is found we'll get cuda arrays
    template = smbhb(f=fvec, modes=modes, tstart=t_obs_start, tend=t_obs_end, gpu=gpu_available)
    As, Es, Ts = template.eval(p0)

    # Ensure we have cupy arrays for the case we use GPUs
    As = xp.array(As)
    Es = xp.array(Es)

    # set random seed. Important in order to sample the same data!
    xp.random.seed(rseed)

    # Generate some random noise for the real data
    n_real = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df)))
    n_imag = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df))) * 1j

    # Ensure we wrap cuda arrays in case a GPU is found
    data_A = xp.array(n_real) + xp.array(n_imag) + As

    n_real = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df)))
    n_imag = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df))) * 1j

    # Ensure we wrap cuda arrays in case a GPU is found
    data_E = xp.array(n_real) + xp.array(n_imag) + Es

    print(data_A.shape)
    # Make sure to get rid of the nan
    data_A[0,0] = 0.0
    data_E[0,0] = 0.0

    # -------------------------------- Make a test plot ------------------------ #
    # 

    if gpu_available:
        As = As.get()
        Es = Es.get()
        Ts = Ts.get()
        data_A = data_A.get()
        data_E = data_E.get()
        Sn = Sn.get()

    As = As.squeeze()
    Es = Es.squeeze()
    Ts = Ts.squeeze()
    data_A = data_A.squeeze()
    data_E = data_E.squeeze()
    Sn = Sn.squeeze()

    if DOPLOT:
        plt.figure()
        plt.loglog(fvec, np.abs(As), label='A')
        plt.loglog(fvec, np.abs(Es), label='E')
        plt.loglog(fvec, np.abs(Ts), label='T')
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"$\tilde{h}(f)$ (Hz$^{-1/2}$)")
        plt.xlim(flims[0], flims[1])
        #plt.savefig('{}smbh_low_snr_signal_only.png'.format(TAG), dpi=600, bbox_inches='tight')

        plt.figure(figsize=(10,4))
        plt.loglog(fvec, 2*df*np.absolute(data_A)**2 , label='Generated Data', alpha=0.3, color='grey')
        plt.loglog(fvec, 2*df*np.absolute(As)**2 , label='signal', alpha=0.5, color='limegreen')
        plt.loglog(fvec, Sn, label='Noise PSD',color='k', linestyle='-.')
        plt.ylabel('[1/Hz]')
        plt.xlabel('Frequency [Hz]')
        plt.xlim(flims[0], flims[1])
        plt.ylim(1e-45, 1e-38)
        plt.legend(loc='upper left',frameon=False)
        #plt.savefig('{}_data.png'.format(TAG), dpi=600, bbox_inches='tight')
        
    return fvec, data_A, As, data_E, Es, Sn, dt, t_obs * YRSID_SI

def plot_spectrogram(Tobs, dt, data, flims, nperseg=1000, whiten=False, debug=False):
    
    Np = int(Tobs/dt + 1)
    f_min = flims[0]
    f_max = flims[1]
    
    time = np.linspace(0, Tobs, data.size, endpoint=True)

    Ns = nperseg
    t = np.zeros(Np//Ns)
    f = sft.rfftfreq(Ns, dt)
    g = np.zeros((f.size, t.size))

    for i in range(g.shape[1]):
        t[i] = time[i*Ns+Ns//2]
        fft = sft.rfft(data[i*Ns:(i+1)*Ns])
        g[:, i] = np.real(fft*np.conjugate(fft))/Ns

        if debug:
            lisa_noise_model = noise_model(f)
            Sn = lisa_noise_model([Sa, Si]).squeeze()
            plt.figure()
            plt.loglog(f, 2*np.absolute(f[0]-f[1])*np.absolute(fft**2))
            plt.loglog(f, Sn)

    filt = np.logical_and(f>=f_min, f<=f_max)
    g = g[filt, :]
    f = f[filt]
    
    if whiten:
        lisa_noise_model = noise_model(f)
        Sn = lisa_noise_model([Sa, Si]).squeeze()

        for k in range(g.shape[1]):
            g[:, k] = g[:, k]/Sn

    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(14)
    ax[0].plot(time, data)
    cf = ax[1].pcolormesh(t, np.log10(f), g, shading='nearest',norm=colors.LogNorm(vmin=g.min(), vmax=g.max()))
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    fig.colorbar(cf, ax=ax[1])
    plt.tight_layout()

    return None

# function to whiten data
def whiten(data, dt):
    Nt = len(data)
    f = np.fft.rfftfreq(Nt, dt)
    # Get the noise psd
    lisa_noise_model = noise_model(f)
    psd = lisa_noise_model([Sa, Si]).squeeze()
    hf = np.fft.rfft(data)
    # white_hf = hf / (np.sqrt(psd)) # 
    white_hf = hf / np.sqrt(psd) / dt
    white_hf[0] = 0.0
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

