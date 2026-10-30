# Viterbi

The Viterbi algorithm for tracking frequency evolution of signals. The software 
follows the implementation of [[`1903.12614`](https://arxiv.org/abs/1903.12614)]. 

The algorithm is used to track a Gravitational Wave (GW) signal on a time-frequency
representation of the data. Applied to the analysis of the future [LISA mission](https://www.esa.int/Science_Exploration/Space_Science/LISA) data.

Dependent on the [`BBHx`](https://github.com/mikekatz04/BBHx) package developed by M. Katz (for simulating MBHB signals) and on the [Superlets](https://github.com/TransylvanianInstituteOfNeuroscience/Superlets) reference implementation (default t-f transform; its `python/` folder must be on the path).

## Code layout

| file | content |
|---|---|
| `viterbi/generate_data.py` | A/E noise PSD, Gaussian noise, MBHB injections (BBHx PhenomD/HM amplitude-phase + long-wavelength LISA response), SNR |
| `viterbi/tf.py` | whitening, t-f transforms (complex superlets, Morlet CWT on a log grid, STFT), white-noise calibration (pixel weights, footprint), A/E channel combination |
| `viterbi/viterbi.py` | trackers (first-order, chirp-aware f x Mc, and pre-merger (tc, Mc) Viterbi), multi-track extraction, evaluation, plotting, `analyse()` / `analyse_channels()` drivers |
| `tests/test_crossing.py` | two MBHBs whose tracks cross (different chirp masses, different merger times) |
| `tests/test_channels.py` | A vs E maps and A+E combination study |
| `viterbi/viterbi_smbh.py` | original script (legacy, superseded by the above) |

Quick start: `cd viterbi && python viterbi.py` (two-source demo, writes `viterbi_demo.png`).

## Environment

Works in the `chimera_env` conda environment once its numpy is the pip-installed 2.4.4
(numba, and hence BBHx/lisatools, refuse numpy 2.5).

N. Galanis, N. Karnesis

<span style="color:gray">Figure below: <i>Left</i>: Time series of a massive Black Hole Binary GW signal in noise. A simualtion of the future LISA data. <i>Center</i>: Time-frequency representation of only the signal time series. <i>Right</i>: The Viterbi tracks recovering the signals in the noise</span>.
![viterbi_gw](images/viterbi_example_gws.png)
