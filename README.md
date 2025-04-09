# Viterbi

The Viterbi algorithm for tracking frequency evolution of signals. The software 
follows the implementation of [1903.12612](https://arxiv.org/abs/1903.12614). 

The algorithm is used to track a Gravitational Wave (GW) signal on a time-frequency
representation of the data. Applied to the analysis of the future [LISA mission](https://www.esa.int/Science_Exploration/Space_Science/LISA) data.

N. Galanis, N. Karnesis

<span style="color:gray">Figure below: <i>Left</i>: Time series of a massive Black Hole Binary GW signal in noise. A simualtion of the future LISA data. <i>Center</i>: Time-frequency representation of only the signal time series. <i>Right</i>: The Viterbi tracks recovering the signals in the noise</span>.
![viterbi_gw](images/viterbi_example_gws.png)
