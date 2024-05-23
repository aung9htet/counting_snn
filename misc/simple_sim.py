import nest
import numpy as np
import matplotlib.pyplot as plt

N = 100
neurons = [0]*N


params = {
            "I_e": 200.0, "tau_m": 20.0
        }

for i in range(N):
    neurons[i] = nest.Create("iaf_psc_alpha", params=params)

A1 = 1.6
A2 = 1.2
sigma1 = 3
sigma2 = 40
W = lambda k: A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2))

noise = nest.Create("poisson_generator")
noise.rate = 10000.0

for i in range(N):
    for j in range(N):
        syn_dict = {"weight": W(i-j)}
        nest.Connect( neurons[i], neurons[j], syn_spec=syn_dict)

# for i in range(10):
#     nest.Connect(noise, neurons[10 + i], syn_spec={"weight": [[1.2]], "delay": 1.0})

spikerecorder = [0]*N

for i in range(N):
    spikerecorder[i] = nest.Create("spike_recorder")
    nest.Connect(neurons[i], spikerecorder[i])

nest.Simulate(1000.0)
plt.figure(2)

for i in range(N):
    events = spikerecorder[i].get("events")
    senders = events["senders"]
    ts = events["times"]
    print(len(ts))
#     plt.plot(ts, [i]*len(ts), "k.")
# plt.show()