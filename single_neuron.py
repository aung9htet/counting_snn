import nest
import matplotlib.pyplot as plt
import numpy as np

poi_gen = nest.Create('poisson_generator')
poi_gen.rate = 500.0
parameters = {
        "V_m": 0.0,
        "E_L": -50,
        "C_m": 80,
        "tau_m": 10,
        "t_ref": 2,
        "V_th": -50,
        "V_reset": -65,
        "tau_syn_ex": 2.25,
        "tau_syn_in": 0.35
        # "tau_syn_ex": 1.5 + np.random.uniform(low=-0.75, high=0.75),
        # "tau_syn_in": 0.7 + np.random.uniform(low=-0.35, high=0.35)
    }
neuron = nest.Create('iaf_psc_alpha', params=parameters)
spikerecorder = nest.Create("spike_recorder")
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"])
nest.Connect(poi_gen, neuron, syn_spec={'weight': 0.1, 'delay': 1.0})
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikerecorder)
nest.Simulate(1000)
dmm = multimeter.get()
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
events = spikerecorder.get("events")
senders = events["senders"]
print(len(senders))
plt.figure(1)
plt.plot(ts,Vms)
plt.show()