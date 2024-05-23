import nest
import numpy as np
import matplotlib.pyplot as plt

def test_homeostasis(rate1, rate2, in_weight = 6, out_weight = -8):
    parameters = {
        "V_m": 0,                       # mV
        "E_L": 0,                       # mV
        "V_th": 0.8,                    # mV
        "V_reset": 0,                   # mV
        "tau_syn_ex": 75,               # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
        "tau_syn_in": 50,               # ms rise time of the inhibitory synaptic alpha function (i.e. to see the current for each spike decays)
        "I_e": 0.45 * (10 ** 6),        # pA input current
        "tau_m": 7.04,                     # ms membrane time constant
        "C_m": 3.96 * 10**6,               # pF membrane capacitance (calculated for R = 1)
        "t_ref": 0                      # ms refractory period
    }

    # poisson generator params
    poi_param_1 = {
        "rate": rate1   # Hz rate of poisson generator
    }

    poi_param_2 = {
        "rate": rate2   # Hz rate of poisson generator
    }
    
    # Reset Network
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")

    # Initialise neurons
    neuron = nest.Create('iaf_psc_alpha', params = parameters)
    poi_gen1 = nest.Create('poisson_generator', params=poi_param_1)
    poi_gen2 = nest.Create('poisson_generator', params=poi_param_2)
    spike_recorder = nest.Create('spike_recorder')
    multimeter = nest.Create('multimeter')
    multimeter.set(record_from=["V_m"])

    # Synapse Connection Management
    nest.Connect(multimeter, neuron)
    nest.Connect(neuron, spike_recorder)
    nest.Connect(poi_gen1, neuron, syn_spec={'weight': out_weight * 10**3})
    nest.Connect(poi_gen2, neuron, syn_spec={'weight': in_weight * 10**3})

    # Simulation
    nest.Simulate(10000)
    
    # print(spike_recorder.get("events")["times"])
    return len(spike_recorder.get("events")["times"])/10, multimeter.get()

no_of_test = 10
rate2_range = np.linspace(200, 0, no_of_test)
rate1_range = np.linspace(0, 200, no_of_test)
record_fr = np.empty((len(rate1_range), len(rate2_range)))
weight_list = [[-8,6], [6, -8]]
fig, axs = plt.subplots(1,2)
fig.set_size_inches(28,10)
imgs = []
for weight_index in range(len(weight_list)):
    weights = weight_list[weight_index]
    for r1_i in range(len(rate1_range)):
        for r2_i in range(len(rate2_range)):
            fr, _ = test_homeostasis(rate1_range[r1_i], rate2_range[r2_i], weights[0], weights[1])
            if fr < 50:
                fr = 0
            record_fr[r1_i][r2_i] = fr
    img = axs[weight_index].imshow(record_fr)
    imgs.append(img)
    axs[weight_index].set_xlabel("External Perception Neuron", fontsize=30)
    if weight_index == 0:
        axs[weight_index].set_ylabel("Representation Perception Neuron", fontsize=30)
    else:
        axs[weight_index].set_yticks(ticks=np.linspace(0, no_of_test - 1,5), labels=[])
    axs[weight_index].tick_params(axis='both', which='both', labelsize=25)
    axs[weight_index].set_xticks(ticks=np.linspace(0, no_of_test - 1,5), labels=np.linspace(0, 200, 5))
cbar = fig.colorbar(imgs[1], ax=axs[1])
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(25)
plt.show()