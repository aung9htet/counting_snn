import nest
import numpy as np
import matplotlib.pyplot as plt

def implement_gain_mod(homeostasis_input, attractor_input, weight = 0.4 * 10**3):
    """
        The following method takes in two parameters to apply gain modulation to the algorithm
        homeostasis_input: The input is a neuron from the homeostasis model.
                           This can either be the higher neuron or the lower neuron
                           based on how it moves the activity in the bump attractor.
        attractor_input: The input is a neuron from the bump attractor.
                         This is connected to the bump attractor to determine how the
                         activity will be transferred.
    """
    parameters = {
        "V_m": 0,                       # mV
        "E_L": 0,                       # mV
        "V_th": 0.8,                    # mV
        "V_reset": 0,                   # mV
        "tau_syn_ex": 1000,             # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
        "I_e": 0,                       # pA input current
        "tau_m": 4.2,                   # ms membrane time constant
        "C_m": 0.8 * 10**6,             # pF membrane capacitance (calculated for R = 1)
        "t_ref": 0                      # ms refractory period
    }

    # Initialise neurons
    neuron = nest.Create('iaf_psc_alpha', params = parameters)

    # Synapse Connection Management
    nest.Connect(homeostasis_input, neuron, syn_spec={'weight': weight})
    nest.Connect(attractor_input, neuron, syn_spec={'weight': 1.5 * 10**3})
    
    return neuron

def implement_homeostasis(external_input, internal_input):
    """
        The following method involves connect to two types of neuron: external and internal.
        If external is higher than internal, this will produce more spikes in high neuron.
        If internal is higher than internal, this will produce more spikes in low neuron.
        external_input: The following is an external neuron. This is the neuron that produces
                        spike in relation to input from the external environment. In this case,
                        we will be inputting spikes based on the number it perceives.
        internal_input: The following is an internal neuron. This is the neuron that produces
                        spike in relation to the bump activity from the ring attractor model.
    """
    parameters = {
        "V_m": 0,                       # mV
        "E_L": 0,                       # mV
        "V_th": 0.8,                    # mV
        "V_reset": 0,                   # mV
        "tau_syn_ex": 75,               # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
        "tau_syn_in": 50,               # ms rise time of the inhibitory synaptic alpha function (i.e. to see the current for each spike decays)
        "I_e": 0.45 * (10 ** 6),        # pA input current
        "tau_m": 7.04,                  # ms membrane time constant
        "C_m": 3.96 * 10**6,            # pF membrane capacitance (calculated for R = 1)
        "t_ref": 0                      # ms refractory period
    }

    # Initialise neurons
    high_neuron = nest.Create('iaf_psc_alpha', params = parameters)
    low_neuron = nest.Create('iaf_psc_alpha', params = parameters)

    # Synapse Connection Management
    nest.Connect(external_input, high_neuron, syn_spec={'weight': 6 * 10**3})
    nest.Connect(internal_input, high_neuron, syn_spec={'weight': -8 * 10**3})
    nest.Connect(internal_input, low_neuron, syn_spec={'weight': 6 * 10**3})
    nest.Connect(external_input, low_neuron, syn_spec={'weight': -8 * 10**3})
    
    # print(spike_recorder.get("events")["times"])
    return high_neuron, low_neuron

def test_homeostasis_gain(rate_ex, rate_in, rate_bump, weight):

    # poisson generator params
    poi_param_ex = {
        "rate": rate_ex   # Hz rate of poisson generator
    }

    poi_param_in = {
        "rate": rate_in   # Hz rate of poisson generator
    }
    
    poi_param_bump = {
        "rate": rate_bump   # Hz rate of poisson generator
    }

    # Reset Network
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")

    # Initialise neuron
    poi_gen_ex = nest.Create('poisson_generator', params=poi_param_ex)
    poi_gen_in = nest.Create('poisson_generator', params=poi_param_in)
    poi_gen_bump = nest.Create('poisson_generator', params=poi_param_bump)

    # Initialise models
    high_neuron, low_neuron = implement_homeostasis(poi_gen_ex, poi_gen_in)
    gain_mod_high = implement_gain_mod(high_neuron, poi_gen_bump, weight)
    gain_mod_low = implement_gain_mod(low_neuron, poi_gen_bump, weight)
    spike_recorder_high = nest.Create('spike_recorder')
    spike_recorder_low = nest.Create('spike_recorder')

    # Measurement Connection Management
    nest.Connect(gain_mod_high, spike_recorder_high)
    nest.Connect(gain_mod_low, spike_recorder_low)

    # Simulation
    nest.Simulate(10000)
    
    # print(spike_recorder.get("events")["times"])
    return len(spike_recorder_high.get("events")["times"])/10, len(spike_recorder_low.get("events")["times"])/10, spike_recorder_high

no_of_test = 5
rate1_range = np.linspace(200, 0, no_of_test)
rate2_range = np.linspace(0, 200, no_of_test)
record_fr = np.empty((len(rate1_range), len(rate2_range)))
weights = np.linspace(0.2, 1, 5)
for r1_i in range(len(rate1_range)):
    for r2_i in range(len(rate2_range)):
        fr,_,_ = test_homeostasis_gain(rate1_range[r1_i], rate2_range[r2_i], 20, weight = 0.4*10**3)
        record_fr[r1_i][r2_i] = fr
        
fig_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
weight_result = []
# for weight in weights:
record_test = []
time = 0
rate = 200
for rate in rate2_range:

    time = 0
    times = []
    _, _, recorder = test_homeostasis_gain(rate, 0, 20, weight=0.4*(10**3))
    window = 300
    total_i = int(np.floor(10000/300))
    fr_list = []
    ts=recorder.get("events")["times"]
    print(ts)
    for i in range(total_i):
        fr_list.append(len([x for x in ts if ((x >= (i * window)) and (x < ((i + 1) * window)))]) * (1000/window))
        time += 300
        times.append(time)
    weight_result.append(fr_list)    

fig, axs = plt.subplots(1,2)
fig.set_size_inches(28,10)

for i in range(len(weight_result)):
    # print(len(times), result)
    result = weight_result[i]
    axs[0].plot(times, result, lw=3.0, color=fig_colors[i + 1])
    axs[0].set_xlabel("Timestep", fontsize = 30)
    axs[0].set_ylabel("HGM Firing Rates", fontsize = 30)
img = axs[1].imshow(record_fr)
axs[1].set_xlabel("External Perception Neuron", fontsize = 30)
axs[1].set_ylabel("Representation Perception Neuron", fontsize = 30)
axs[1].set_xticks(ticks=np.linspace(0, no_of_test - 1,5), labels=np.linspace(0, 200, 5))
axs[1].set_yticks(ticks=np.linspace(0, no_of_test - 1,5), labels=np.linspace(200, 0, 5))
axs[0].tick_params(axis='both', labelsize=25)
axs[1].tick_params(axis='both', labelsize=25)
cbar = fig.colorbar(img, ax=axs[1])
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(25)
plt.show()