#!/usr/bin/env python

"""
The following code consists of the model that is packaged as a class, Implementation Model().
This will then used by another class to define how the testing phase will go for the 
whole model. The code for this model consists of different representations of data for the model.
"""

__author__      = "Aung Htet"
__email__ = "aung.htet@student.shu.ac.uk"

import nest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import os

class Implementation_Model():
    """Full Model Implementation of Homeostasis and Bump Attractor Model

    The following model uses a nest simulator to define a model for homeostasis to be used
    as perception. This utilizes a population of neurons that are adjusted through gain
    modulation to define how the homeostasis perception will change memory that is retained
    in the bump attractor model.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def init_parameters(self):
        """
            The following method initializes the parameters for different neurons that 
            will be used in the model.
        """
        
        # Homeostasis Model Parameters
        self.homeostasis_parameters={
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

        # Ring Attractor Model Parameters
        self.ring_parameters = {
            "V_m": 0,                       # mV
            "E_L": 0,                       # mV
            "V_th": 0.8,                    # mV
            "V_reset": 0,                   # mV
            "tau_syn_ex": 27,               # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
            "tau_syn_in": 50,               # ms rise time of the inhibitory synaptic alpha function (i.e. to see the current for each spike decays)
            "I_e": 0.45 * (10 ** 6),        # pA input current
            "tau_m": 7.04,                  # ms membrane time constant
            "C_m": 3.96 * 10**6,            # pF membrane capacitance (calculated for R = 1)
            "t_ref": 0                      # ms refractory period
        }

        # Gain Modulation Parameters
        self.gain_mod_parameters = {
            "V_m": 0,                       # mV
            "E_L": 0,                       # mV
            "V_th": 0.8,                    # mV
            "V_reset": 0,                   # mV
            "tau_syn_ex": self.tau_gain,    # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
            "I_e": 0,                       # pA input current
            "tau_m": 4.2,                   # ms membrane time constant
            "C_m": 0.8 * 10**6,             # pF membrane capacitance (calculated for R = 1)
            "t_ref": 0                      # ms refractory period
        }

        # Perirhinal Cortex Parameters
        self.perirhinal_parameters = {
            "V_m": 0,                       # mV
            "E_L": 0,                       # mV
            "V_th": 0.8,                    # mV
            "V_reset": 0,                   # mV
            "tau_syn_ex": 100,              # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
            "tau_syn_in": 50,               # ms rise time of the inhibitory synaptic alpha function (i.e. to see the current for each spike decays)
            "I_e": 0.45 * (10 ** 6),        # pA input current
            "tau_m": 7.04,                  # ms membrane time constant
            "C_m": 3.96 * 10**6,            # pF membrane capacitance (calculated for R = 1)
            "t_ref": 0                      # ms refractory period
        }

        # Perirhinal Decision Parameters
        self.perirhinal_decision_parameters = {
            "V_m": 0,                       # mV
            "E_L": 0,                       # mV
            "V_th": 0.8,                    # mV
            "V_reset": 0,                   # mV
            "tau_syn_ex": 75,               # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
            "tau_syn_in": 50,               # ms rise time of the inhibitory synaptic alpha function (i.e. to see the current for each spike decays)
            "I_e": 0.45 * (10 ** 6),        # pA input current
            "tau_m": 7.04,                  # ms membrane time constant
            "C_m": 3.96 * 10**6,            # pF membrane capacitance (calculated for R = 1)
            "t_ref": 0 
        }

        # Perirhinal Vision Parameters
        self.perirhinal_vision_parameters = {
            "V_m": 0,                       # mV
            "E_L": 0,                       # mV
            "V_th": 0.8,                    # mV
            "V_reset": 0,                   # mV
            "tau_syn_ex": 1,                # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
            "I_e": 0,                       # pA input current
            "tau_m": 7.06,                  # ms membrane time constant
            "C_m": 0.8 * 10**6,             # pF membrane capacitance (calculated for R = 1)
            "t_ref": 0                      # ms refractory period
        }

    def init_neurons_recorders(self):
        """
            The following method is a generalised method for initialising the neurons. as
            well as the spike recorders and multimeters associated with it.
        """

        self.init_homeostasis()
        self.init_ring_attractor()
        self.init_gain_mod()
        self.init_perirhinal_cortex()
        self.init_perirhinal_v_cortex()

    def init_homeostasis(self):
        """
            The following method initialises the neurons, spike recorders and multimeters
            required for the homeostasis model.
        """

        # Initialise neurons
        self.high_neuron = nest.Create('iaf_psc_alpha', params = self.homeostasis_parameters)
        self.low_neuron = nest.Create('iaf_psc_alpha', params = self.homeostasis_parameters)

        # Initialise recorders
        self.high_spike = nest.Create('spike_recorder')
        self.high_multimeter = nest.Create('multimeter')
        self.high_multimeter.set(record_from=["V_m"])
        self.low_spike = nest.Create('spike_recorder')
        self.low_multimeter = nest.Create('multimeter')
        self.low_multimeter.set(record_from=["V_m"])

        # Connect recorders
        nest.Connect(self.high_multimeter, self.high_neuron)
        nest.Connect(self.high_neuron, self.high_spike)
        nest.Connect(self.low_multimeter, self.low_neuron)
        nest.Connect(self.low_neuron, self.low_spike)

    def init_ring_attractor(self):
        """
            The following method initialises the neurons, spike recorders and multimeters
            required for the ring attractor model.
            The following method also initialises how connection would be done for ring
            attractor model.
        """

        # Initialise an empty list of neurons and recorders
        self.ring_neurons = []
        self.ring_spike_recorders = []
        self.ring_multimeters = []

        # Create neurons and add to list
        for _ in range(self.population_size):
            # Initialise neurons
            neuron = nest.Create("iaf_psc_alpha", params=self.ring_parameters)

            # Initialise recorders
            spike_recorder = nest.Create('spike_recorder')
            multimeter = nest.Create('multimeter')
            multimeter.set(record_from=["V_m"])

            # Connect recorders
            nest.Connect(multimeter, neuron)
            nest.Connect(neuron, spike_recorder)

            # Save to list
            self.ring_neurons.append(neuron)
            self.ring_spike_recorders.append(spike_recorder)
            self.ring_multimeters.append(multimeter)
        
        # Synapse Connection Management
        distances = self.neuron_distance()
        index_readjust = lambda x, y, max: x + y if (x + y) < max else (x + y) - max   # redefine current index to connect to
        for i_neuron in range(len(self.ring_neurons)):
            for i_to_conn in range(len(distances)):
                index_to_conn = index_readjust(i_to_conn, i_neuron, len(self.ring_neurons))
                syn_dict = {"weight": self.get_weight(distances[i_to_conn])}
                nest.Connect(self.ring_neurons[i_neuron], self.ring_neurons[index_to_conn], syn_spec=syn_dict)

    def init_gain_mod(self):
        """
            The following method initialises the neurons, spike recorders and multimeters
            required for the gain modulation model.
            The following method also initialises how connection would be done for ring
            attractor model.
        """

        # Initialise an empty list of neurons and recorders
        self.move_high_neurons = []
        self.move_high_spike_recorders = []
        self.move_high_multimeters = []
        self.move_low_neurons = []
        self.move_low_spike_recorders = []
        self.move_low_multimeters = []

        # Create neurons and add to list
        for _ in range(self.population_size):
            # Initialise neurons
            move_high_neuron = nest.Create('iaf_psc_alpha', params = self.gain_mod_parameters)
            move_low_neuron = nest.Create('iaf_psc_alpha', params = self.gain_mod_parameters)

            # Initialise recorders
            move_high_spike_recorder = nest.Create('spike_recorder')
            move_high_multimeter = nest.Create('multimeter')
            move_high_multimeter.set(record_from=["V_m"])
            move_low_spike_recorder = nest.Create('spike_recorder')
            move_low_multimeter = nest.Create('multimeter')
            move_low_multimeter.set(record_from=["V_m"])

            # Connect recorders
            nest.Connect(move_high_multimeter, move_high_neuron)
            nest.Connect(move_high_neuron, move_high_spike_recorder)
            nest.Connect(move_low_multimeter, move_low_neuron)
            nest.Connect(move_low_neuron, move_low_spike_recorder)

            # Save to list
            self.move_high_neurons.append(move_high_neuron)
            self.move_high_spike_recorders.append(move_high_spike_recorder)
            self.move_high_multimeters.append(move_high_multimeter)
            self.move_low_neurons.append(move_low_neuron)
            self.move_low_spike_recorders.append(move_low_spike_recorder)
            self.move_low_multimeters.append(move_low_multimeter)
    
    def init_perirhinal_cortex(self):
        """
            The following method initialises the neurons, spike recorders and multimeters
            required for perirhinal cortex function.
        """

        # Initialise an empty list of neurons and recorders
        self.perirhinal_fdn_neurons = []
        self.perirhinal_fdn_spike_recorders = []
        self.perirhinal_fdn_multimeters = []
        
        # Initialise neuron
        self.perirhinal_inhibitory_neuron = nest.Create('iaf_psc_alpha', params = self.perirhinal_parameters)
        self.perirhinal_decision_neuron = nest.Create('iaf_psc_alpha', params = self.perirhinal_decision_parameters)

        # Initialise recorder
        self.perirhinal_inhibitory_spike = nest.Create('spike_recorder')
        self.perirhinal_inhibitory_multimeter = nest.Create('multimeter')
        self.perirhinal_inhibitory_multimeter.set(record_from=["V_m"])
        self.perirhinal_decision_spike = nest.Create('spike_recorder')
        self.perirhinal_decision_multimeter = nest.Create('multimeter')
        self.perirhinal_decision_multimeter.set(record_from=["V_m"])

        # Connect recorder
        nest.Connect(self.perirhinal_inhibitory_multimeter, self.perirhinal_inhibitory_neuron)
        nest.Connect(self.perirhinal_inhibitory_neuron, self.perirhinal_inhibitory_spike)
        nest.Connect(self.perirhinal_decision_multimeter, self.perirhinal_decision_neuron)
        nest.Connect(self.perirhinal_decision_neuron, self.perirhinal_decision_spike)
        
        # Create neurons and add to list
        for _ in range(5):
            # Initialise neurons
            perirhinal_fdn_neuron = nest.Create('iaf_psc_alpha', params = self.perirhinal_parameters)

            # Initialise recorders
            perirhinal_fdn_spike_recorder = nest.Create('spike_recorder')
            perirhinal_fdn_multimeter = nest.Create('multimeter')
            perirhinal_fdn_multimeter.set(record_from=["V_m"])

            # Connect recorders
            nest.Connect(perirhinal_fdn_multimeter, perirhinal_fdn_neuron)
            nest.Connect(perirhinal_fdn_neuron, perirhinal_fdn_spike_recorder)

            # Save to list
            self.perirhinal_fdn_neurons.append(perirhinal_fdn_neuron)
            self.perirhinal_fdn_spike_recorders.append(perirhinal_fdn_spike_recorder)
            self.perirhinal_fdn_multimeters.append(perirhinal_fdn_multimeter)

    def init_perirhinal_v_cortex(self):
        """
            The following method initialises the neurons, spike recorders and multimeters
            required for perirhinal visual cortex function. This part of the model define 
            the calculation for the model.
        """

        # Initialise an empty list of neurons and recorders
        self.perirhinal_v_r_neurons = []

        # Initialise neuron
        self.perirhinal_v_decision_neuron = nest.Create('iaf_psc_alpha', params = self.perirhinal_vision_parameters)

        # Initialise recorder
        self.perirhinal_v_decision_spike = nest.Create('spike_recorder')
        self.perirhinal_v_decision_multimeter = nest.Create('multimeter')
        self.perirhinal_v_decision_multimeter.set(record_from=["V_m"])

        # Connect recorder
        nest.Connect(self.perirhinal_v_decision_multimeter, self.perirhinal_v_decision_neuron)
        nest.Connect(self.perirhinal_v_decision_neuron, self.perirhinal_v_decision_spike)
        
        # Create neurons and add to list
        for _ in range(5):
            # Initialise neurons
            perirhinal_v_r_neuron = nest.Create('poisson_generator', params={"rate": 0})

            # Save to list
            self.perirhinal_v_r_neurons.append(perirhinal_v_r_neuron)

    def init_connections(self):
        """
            The following method contains all the connection between different parts of the model
            required make the model work.
        """

        self.connect_input_homeostasis()
        self.connect_homeostasis_gain()
        if self.test_homeostasis_gain == True:
            self.connect_gain_test()
        else:
            self.connect_gain_ring()
        self.connect_perirhinal_cortex()
        self.connect_perirhinal_v_cortex()

    def connect_input_homeostasis(self):
        """
            The following method contains the mangement of connection between inputs to the model
            externally and from the ring attractor model. These form the inhibition and excitation
            balance for a negative feedback control loop.
        """

        # Synapse Connection Management
        nest.Connect(self.external_input, self.high_neuron, syn_spec={'weight': 6 * 10**3})
        nest.Connect(self.representation_input, self.high_neuron, syn_spec={'weight': -8 * 10**3})
        nest.Connect(self.representation_input, self.low_neuron, syn_spec={'weight': 6 * 10**3})
        nest.Connect(self.external_input, self.low_neuron, syn_spec={'weight': -8 * 10**3})

    def connect_homeostasis_gain(self):
        """
            The following method contains the management of connection between the homeostasis
            mechanism and the gain modulation mechanism.
        """

        # Synapse Connection Management
        for neuron in self.move_high_neurons:
            nest.Connect(self.high_neuron, neuron, syn_spec={'weight': 0.48 * 10**3})
        for neuron in self.move_low_neurons:
            nest.Connect(self.low_neuron, neuron, syn_spec={'weight': 0.48 * 10**3})

    def connect_gain_ring(self):
        """
            The following method contains the management of connection between the gain modulation
            mechanism and the ring attractor model.
        """

        # Calculations of index for connection
        next_calc = lambda x: (x + 1) if ((x + 1) < self.population_size) else 0
        prev_calc = lambda x: (x - 1) if ((x - 1) >= 0) else (self.population_size - 1)
        # Synapse Connection Management
        for index in range(self.population_size):
            # Ring to Gain Mod Neuron Connection
            nest.Connect(self.ring_neurons[index], self.move_high_neurons[index], syn_spec={'weight': 1.9 * 10**3})
            nest.Connect(self.ring_neurons[index], self.move_low_neurons[index], syn_spec={'weight': 1.9 * 10**3})
            # Gain Mod Neuron to Ring Connection
            nest.Connect(self.move_high_neurons[index], self.ring_neurons[prev_calc(index)], syn_spec={'weight': -0.5})
            nest.Connect(self.move_low_neurons[index], self.ring_neurons[next_calc(index)], syn_spec={'weight': -0.5})
    
    def connect_gain_test(self):
        """
            The following method is made to connect Poisson Generators for testing the gain modulation module. These
            Poisson Generators replace the ring attractor model.
        """

        # Calculations of index for connection
        next_calc = lambda x: (x + 1) if ((x + 1) < self.population_size) else 0
        prev_calc = lambda x: (x - 1) if ((x - 1) >= 0) else (self.population_size - 1)
        rate = 5
        ring_poisson = nest.Create('poisson_generator', params={"rate": rate})
        for index in range(self.population_size):
            # Ring to Gain Mod Neuron Connection
            nest.Connect(ring_poisson, self.move_high_neurons[index], syn_spec={'weight': 1.6 * 10**3})
            nest.Connect(ring_poisson, self.move_low_neurons[index], syn_spec={'weight': 1.6 * 10**3})
            # Gain Mod Neuron to Ring Connection
            nest.Connect(self.move_high_neurons[index], self.ring_neurons[next_calc(index)], syn_spec={'weight': 5})
            nest.Connect(self.move_low_neurons[index], self.ring_neurons[prev_calc(index)], syn_spec={'weight': 5})

    def connect_perirhinal_cortex(self):
        """
            The following method contains the management of connection between the ring attractor
            and the perirhinal cortex familiarity discrimination neurons. These includes the perirhinal
            inhibitory network required to normalise the firing rate.
            The following method will also connect the perirhinal cortex back to the homeostasis model.
        """

        for ring_neuron in self.ring_neurons:
            # Ring Attractor to Perirhinal Inhibitory Neuron Connection
            nest.Connect(ring_neuron, self.perirhinal_inhibitory_neuron, syn_spec={'weight': 1 * 10**3})

        for perirhinal_fdn_index in range(int(len(self.perirhinal_fdn_neurons))):
            # Calculate which region of ring attractor to connect
            start = perirhinal_fdn_index * 20
            stop = (perirhinal_fdn_index + 1) * 20
            list_ring = self.ring_neurons[start:stop]
            for ring_neuron in list_ring:
                # Ring Attractor to Perirhinal Familiarity Discrimination Neurons Connection
                nest.Connect(ring_neuron, self.perirhinal_fdn_neurons[perirhinal_fdn_index], syn_spec={'weight': 1 * 10**3})
            
            # Perirhinal Inhibitory to Perirhinal Familiarity Discrimination Neurons Connection
            nest.Connect(self.perirhinal_inhibitory_neuron, self.perirhinal_fdn_neurons[perirhinal_fdn_index], syn_spec={'weight': -2.5 * 10**3})

        # # Perihinal Familiarity Discrimination Neurons to Perihinal Decision Neurons Connection
        nest.Connect(self.perirhinal_fdn_neurons[0], self.perirhinal_decision_neuron, syn_spec={'weight': 0})
        nest.Connect(self.perirhinal_fdn_neurons[1], self.perirhinal_decision_neuron, syn_spec={'weight': 2 * 10**3})
        nest.Connect(self.perirhinal_fdn_neurons[2], self.perirhinal_decision_neuron, syn_spec={'weight': 1 * 10**4})
        nest.Connect(self.perirhinal_fdn_neurons[3], self.perirhinal_decision_neuron, syn_spec={'weight': 2.7 * 10**4})
        nest.Connect(self.perirhinal_fdn_neurons[4], self.perirhinal_decision_neuron, syn_spec={'weight': 4 * 10**4})

    def connect_perirhinal_v_cortex(self):
        """
            The following method contains the management of connection between the perirhinal vision
            cortex and the homeostasis model.
        """

        # Representation to Perirhinal Vision Decision Cortex Connection
        for neuron in self.perirhinal_v_r_neurons:
            nest.Connect(neuron, self.perirhinal_v_decision_neuron, syn_spec={'weight': 6 * 10**5})

    def __init__(self, representation=False, external=False, test_homeostasis_gain = False, tau_gain = 1000):
        """
            The following parameters give a list of attribute initialisation required for the model to work.
            Attributes:
                representation:         True decides the representation input to use a perirhinal vision cortex while False decides
                                        the representation input to use Poisson Generator.
                external:               True decides the external input to use a perirhinal cortex while False decides
                                        the external input to use Poisson Generator.
                test_homeostasis_gain:  True decides to use the ring attractor as a Poisson Generator for testing while False
                                        decides to use the real ring attractor. 
        """

        # Reset Network
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")
        nest.rng_seed = np.random.randint(1, (2**31) - 1)

        # Simulation Parameters
        self.timestep = 10000       # default is 10000
        self.test_homeostasis_gain = test_homeostasis_gain
        self.total_timestep = 50
        self.tau_gain = tau_gain

        # Ring Attractor Parameters
        self.population_size = 100
        self.max_distance = 50
        self.sd_1 = 10
        self.sd_2 = 5

        # Initialise All Region of Neurons and Recorders
        self.init_parameters()
        self.init_neurons_recorders()

        # Inputs To The Model
        if (external == False):
            self.external_input = nest.Create('poisson_generator', params={"rate": 0})
        else:
            self.external_input = self.perirhinal_v_decision_neuron
        if (representation == False):
            self.representation_input = nest.Create('poisson_generator', params={"rate": 0})
        else:
            self.representation_input = self.perirhinal_decision_neuron

        self.init_connections()

        # Initialise Ring Attractor to start activity
        self.start_ring()

    def neuron_distance(self):
        """
            The following method determines how the distance should be distributed between
            each neurons and returns an array of these distances.
        """

        if self.population_size % 2 == 0:
            distance = np.append(0, np.linspace(0, self.max_distance, int(self.population_size/2) + 1)[1:(int(self.population_size/2))])
            distance = np.append(distance, self.max_distance)
            distance = np.append(distance, np.linspace(self.max_distance, 0, int(self.population_size/2) + 1)[1:(int(self.population_size/2))])
        else:
            distance = np.append(0, np.linspace(0, self.max_distance, int(self.population_size/2) + 1)[1:])
            distance = np.append(distance, np.linspace(self.max_distance, 0, int(self.population_size/2) + 1)[:(int(self.population_size/2))])
        return distance

    def get_weight(self, distance):
        """
            The following method calculates the weight where n1 is pre-synaptic and n2 is
            post-synaptic.
        """

        sd1 = self.sd_1/(self.population_size/100)
        sd2 = self.sd_2/(self.population_size/100)
        w = lambda sd1, sd2, x: ((sd2 * np.exp((-(x**2))/(2*(sd1**2)))) - (sd1 * np.exp((-(x**2))/(2*(sd2**2)))))/(sd2 - sd1)
        weight = 3 * w(sd1, sd2, distance)
        return np.around(weight, 3)
    
    def run(self):
        """
            The following method runs the model for a set amount of timestep
        """

        nest.Simulate(self.timestep)
        self.total_timestep += self.timestep
    
    def start_ring(self):
        """
            The following method gives the ring attractor a pulse of activity required to
            hold the bump
        """

        ring_starter = nest.Create('poisson_generator', params={"rate": 200})
        for neuron in self.ring_neurons[0:10]:
            nest.Connect(ring_starter, neuron, syn_spec={'weight': 4.5 * 10**3})
        nest.Simulate(50)
        ring_starter.rate = 0
    
class Implementation_Run():

    def __init__(self):

        self.model = Implementation_Model()

    def record_data(self):
        """
            The following method records the data for all spike records
        """
        tau_test = [900, 1000]
        count_numbers = [1,2,3,4]
        for tau in tau_test:
            for count_number in count_numbers:
                data_set = []
                for run_count in range(21):
                    print(F"Running Number:{count_number}, Running Index: {run_count}")
                    self.model = Implementation_Model(representation=True, external=True, tau_gain=tau)

                    # Experiment setup
                    vision_data = [0, 0, 0, 0, 0]
                    choice_index = [0, 1, 2, 3, 4]
                    for _ in range(int(count_number)):
                        index_change = np.random.choice(choice_index)
                        choice_index.remove(index_change)
                        vision_data[index_change] = 1

                    # Set representation neuron to activate
                    for i in range(len(vision_data)):
                        if vision_data[i] == 1:
                            self.model.perirhinal_v_r_neurons[i].rate = 20
                    self.model.timestep = 50000
                    self.model.run()
                    # Convert Data
                    high_spike = self.model.high_spike.get("events")["times"]
                    low_spike = self.model.low_spike.get("events")["times"]

                    ring_spike_recorders = []
                    for recorder in self.model.ring_spike_recorders:
                        ring_spike_recorders.append(recorder.get("events")["times"])
                    
                    move_high_spike_recorders = []
                    for recorder in self.model.move_high_spike_recorders:
                        move_high_spike_recorders.append(recorder.get("events")["times"])

                    move_low_spike_recorders = []
                    for recorder in self.model.move_low_spike_recorders:
                        move_low_spike_recorders.append(recorder.get("events")["times"])

                    perirhinal_fdn_spike_recorders = []
                    for recorder in self.model.perirhinal_fdn_spike_recorders:
                        perirhinal_fdn_spike_recorders.append(recorder.get("events")["times"])
                    
                    perirhinal_inhibitory_spike = self.model.perirhinal_inhibitory_spike.get("events")["times"]
                    perirhinal_decision_spike = self.model.perirhinal_decision_spike.get("events")["times"]
                    perirhinal_v_decision_spike = self.model.perirhinal_v_decision_spike.get("events")["times"]

                    spike_dict = {"high_spike": high_spike, "low_spike": low_spike, "ring_spike": ring_spike_recorders, "gain_high_spike": move_high_spike_recorders, "gain_low_spike": move_low_spike_recorders, "per_fdn_spike": perirhinal_fdn_spike_recorders, "per_in_spike": perirhinal_inhibitory_spike, "per_de_spike": perirhinal_decision_spike, "per_vde_spike": perirhinal_v_decision_spike}
                    data_set.append(spike_dict)
                    data_file = "exp_data/" + str(tau) + "_tau/data_spike_" + str(count_number)
                    np.save(data_file, data_set)
    
    
# The following code runs the test cases depending on the arguments.
if __name__ == "__main__":
    implement = Implementation_Run()
    implement.record_data()