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
            "tau_syn_ex": 1000,             # ms rise time of the excitatory synaptic alpha function (i.e. to see the current for each spike decays)
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

    def __init__(self, representation=False, external=False, test_homeostasis_gain = False):
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

        # Simulation Parameters
        self.timestep = 10000       # default is 10000
        self.test_homeostasis_gain = test_homeostasis_gain
        self.total_timestep = 50

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

    def get_homeostasis_info(self):
        """
            The following method gets the firing rate of the model for high neuron and low
            neuron
        """

        high_fr = len(self.high_spike.get("events")["times"])/(self.timestep/1000)
        low_fr = len(self.low_spike.get("events")["times"])/(self.timestep/1000)
        return high_fr, low_fr
    
    def get_gain_mod_info(self):
        """
            The following method gives an array of firing rate for each of the gain modulation
            neurons in the model.
        """

        gain_mod_high_fr = []
        gain_mod_low_fr = []
        for spike_recorder in self.move_high_spike_recorders:
            gain_mod_high_fr.append(len(spike_recorder.get("events")["times"])/(self.timestep/1000))
        for spike_recorder in self.move_low_spike_recorders:
            gain_mod_low_fr.append(len(spike_recorder.get("events")["times"])/(self.timestep/1000))
        return gain_mod_high_fr, gain_mod_low_fr
    
    def get_ring_info(self):
        """
            The following method gives an array of firing rate for each of the ring attractor
            neurons in the model.
        """

        ring_fr = []
        for spike_recorder in self.ring_spike_recorders:
            ring_fr.append(len(spike_recorder.get("events")["times"])/(self.timestep/1000))
        return ring_fr
    
    def get_perirhinal_decision_info(self):
        """
            The following neuron gets the firing rate of the model for perirhinal vision decision
            neuron.
        """

        ts = self.perirhinal_decision_spike.get("events")["times"]
        perirhinal_decision_fr = len([x for x in ts if x >= self.total_timestep - 1000])
        return perirhinal_decision_fr
    
    def get_perirhinal_vision_decision_info(self):
        """
            The following neuron gets the firing rate of the model for perirhinal vision decision
            neuron
        """

        ts = self.perirhinal_v_decision_spike.get("events")["times"]
        perirhinal_decision_fr = len([x for x in ts if x >= self.total_timestep - 1000])
        return perirhinal_decision_fr
    
    def get_perirhinal_fdn_info(self):
        """
            The following method gets the firing rate of the model for high neuron and low
            neuron
        """

        perirhinal_fdn_fr = []
        for spike_recorder in self.perirhinal_fdn_spike_recorders:
            perirhinal_fdn_fr.append(len(spike_recorder.get("events")["times"])/(self.timestep/1000))
        return perirhinal_fdn_fr
    
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

    def init_exp_params(self):
        """
            The following method initialises the parameters of the neuron to be used for experimentation
            in the model.
        """
        # Experiment 1 neuron parameters
        self.exp1_parameters={
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

    def __init__(self):

        self.model = Implementation_Model()
        self.init_exp_params()
    
    def plot_homeostasis_heatmap(self):
        """
            The following method runs the run plot homeostasis where it shows the difference in
            firing rate when two inputs to the model differ.
        """

        no_of_test = 10
        external_range = np.linspace(200, 0, no_of_test)
        representation_range = np.linspace(0, 200, no_of_test)
        record_fr_high = np.empty((len(external_range), len(representation_range)))
        record_fr_low = np.empty((len(external_range), len(representation_range)))

        # Run the model
        for external_index in range(len(external_range)):
            for representation_index in range(len(representation_range)):
                # Running the Model
                progress_print ="Current Progress\nExternal Neuron Rate Progress: " + str(external_index + 1) + "/" + str(no_of_test) + "\nRepresentation Neuron Rate Progress: " + str(representation_index + 1) + "/" + str(no_of_test)
                os.system('clear')
                sys.stdout.write(progress_print)
                self.model = Implementation_Model()
                self.model.external_input.rate = external_range[external_index]
                self.model.representation_input.rate = representation_range[representation_index]
                self.model.run()
                record_fr_high[external_index][representation_index], record_fr_low[external_index][representation_index] = self.model.get_homeostasis_info()
        
        # Start Plotting
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(13.5, 5)
        img1 = axs[0].imshow(record_fr_high)
        img2 = axs[1].imshow(record_fr_low)
        titles = ["Homeostasis High Neuron", "Homeostasis Low Neuron"]
        for i in range(2):
            axs[i].set_title(titles[i], weight='bold')
            axs[i].set_xlabel("Poisson Generator 1")
            axs[i].set_ylabel("Poisson Generator 2")
            axs[i].set_xticks(np.linspace(0, no_of_test - 1,5))
            axs[i].set_xticklabels(np.linspace(0, 200, 5))
            axs[i].set_yticks(np.linspace(0, no_of_test - 1,5)) 
            axs[i].set_yticklabels(labels=np.linspace(200, 0, 5))
        plt.colorbar(img1, ax=axs[0])
        plt.colorbar(img2, ax=axs[1])
        plt.show()
    
    def plot_bar_gain_mod(self, test):
        """
            The following method gives a bar chart of all the firing rate of the neurons in gain modulation.
        """

        if test == True:
            self.model = Implementation_Model(test_homeostasis_gain=True)
        fig, axs = plt.subplots(2,1)
        fig.set_size_inches(10,10)
        x_axis = np.arange(self.model.population_size)
        self.model.run()
        gain_mod_high_fr, gain_mod_low_fr = self.model.get_gain_mod_info()
        axs[0].bar(x_axis, gain_mod_high_fr)
        axs[1].bar(x_axis, gain_mod_low_fr)
        bar_title = ["Firing Rate of High Gain Modulation Neuron", "Firing Rate of Low Gain Modulation Neuron"]
        for i in range(2):
            axs[i].set_xlabel("Neuron Index")
            axs[i].set_ylabel("Firing Rate of Neuron")
            axs[i].set_title(bar_title[i])
        plt.show()

    def plot_bar_ring(self):
        """
            The following method gives a bar chart of all the firing rate of the neurons in the ring attractor.
        """

        fig = plt.figure()
        fig.set_size_inches(10,5)
        x_axis = np.arange(self.model.population_size)
        ring_fr = self.model.get_ring_info()
        plt.bar(x_axis, ring_fr)
        bar_title = "Firing Rate of Ring Attractor Neuron"
        plt.xlabel("Neuron Index")
        plt.ylabel("Firing Rate of Neuron")
        plt.title(bar_title)
        plt.show()

    def plot_bar_perirhinal_fdn(self):
        """
            The following method gives a bar chart of all the firing rate of the neurons in the perirhinal cortex's
            fdn neurons.
        """
        
        fig = plt.figure()
        fig.set_size_inches(10,5)
        self.model.representation_input.rate = 0
        self.model.external_input.rate = 0
        self.model.run()
        x_axis = np.arange(int(len(self.model.perirhinal_fdn_neurons)))
        perirhinal_fdn_fr = self.model.get_perirhinal_fdn_info()
        plt.bar(x_axis, perirhinal_fdn_fr)
        bar_title = "Firing Rate of Perirhinal Representation Neuron"
        plt.xlabel("Neuron Index")
        plt.ylabel("Firing Rate of Neuron")
        plt.title(bar_title)
        plt.show()
    
    def plot_ring_weights(self):
        """
            The following method a diagram to illustrate the weights for the ring attractor model.
        """

        distances = self.model.neuron_distance()
        weight_list = []
        for d in distances[50:]:
            weight_list.append(self.model.get_weight(d))
        for d in distances[:50]:
            weight_list.append(self.model.get_weight(d))
        plt.figure()
        plt.plot(weight_list)
        plt.show()
    
    def plot_ring_behavior(self, title = None):
        """
            The following method gives a diagram to illustrate the behavior of neurons in ring attractor
            model.
        """

        plt.figure()
        for index in range(self.model.population_size):
            events = self.model.ring_spike_recorders[index].get("events")
            ts = events["times"]
            plt.plot(ts[ts>50], np.full(shape=len(ts[ts>50]),fill_value=index,dtype=np.int64), "k.")
        if not title is None:
            plt.title(title)
        plt.ylim([0, self.model.population_size])
        plt.xlabel("Timestep (0.1ms/step)")
        plt.ylabel("Neuron Representation")

    def plot_perirhinal_fdn_behavior(self, title):
        """
            The following method gives a diagram to illustrate the behavior of neurons in the perirhinal
            cortex familiar discrimination
        """

        plt.figure()
        for index in range(len(self.model.perirhinal_fdn_neurons)):
            events = self.model.perirhinal_fdn_spike_recorders[index].get("events")
            ts = events["times"]
            plt.plot(ts[ts>50], np.full(shape=len(ts[ts>50]),fill_value=index,dtype=np.int64), "k.")
        if not title is None:
            plt.title(title)
        plt.ylim([0, len(self.model.perirhinal_fdn_neurons)])
        plt.xlabel("Timestep (0.1ms/step)")
        plt.ylabel("Neuron Representation")

    def debugger_model(self, test):
        """
            The following method prints the firing rate of all the neurons in the model.
        """

        self.model = Implementation_Model(test_homeostasis_gain=test)
        external_rate = input("External Neuron Input Rate (Max: 200): ")
        representation_rate = input("Representation Neuron Input Rate (Max: 200): ")
        time_step = input("Number of simulation timesteps to be run (default=10000): ")
        self.model.external_input.rate = int(external_rate)
        self.model.representation_input.rate = int(representation_rate)
        self.model.timestep = int(time_step)
        self.model.run()
        self.model.representation_input.rate = 0
        self.model.external_input.rate = 0
        self.model.run()
        record_fr_high, record_fr_low = self.model.get_homeostasis_info()
        gain_mod_high_fr, gain_mod_low_fr = self.model.get_gain_mod_info()
        ring_fr = self.model.get_ring_info()
        perirhinal_decision_fr = self.model.get_perirhinal_decision_info()
        print(F"Homeostasis High Neuron Firing Rate: {record_fr_high},Homeostasis Low Neuron Firing Rate: {record_fr_low}.")
        print(F"Gain Modulation High Neurons Firing Rates (acccording to index of array): {gain_mod_high_fr}.")
        print(F"Gain Modulation Low Neurons Firing Rates (acccording to index of array): {gain_mod_low_fr}.")
        print(F"Ring Attracotr Neurons Firing Rates (acccording to index of array): {ring_fr}.")
        print(F"Perihinal Decision Neuron Firing Rates (taken from last 1000 timesteps): {perirhinal_decision_fr}")
        self.plot_ring_behavior("Ring Behavior")
        self.plot_perirhinal_fdn_behavior("Perirhinal FDN Behavior")
        plt.show()
    
    def debugger_full_model(self):
        """
            The following method gets info for when the feedback loop is connected.
        """

        self.model = Implementation_Model(representation=True)
        external_rate = input("External Neuron Input Rate (Max: 200): ")
        time_step = input("Number of simulation timesteps to be run (default=10000): ")
        self.model.external_input.rate = int(external_rate)
        self.model.timestep = int(time_step)
        self.model.run()
        record_fr_high, record_fr_low = self.model.get_homeostasis_info()
        gain_mod_high_fr, gain_mod_low_fr = self.model.get_gain_mod_info()
        ring_fr = self.model.get_ring_info()
        perirhinal_decision_fr = self.model.get_perirhinal_decision_info()
        print(F"Homeostasis High Neuron Firing Rate: {record_fr_high},Homeostasis Low Neuron Firing Rate: {record_fr_low}.")
        print(F"Gain Modulation High Neurons Firing Rates (acccording to index of array): {gain_mod_high_fr}.")
        print(F"Gain Modulation Low Neurons Firing Rates (acccording to index of array): {gain_mod_low_fr}.")
        print(F"Ring Attracotr Neurons Firing Rates (acccording to index of array): {ring_fr}.")
        print(F"Perihinal Decision Neuron Firing Rates (taken from last 1000 timesteps): {perirhinal_decision_fr}")
        self.plot_ring_behavior("Ring Behavior")
        self.plot_perirhinal_fdn_behavior("Perirhinal FDN Behavior")
        plt.show()

    def test_perirhinal_vision_cortex(self, number = 4):
        """
            The following method prints the perirhinal vision cortex and perirhinal cortex neuron's
            info.
        """

        self.model = Implementation_Model(representation=True, external=True)
        for i in range(number):
            self.model.perirhinal_v_r_neurons[i].rate = 20
        self.model.timestep = 30000
        self.model.run()
        print(self.model.get_perirhinal_vision_decision_info())
        self.plot_ring_behavior("Ring Behavior")
        self.plot_perirhinal_fdn_behavior("Perirhinal FDN Behavior")
        plt.show()

    def test_poisson(self):
        """
            The following method tests the function for usage with two poisson generators for the
            external input and representation input.
        """
        external_rate = input("External Neuron Input Rate (Max: 200): ")
        representation_rate = input("Representation Neuron Input Rate (Max: 200): ")
        self.model.external_input.rate = int(external_rate)
        self.model.representation_input.rate = int(representation_rate)
        self.model.timestep = 10000
        self.model.run()
        self.model.external_input.rate = 0
        self.model.representation_input.rate = 0
        self.model.run()
        title = F"External Rate: {external_rate}, Representation Rate: {representation_rate}"
        self.plot_ring_behavior(title)
    
    def full_run(self):
        """
            The following method runs the analysis for the full run of the whole project. These will include
            plotting of data as required by the model.
        """
        self.model = Implementation_Model(representation=True, external=True)
        # Set up experiment layout
        count_number = input("Please type which number the model should count: ")
        vision_data = [0, 0, 0, 0, 0]
        choice_index = [0, 1, 2, 3, 4]
        for _ in range(int(count_number)):
            index_change = np.random.choice(choice_index)
            choice_index.remove(index_change)
            vision_data[index_change] = 1
        print(F"The following neurons will be activated: {str(vision_data)}")
        # Set representation neuron to activate
        for i in range(len(vision_data)):
            if vision_data[i] == 1:
                print(i)
                self.model.perirhinal_v_r_neurons[i].rate = 20
        self.model.timestep = 100000
        self.model.run()
        
        # Start Drawing
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(20,5)

        # Draw Ring Attractor Firing Rates
        for index in range(self.model.population_size):
            events = self.model.ring_spike_recorders[index].get("events")
            ts = events["times"]
            axs[0].plot(ts[ts>50], np.full(shape=len(ts[ts>50]),fill_value=index,dtype=np.int64), "k.")
        axs[0].set_title("Activity for Ring Attractor Neurons")
        axs[0].set_ylim([0, self.model.population_size])
        axs[0].set_xlabel("Timestep (0.1ms/step)")
        axs[0].set_ylabel("Ring Attractor Neuron Representation")

        # Draw perirhinal vision cortex firing rates
        for index in range(len(self.model.perirhinal_fdn_neurons)):
            events = self.model.perirhinal_fdn_spike_recorders[index].get("events")
            ts = events["times"]
            axs[1].plot(ts[ts>50], np.full(shape=len(ts[ts>50]),fill_value=index,dtype=np.int64), "k.")
        axs[1].set_title("Activity for Perirhinal Familiarity Discrimination Neurons (Feedback)")
        axs[1].set_ylim([-1, len(self.model.perirhinal_fdn_neurons)])
        axs[1].set_yticklabels(range(len(self.model.perirhinal_fdn_neurons) + 1))
        axs[1].set_xlabel("Timestep (0.1ms/step)")
        axs[1].set_ylabel("Familiarity Discrimination Neuron Representation")

        plt.show()

    def exp_1(self):
        """
            The following method uses the perirhinal cortex familiarity discrimination neurons to collect data
            for anaylsing run time.
        """
        for count_number in range(1,6):
            reaction_time_list = []
            for number_of_run in range(1,21):
                print(F"Number Testing: {count_number}, Run Count: {number_of_run}")
                self.model = Implementation_Model(representation=True, external=True)

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
                self.model.timestep = 20000
                self.model.run()

                events = self.model.perirhinal_fdn_spike_recorders[count_number - 1].get("events")
                ts = events["times"]
                ts_new = [x for x in ts if x > 0]
                reaction_time = ts_new[0]
                reaction_time_list.append(reaction_time)
            file_name = 'temp_save/data_count_' + str(count_number)
            np.save(file_name, reaction_time_list)
            
    def exp_2(self):
        """
            The following method runs the analyses for experiment 1 that is to determine the prediction
            for reaction time.
        """
        count_number = 4
        # for count_number in range(1,6):
        self.model = Implementation_Model(representation=True, external=True)
        
        # Set up Neurons and Recorders for bg
        self.experiment_neuron = nest.Create('iaf_psc_alpha', params = self.exp1_parameters)
        self.excitation_exp_neuron = nest.Create('poisson_generator', params={"rate": 100})
        self.experiment_neuron_spike = nest.Create('spike_recorder')
        self.experiment_neuron_multimeter = nest.Create('multimeter')
        self.experiment_neuron_multimeter.set(record_from=["V_m"])

        # Set up connection
        # nest.Connect(self.excitation_exp_neuron, self.experiment_neuron, syn_spec={'weight': 1 * 10**3})
        nest.Connect(self.model.high_neuron, self.experiment_neuron, syn_spec={'weight': 3 * 10**3})
        nest.Connect(self.model.low_neuron, self.experiment_neuron, syn_spec={'weight': 3 * 10**3})

        # Connect recorders
        nest.Connect(self.experiment_neuron_multimeter, self.experiment_neuron)
        nest.Connect(self.experiment_neuron, self.experiment_neuron_spike)

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
                print(i)
                self.model.perirhinal_v_r_neurons[i].rate = 20
        self.model.timestep = 20000
        self.model.run()

        # plt.figure()
        events = self.experiment_neuron_spike.get("events")
        ts = events["times"]
        # plt.plot(ts[ts>50], np.full(shape=len(ts[ts>50]),fill_value=1,dtype=np.int64), "k.")
        # plt.title("BG test")
        # plt.xlabel("Timestep (0.1ms/step)")
        # plt.ylabel("Neuron Representation")

        plt.figure()
        fr_list = []
        x_axis = []
        x = 0
        window_frame = 500
        number_of_windows = int(np.floor(self.model.total_timestep/window_frame))
        for spacing in range(number_of_windows):
            fr_list.append(len([x for x in ts if ((x >= spacing * window_frame) and (x < ((spacing + 1) * window_frame)))]) * (1000/window_frame))
            x_axis.append(x)
            x += window_frame
        plt.plot(x_axis, fr_list, 'o')

        self.plot_perirhinal_fdn_behavior("Perirhinal FDN Behavior")
        plt.show()

    def read_exp1(self):
        """
            Display results for exp1
        """
        reaction_times = []
        numbers = []
        for filename in os.listdir('temp_save/'):
            path = 'temp_save/' + filename
            reaction_times.append(np.sum(np.load(path))/len(np.load(path)))
            numbers.append(int(path[-5]))
        y_items = []
        for i in numbers:
            y_items.append(reaction_times[i - 1])
        numbers.sort()
        plt.figure()
        plt.plot(numbers, y_items,'-xg')
        plt.title("Reaction Time In Order Of Number Representation")
        plt.xlabel("Number Representation")
        plt.ylabel("Reaction Time (ms)")
        plt.show()

# The following code runs the test cases depending on the arguments.
if __name__ == "__main__":
    init_introduction = "The following code is an implementation of a model for using homeostasis and ring attractor as a form of counting. These involves different parts/methods of the code that can be run amongst which the following are the options.\n Add -t as second args in script to run a test for ring attractor in -pbgm and -d"
    table_insert = lambda x, y: "|  " + x + "  |  " + y + "  |\n"
    bold_insert = lambda x: '\033[1m' + x + '\033[0m'
    row_1 = table_insert(bold_insert("Argument "), bold_insert("Short Description                 "))
    row_2 = table_insert("-h        ", "List of args runnable             ")
    row_3 = table_insert("-phh      ", "Plot Homeostasis Heatmap          ")
    row_4 = table_insert("-pbgm     ", "Plot Bar for Gain Mod FR          ")
    row_5 = table_insert("-prw      ", "Plot Ring Attractor Weights       ")
    row_6 = table_insert("-pbr      ", "Plot Bar for Attractor FR         ")
    row_7 = table_insert("-d        ", "Debugger                          ")
    row_8 = table_insert("-tp       ", "Test with Poisson Generator       ")
    row_9 = table_insert("-pbpf     ", "Plot Bar Perirhinal Representation")
    row_10 = table_insert("-fd       ", "Debugger Full Model               ")
    row_11 = table_insert("-tpv      ", "Test Perirhinal Vision Cortex     ")
    row_12 = table_insert("-exp1     ", "Experiment 1 Analysis             ")
    args_print = init_introduction + row_1 + row_2 + row_3 + row_4 + row_5 + row_6 + row_7 + row_8 + row_9 + row_10 + row_11
    if len(sys.argv) > 1:
        run_type = str(sys.argv[1])
    else:
        run_type = None
    implement = Implementation_Run()
    if run_type == None:
        implement.full_run()
    elif run_type == "-phh":
        implement.plot_homeostasis_heatmap()
    elif run_type == "-pbgm":
        test = False
        if len(sys.argv) > 2:
            test = str(sys.argv[2])
            if test == "-t":
                test = True
        implement.plot_bar_gain_mod(test=test)
    elif run_type == "-prw":
        implement.plot_ring_weights()
    elif run_type == "-pbr":
        implement.plot_bar_ring()
    elif run_type == "-d":
        test = False
        if len(sys.argv) > 2:
            test = str(sys.argv[2])
            if test == "-t":
                test = True
        os.system("clear")
        implement.debugger_model(test=test)
    elif run_type == "-h":
        print(args_print)
    elif run_type == "-tp":
        implement.test_poisson()
        plt.show()
    elif run_type == "-pbpf":
        implement.plot_bar_perirhinal_fdn()
    elif run_type == "-fd":
        implement.debugger_full_model()
    elif run_type == "-tpv":
        implement.test_perirhinal_vision_cortex()
    elif run_type == "-exp1":
        implement.exp_1()
    elif run_type == "-rexp1":
        implement.read_exp1()