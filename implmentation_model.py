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
    
    def init_neurons_recorders(self):
        """
            The following method is a generalised method for initialising the neurons. as
            well as the spike recorders and multimeters associated with it.
        """
        self.init_homeostasis()
        self.init_ring_attractor()
        self.init_gain_mod()

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
            nest.Connect(self.high_neuron, neuron, syn_spec={'weight': 0.5 * 10**3})
        for neuron in self.move_low_neurons:
            nest.Connect(self.low_neuron, neuron, syn_spec={'weight': 0.5 * 10**3})

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
            nest.Connect(self.ring_neurons[index], self.move_high_neurons[index], syn_spec={'weight': 1.6 * 10**3})
            nest.Connect(self.ring_neurons[index], self.move_low_neurons[index], syn_spec={'weight': 1.6 * 10**3})
            # Gain Mod Neuron to Ring Connection
            nest.Connect(self.move_high_neurons[index], self.ring_neurons[next_calc(index)], syn_spec={'weight': 5})
            nest.Connect(self.move_low_neurons[index], self.ring_neurons[prev_calc(index)], syn_spec={'weight': 5})
            nest.Connect(self.move_high_neurons[index], self.ring_neurons[prev_calc(index)], syn_spec={'weight': -50000})
            nest.Connect(self.move_low_neurons[index], self.ring_neurons[next_calc(index)], syn_spec={'weight': -50000})
    
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

    def __init__(self, representation=None, external=None, test_homeostasis_gain = False):
        """
            The following parameters give a list of attribute initialisation required for
            the model to start.
        """
        # Reset Network
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        # Simulation Parameters
        self.timestep = 10000       # default is 10000
        self.test_homeostasis_gain = test_homeostasis_gain

        # Inputs To The Model
        if (representation is None) or (external is None):
            self.representation_rate = 200
            self.external_rate = 0
            self.representation_input = nest.Create('poisson_generator', params={"rate": self.representation_rate})
            self.external_input = nest.Create('poisson_generator', params={"rate": self.external_rate})
        else:
            self.representation_input = representation
            self.external_input = external

        # Ring Attractor Parameters
        self.population_size = 100
        self.max_distance = 50
        self.sd_1 = 10
        self.sd_2 = 5

        # Initialise All Region of Neurons and Recorders
        self.init_parameters()
        self.init_neurons_recorders()
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
    
    def start_ring(self):
        ring_starter = nest.Create('poisson_generator', params={"rate": 200})
        for neuron in self.ring_neurons[0:10]:
            nest.Connect(ring_starter, neuron, syn_spec={'weight': 5 * 10**3})
        nest.Simulate(50)
        ring_starter.rate = 0
    
class Implementation_Run():

    def __init__(self):
        self.model = Implementation_Model()        
    
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
    
    def plot_ring_behavior(self):
        """
            The following method gives a diagram to illustrate the behavior of neurons in ring attractor
            model.
        """
        for index in range(self.model.population_size):
            events = self.model.ring_spike_recorders[index].get("events")
            ts = events["times"]
            plt.plot(ts[ts>50], np.full(shape=len(ts[ts>50]),fill_value=index,dtype=np.int64), "k.")
        plt.ylim([0, self.model.population_size])
        plt.xlabel("Timestep (0.1ms/step)")
        plt.ylabel("Neuron Representation")
        plt.show()

    def debugger_model(self, test):
            """
                The following method prints the firing rate of all the neurons in the model.
            """
            self.model = Implementation_Model(test_homeostasis_gain=test)
            external_rate = input("External Neuron Input Rate (Max: 150): ")
            representation_rate = input("Representation Neuron Input Rate (Max: 150): ")
            time_step = input("Number of simulation timesteps to be run (default=10000): ")
            self.model.external_input.rate = int(external_rate)
            self.model.representation_input.rate = int(representation_rate)
            self.model.timestep = int(time_step)
            self.model.run()
            record_fr_high, record_fr_low = self.model.get_homeostasis_info()
            gain_mod_high_fr, gain_mod_low_fr = self.model.get_gain_mod_info()
            ring_fr = self.model.get_ring_info()
            print(F"Homeostasis High Neuron Firing Rate: {record_fr_high},Homeostasis Low Neuron Firing Rate: {record_fr_low}.")
            print(F"Gain Modulation High Neurons Firing Rates (acccording to index of array): {gain_mod_high_fr}.")
            print(F"Gain Modulation Low Neurons Firing Rates (acccording to index of array): {gain_mod_low_fr}.")
            print(F"Ring Attracotr Neurons Firing Rates (acccording to index of array): {ring_fr}.")
            self.plot_ring_behavior()
    
    def test_poisson(self):
        external_rate = input("External Neuron Input Rate (Max: 150): ")
        representation_rate = input("Representation Neuron Input Rate (Max: 150): ")
        self.model.external_input.rate = int(external_rate)
        self.model.representation_input.rate = int(representation_rate)
        self.model.timestep = 30000
        self.model.run()
        self.model.external_input.rate = 0
        self.model.representation_input.rate = 0
        self.model.run()
        self.plot_ring_behavior()
        
if __name__ == "__main__":
    init_introduction = "The following code is an implementation of a model for using homeostasis and ring attractor as a form of counting. These involves different parts/methods of the code that can be run amongst which the following are the options.\n Add -t as second args in script to run a test for ring attractor in -pbgm and -d"
    table_insert = lambda x, y: "|  " + x + "  |  " + y + "  |\n"
    bold_insert = lambda x: '\033[1m' + x + '\033[0m'
    row_1 = table_insert(bold_insert("Argument "), bold_insert("Short Description          "))
    row_2 = table_insert("-h        ", "List of args runnable      ")
    row_3 = table_insert("-phh      ", "Plot Homeostasis Heatmap   ")
    row_4 = table_insert("-pbgm     ", "Plot Bar for Gain Mod FR   ")
    row_5 = table_insert("-prw      ", "Plot Ring Attractor Weights")
    row_6 = table_insert("-pbr      ", "Plot Bar for Attractor FR  ")
    row_7 = table_insert("-d        ", "Debugger                   ")
    row_8 = table_insert("-tp       ", "Test with Poisson Generator")
    args_print = init_introduction + row_1 + row_2 + row_3 + row_4 + row_5 + row_6 + row_7 + row_8
    if len(sys.argv) > 1:
        run_type = str(sys.argv[1])
    else:
        run_type = None
    implement = Implementation_Run()
    if run_type == None:
        print("Run_Normally")
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