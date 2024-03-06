import os
import nest
import numpy as np
import matplotlib.pyplot as plt
import csv

POPULATION_SIZE = 100

"""
    The following class will define the initialisation for the neuron and the information it will represent (as in readout values)
"""
class Neuron:
    def __init__(self, distance, representation, parameters = []):
        # default parameters for neuron
        if parameters == []:
            params = {
                "I_e": 120.0, "tau_m": 20.0
            }
        else:           
            params = {
                "E_L": parameters[0],
                # "C_m": parameters[1],
                "tau_m": parameters[2],
                "t_ref": parameters[3],
                "V_th": parameters[4],
                "V_reset": parameters[5],
                "I_e": parameters[6],
                "tau_syn_ex": parameters[7]
                # "tau_syn_in": parameters[8]
            }
        # initialise neuron (Integrat and fire model)
        self.data = nest.Create("iaf_psc_alpha", params=params)
        # what the neuron represents
        self.representation = representation
        # distance away from the focused neuron
        self.distance = distance
        # initialise spike recorder and connect to neuron
        self.spike_recorder = nest.Create("spike_recorder")
        nest.Connect(self.data, self.spike_recorder)
        self.next = None
        self.prev = None
    
    def get_metric(self):
        TIME_HORIZON = 300
        times = self.spike_recorder.events['times']
        return len(times[times > TIME_HORIZON])

"""
    The following class will define how the neurons will be connected and how it will define the representation for each neuron
"""
class NeuronList:
    def __init__(self, parameters = []):
        self.distance_arr = self.set_distance()
        print(self.distance_arr)
        self.neuron_list = np.array([])
        self.header = None
        self.parameters = parameters
        self.create_neuron_list()

    def set_distance(self):
        """
            Set the order of neurons onto which the distance will be sorted
        """
        dist_arr = None
        for j in range(2):
            dist_row = np.array([])
            distance = 0
            side_size = int(np.floor((POPULATION_SIZE-1)/2))
            for i in range(side_size):
                if j % 2 == 0:
                    distance += 1
                else:
                    distance -= 1
                dist_row = np.append(dist_row, (distance))
            if dist_arr is None:
                dist_arr = dist_row
            else:
                dist_arr = np.vstack([dist_arr, dist_row])
        dist_return = np.array([0])
        dist_return = np.append(dist_return, dist_arr[0])
        if ((POPULATION_SIZE % 2) == 0):
            dist_return = np.append(dist_return, distance + 1)
        dist_return = np.append(dist_return, np.flip(dist_arr[1]))
        return dist_return
    
    def create_neuron_list(self):
        """
            Make an linked list of the neurons
        """
        num_rep = 1
        for i in range(len(self.distance_arr)):
            if i == 0:
                self.header = Neuron(self.distance_arr[i], num_rep)
                self.neuron_list = np.append(self.neuron_list, self.header)
            else:
                self.header.next = Neuron(self.distance_arr[i], num_rep, parameters=self.parameters)
                self.header = self.header.next   
                self.neuron_list = np.append(self.neuron_list, self.header)
                self.header.prev = self.neuron_list[i-1]
            num_rep += 1
        self.header.next = self.neuron_list[0]
        self.header.next.prev = self.header
        
    def move_distances(self, right = True, steps = 1):
        """
            Move the distances of the neuron along the number line 
            representation depending on the steps
        """
        current_neuron = self.header
        while self.header.distance != 0:
            self.header = self.header.next
        self.header = self.header.next
        for _ in range(steps):
            if right == True:
                self.header = self.header.next
            else:
                self.header = self.header.prev
        for distance in self.distance_arr:
            self.header.distance = distance
            self.header = self.header.next
        self.header = current_neuron

    def get_weight(self, distance):
        """
            calculate the weight where n1 is pre-synaptic and n2 is post-synaptic
        """
        A1 = 0.6
        A2 = 0.2
        sigma1 = 3
        sigma2 = 30
        # weight_calc = lambda k: A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2)) + np.heaviside(k, 0)*-0.5*np.exp(-np.abs(k)/sigma1)
        # weight_calc = lambda k: A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2)) + np.heaviside(k, 0)*0.4*np.exp(-np.abs(k)/sigma2)
        weight_calc = lambda k: 10 * (A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2)))
        
        # w = lambda a, z: (1.0/np.sqrt(a*np.pi))*np.exp(-z**2/a)
        # J = lambda z: 5 * (1.1 * w(1/28, z) - w(1/20,z))
        # weight_calc = lambda k: J(np.abs(k)/ POPULATION_SIZE)
        
        # weight_calc = lambda k: 4 * (sigma2*np.exp(-k**2/(2*sigma1**2)) - sigma1*np.exp(-k**2/(2*sigma2**2)))/((sigma2 - sigma1))
        # while it != n2:
        #     it = it.next
        #     distance = distance + 1

        # if distance > POPULATION_SIZE/2:
        #     distance = POPULATION_SIZE - distance
        # print("distance = %s, n1 distance = %s" % (distance, n2.distance))
        weight = weight_calc(distance)
        return weight

    def show_weight(self, right = True, steps = 0):
        self.move_distances(right = right, steps = steps)
        while not (self.header.distance == 0):
            self.header = self.header.next
        current_neuron = self.header
        data_dict = {}
        self.header = self.header.next
        for i in range(len(self.neuron_list)):
            # print(self.distance_arr[i])
            data_dict[self.header.representation] = self.get_weight(self.distance_arr[i])
            self.header = self.header.next
        lists = sorted(data_dict.items())
        x, y = zip(*lists)
        plt.plot(x,y)
        plt.title("Mexican-hat shaped function on interneuron distance")
        plt.ylabel("Weight")
        plt.xlabel("Neuron Index")
        plt.show()
        self.header = current_neuron.representation

    def connect_neurons(self):
        for _ in self.neuron_list:
            current_neuron = self.header
            for i in range(len(self.neuron_list)):
                syn_dict = {"weight": self.get_weight(self.distance_arr[i])}
                nest.Connect(current_neuron.data, self.header.data, syn_spec=syn_dict)
                self.header = self.header.next
            self.header = current_neuron.next

    def plot_spikes(self):
        plt.figure(2)
        for neuron in self.neuron_list:
            events = neuron.spike_recorder.get("events")
            ts = events["times"]
            plt.plot(ts, np.full(shape=len(ts),fill_value=neuron.representation,dtype=np.int64), "k.")
        plt.ylim([0, POPULATION_SIZE + 20])
        plt.xlabel("Timestep (0.1ms/step)")
        plt.ylabel("Neuron Representation")
        plt.show()
    
    def add_noise(self, representation, level = 10000.0):
        noise = nest.Create("poisson_generator")
        noise.rate = level
        for i in range(representation[0], representation[1] + 1):
            while not self.header.representation == i:
                self.header = self.header.next
            nest.Connect(noise, self.header.data, syn_spec={"weight": [[4.2]], "delay": 1.0})
        return noise

    def get_metric(self):
        numerator = 0 
        denominator = 0
        for idx, neuron in enumerate(self.neuron_list):
            if (idx >= 20) and (idx <= 30):
                numerator += neuron.get_metric()
            else:
                denominator += neuron.get_metric()        
        # numerator/= 10
        # denominator/=(POPULATION_SIZE-10)
        if denominator < 1:
            denominator = 1
        return numerator/denominator

def parameter_search():
    print(os.listdir(os.getcwd() + "/results"))
    index = 1
    filename = "results/result"
    full_filename = filename + str(index) + ".csv"
    while full_filename in os.listdir(os.getcwd() + "/results"):
        index += 1
        full_filename = filename + str(index) + ".csv"
        print(full_filename)
    with open(full_filename, 'w', newline='') as csvfile:

        fieldnames = ['I_e', 'Tau_Syn_Ex', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({"I_e": "I_e", "Tau_Syn_Ex": "Tau_Syn_Ex", "result": "Result"})
        
        # params to test
        params_pg1 = 100 # in between 1-200Hz
        params_pg2 = 30000 # in between 1-200Hz
        params_e_l = 0.0 # Resting Membrane Potential (-70mV)
        params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
        params_tau_m = 1.0 # Membrane time constant (1-100 ms)
        params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
        params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
        params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
        params_i_e = [0, 10] # Constant input current (0-10 pA)
        params_tau_syn_ex = np.arange(1,10,0.1)
        for i_e in range(params_i_e[0], params_i_e[1]):
            for tau_syn_ex in params_tau_syn_ex:
                # nest.Prepare()
                parameter_search = [params_e_l, params_c_m, params_tau_m, params_t_ref, params_v_th, params_v_reset, i_e, tau_syn_ex]
                print("Current Progress: params = %s" % (parameter_search))
                listNeuron = NeuronList(parameter_search)
                listNeuron.connect_neurons()
                listNeuron.add_noise((1, 99), params_pg1)
                noise = listNeuron.add_noise((20,30), params_pg2)
                nest.Simulate(10)
                noise.rate = 0
                nest.Simulate(1000)
                writer.writerow({"I_e": i_e, "Tau_Syn_Ex": tau_syn_ex, "result": listNeuron.get_metric()})
                nest.ResetKernel()

def single_test():
    
    # params to test
    params_pg1 = 100 # in between 1-200Hz
    params_pg2 = 5000 # in between 1-200Hz
    params_e_l = 0.0 # Resting Membrane Potential (-70mV)
    params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
    params_tau_m = 1.0 # Membrane time constant (1-100 ms)
    params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
    params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
    params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
    params_i_e = 8 # Constant input current (0-10 pA)
    params_tau_syn_ex = 15
    parameter_search = [params_e_l, params_c_m, params_tau_m, params_t_ref, params_v_th, params_v_reset, params_i_e, params_tau_syn_ex]

    listNeuron = NeuronList(parameter_search)
    listNeuron.connect_neurons()
    listNeuron.add_noise((1, 99), params_pg1)
    noise = listNeuron.add_noise((80,90), params_pg2)
    nest.Simulate(10)
    noise.rate = 0
    nest.Simulate(1000)
    listNeuron.plot_spikes()

def plot_weights():
    params_pg1 = 100 # in between 1-200Hz
    params_pg2 = 30000 # in between 1-200Hz
    params_e_l = 0.0 # Resting Membrane Potential (-70mV)
    params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
    params_tau_m = 1.0 # Membrane time constant (1-100 ms)
    params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
    params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
    params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
    params_i_e = 3 # Constant input current (0-10 pA)
    params_tau_syn_ex = 6.5
    parameter_search = [params_e_l, params_c_m, params_tau_m, params_t_ref, params_v_th, params_v_reset, params_i_e, params_tau_syn_ex]
    listNeuron = NeuronList(parameter_search)
    listNeuron.move_distances(steps=50)
    listNeuron.show_weight()

# parameter_search()
single_test()
# plot_weights()