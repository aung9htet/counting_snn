import os
import sys
import nest
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import pandas as pd
from multiprocessing import Process

POPULATION_SIZE = 100
PARAM_RANGE = [1,100] # tau_syn_ex search range

"""
    The following class will define the initialisation for the neuron and the information it will represent (as in readout values)
"""
class Neuron:
    def __init__(self, distance = None, representation = None, parameters = [], pulse_time = 0):
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
        self.pulse_time = pulse_time
    
    def get_metric(self):
        TIME_HORIZON = 700 + self.pulse_time
        times = self.spike_recorder.events['times']
        return len(times[times > TIME_HORIZON])
    

"""
    The following class will define how the neurons will be connected and how it will define the representation for each neuron
"""
class NeuronList():
    def __init__(self, parameters = [], pulse_time = 0):
        self.distance_arr = self.set_distance()
        self.neuron_list = np.array([])
        self.header = None
        self.parameters = parameters
        self.pulse_time = pulse_time
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
                self.header = Neuron(self.distance_arr[i], num_rep, pulse_time = self.pulse_time)
                self.neuron_list = np.append(self.neuron_list, self.header)
            else:
                self.header.next = Neuron(self.distance_arr[i], num_rep, parameters=self.parameters, pulse_time=self.pulse_time)
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

    def add_external_neurons(self, parameters = None, weight = []):
        """
            The neurons will have their own parameters different from the ring attractor's neurons
            This is used to move the ring attractor
            Weight = [left weight, right weight]
        """
        noise_left = nest.Create("poisson_generator")
        noise_right = nest.Create("poisson_generator")
        noise_left.rate = 1000
        noise_right.rate = 1000
        # for i in range(representation[0], representation[1] + 1):
        #     while not self.header.representation == i:
        #         self.header = self.header.next
        #     nest.Connect(noise, self.header.data, syn_spec={"weight": [[4.2]], "delay": 1.0})
        self.right = np.array([])
        self.left = np.array([])
        for neuron in self.neuron_list:
            if parameters == None:
                self.neuron_right = Neuron()
                self.neuron_left = Neuron()
            else:
                self.neuron_right = Neuron(parameters=parameters)
                self.neuron_left = Neuron(parameters=parameters)
            curr_neuron = neuron
            prev_neuron = neuron.next
            syn_dict_right_in = {"weight": weight[0]}
            syn_dict_right_out = {"weight": weight[1]}
            syn_dict_left_in = {"weight": weight[2]}
            syn_dict_left_out = {"weight": weight[3]}
            nest.Connect(self.neuron_right.data, curr_neuron.data, syn_spec=syn_dict_right_in)
            nest.Connect(prev_neuron.data, self.neuron_right.data, syn_spec=syn_dict_right_out)
            nest.Connect(self.neuron_left.data, prev_neuron.data, syn_spec=syn_dict_left_in)
            nest.Connect(curr_neuron.data, self.neuron_left.data, syn_spec=syn_dict_left_out)
            self.right = np.append(self.right, self.neuron_right)
            self.left = np.append(self.left, self.neuron_left)
        return self.right, self.left

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
        # weight_calc = lambda k: 10 * (A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2)))
        
        w = lambda a, z: (1.0/np.sqrt(a*np.pi))*np.exp(-z**2/a)
        J = lambda z: 5 * (1.1 * w(1/28, z) - w(1/20,z))
        weight_calc = lambda k: J(np.abs(k)/ POPULATION_SIZE)
        
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
        within_range = 0 
        out_of_range = 0
        for idx, neuron in enumerate(self.neuron_list):
            if (idx >= 20) and (idx <= 30):
                within_range += neuron.get_metric()
            else:
                out_of_range += neuron.get_metric()        
        # numerator/= 10
        # out_of_range/=(POPULATION_SIZE-10)
        if out_of_range < 1:
            denominator = 1
        else:
            denominator = out_of_range
        result = (within_range - out_of_range)/ denominator
        return result

def parameter_search(rank, process):
    param_tau_syn_ex_range = [70, 100]
    full_filename = "temp_results/results" + str(rank) + ".csv"
    range_division = (param_tau_syn_ex_range[1] - param_tau_syn_ex_range[0])/process
    start_tau_syn_ex = param_tau_syn_ex_range[0] + (rank * range_division)
    end_tau_syn_ex =  param_tau_syn_ex_range[0] + ((rank + 1) * range_division)

    with open(full_filename, 'w', newline='') as csvfile:

        fieldnames = ['Pulse_time', 'I_e', 'Tau_Syn_Ex', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({"Pulse_time": "Pulse_time", "I_e": "I_e", "Tau_Syn_Ex": "Tau_Syn_Ex", "result": "Result"})
        
        # params to test
        params_pg1 = 0 # in between 1-200Hz
        params_pg2 = 200 # in between 1-200Hz
        params_e_l = 0.0 # Resting Membrane Potential (-70mV)
        params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
        params_tau_m = 1.0 # Membrane time constant (1-100 ms)
        params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
        params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
        params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
        params_pulse_time = np.arange(600, 700, 5)
        params_i_e = [6, 10] # Constant input current (0-10 pA)
        params_tau_syn_ex = np.arange(start_tau_syn_ex, end_tau_syn_ex, 0.1)
        for pulse_time in params_pulse_time:
            for i_e in range(params_i_e[0], params_i_e[1]):    
                for tau_syn_ex in params_tau_syn_ex:
                    # nest.Prepare()
                    parameter_search = [params_e_l, params_c_m, params_tau_m, params_t_ref, params_v_th, params_v_reset, i_e, tau_syn_ex]
                    listNeuron = NeuronList(parameter_search, pulse_time)
                    listNeuron.connect_neurons()
                    listNeuron.add_noise((1, 99), params_pg1)
                    noise = listNeuron.add_noise((20,30), params_pg2)
                    nest.Simulate(pulse_time)
                    noise.rate = 0
                    nest.Simulate(1000)
                    result = listNeuron.get_metric()
                    print("Current Progress: Pulse_time = %s, I_e = %s, Tau_Syn_Ex = %s, Result = %s" % (pulse_time, i_e, tau_syn_ex, result))
                    writer.writerow({"Pulse_time": pulse_time, "I_e": i_e, "Tau_Syn_Ex": tau_syn_ex, "result": result})
                    nest.ResetKernel()
                    

def combine_results():
    cwd = os.getcwd() + "/temp_results"
    files_to_add = [os.path.join(cwd, f) for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]
    full_filename_to_save = "results/results" + str(datetime.now()) + ".csv"
    with open(full_filename_to_save, 'w', newline='') as csvfilewriter:
        fieldnames = ['Pulse_time', 'I_e', 'Tau_Syn_Ex', 'result']
        writer = csv.DictWriter(csvfilewriter, fieldnames=fieldnames)
        writer.writerow({"Pulse_time": "Pulse_time", "I_e": "I_e", "Tau_Syn_Ex": "Tau_Syn_Ex", "result": "Result"})
        for file in files_to_add:
            with open(file) as csvfile:
                results_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i = 0
                for row in results_reader:
                    if i > 0:
                        writer.writerow({"Pulse_time": row[0], "I_e": row[1], "Tau_Syn_Ex": row[2], "result": row[3]})
                    i += 1
    dataFrame = pd.read_csv(full_filename_to_save, delimiter=",")
    dataFrame.sort_values(["Pulse_time", "I_e","Tau_Syn_Ex"], axis=0, ascending=True,inplace=True, na_position='first')
    dataFrame.to_csv(full_filename_to_save, index = False)


def single_test():
    # params to test
    params_pg1 = 0 # in between 1-200Hz
    params_pg2 = 200 # in between 1-200Hz
    params_e_l = 0.0 # Resting Membrane Potential (-70mV)
    params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
    params_tau_m = 1.0 # Membrane time constant (1-100 ms)
    params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
    params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
    params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
    params_i_e = 9 # Constant input current (0-10 pA)
    params_tau_syn_ex = 80.8
    parameter_search = [params_e_l, params_c_m, params_tau_m, params_t_ref, params_v_th, params_v_reset, params_i_e, params_tau_syn_ex]
    pulse_time = 595
    listNeuron = NeuronList(parameter_search, pulse_time=pulse_time)
    listNeuron.connect_neurons()
    listNeuron.add_noise((1, 99), params_pg1)
    noise = listNeuron.add_noise((80,90), params_pg2)
    nest.Simulate(pulse_time)
    noise.rate = 0
    nest.Simulate(1000)
    print(listNeuron.get_metric())
    listNeuron.plot_spikes()
    return listNeuron.get_metric()

def move_bump_gain_mod():
    # params for external neurons
    move_params_e_l = 0.0 # Resting Membrane Potential (-70mV)
    move_params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
    move_params_tau_m = 1.0 # Membrane time constant (1-100 ms)
    move_params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
    move_params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
    move_params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
    move_params_i_e = 8.134 # Constant input current (0-10 pA)
    move_params_tau_syn_ex = 86
    move_weight = [5, 5, 0, 0]
    move_parameter_search = [move_params_e_l, move_params_c_m, move_params_tau_m, move_params_t_ref, move_params_v_th, move_params_v_reset, move_params_i_e, move_params_tau_syn_ex]
    
    # params for internal neurons
    params_pg1 = 0 # in between 1-200Hz
    params_pg2 = 200 # in between 1-200Hz
    params_e_l = 0.0 # Resting Membrane Potential (-70mV)
    params_c_m = 0.0 # Capacity of the membrane (0.5-1 microFaraday)
    params_tau_m = 1.0 # Membrane time constant (1-100 ms)
    params_t_ref = 0.0 # Duration of refractory period (1-5 ms)
    params_v_th = 1.0 # Spike threshold (0.22-0.122 mV)
    params_v_reset = 0.0 # Reset potential of the membrane (-80 - -70 mV)
    params_i_e = 8.134 # Constant input current (0-10 pA)
    params_tau_syn_ex = 86
    parameter_search = [params_e_l, params_c_m, params_tau_m, params_t_ref, params_v_th, params_v_reset, params_i_e, params_tau_syn_ex]
    pulse_time = 556

    # running simulation
    listNeuron = NeuronList(parameter_search, pulse_time=pulse_time)
    listNeuron.connect_neurons()
    listNeuron.add_noise((1, 99), params_pg1)
    noise = listNeuron.add_noise((80,90), params_pg2)
    right_external, left_external = listNeuron.add_external_neurons(move_parameter_search, move_weight)
    nest.Simulate(pulse_time)
    noise.rate = 0
    nest.Simulate(1000)
    print(listNeuron.get_metric())
    nest.SyncProcesses()
    for neuron in right_external:
        print(neuron.get_metric())
    listNeuron.plot_spikes()

def plot_weights():
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

if __name__ == "__main__":
    option = sys.argv[1]
    # options decide whether to do a single run, combine results, a parameter search or to plot weights
    # 1 for single test
    if option == "1":
        single_test()
    # 2 for plotting weights
    elif option == "2":
        plot_weights()
    # 3 for combine results
    elif option == "3":
        combine_results()
    # 4 for multiprocessing parameter search
    elif option == "4":
        cores = 8
        process_list = []
        for i in range(cores):
            process = Process(target=parameter_search, args=(i, cores))
            process_list.append(process)    
            process.start()
        for process in process_list:
            process.join()
    elif option == "5":
        gain_mod_weight = [0,0]
        move_bump_gain_mod()