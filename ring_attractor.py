import nest
import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 80

class Neuron:
    def __init__(self, distance, representation):
        params = {
            "I_e": 120.0, "tau_m": 20.0
        }
        self.data = nest.Create("iaf_psc_alpha", params=params)
        self.representation = representation
        self.distance = distance
        self.spike_recorder = nest.Create("spike_recorder")
        nest.Connect(self.data, self.spike_recorder)
        self.next = None
        self.prev = None

class NeuronList:
    def __init__(self):
        self.distance_arr = self.set_distance()
        self.neuron_list = np.array([])
        self.header = None
        self.create_neuron_list()

    def set_distance(self):
        """
            Set the order of neurons onto which the distance will be sorted
        """
        dist_arr = None
        for _ in range(2):
            dist_row = np.array([])
            distance = 0
            side_size = int(np.floor((POPULATION_SIZE-1)/2))
            for i in range(side_size):
                distance += 1
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
        return np.flip(dist_return)
    
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
                self.header.next = Neuron(self.distance_arr[i], num_rep)
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

    def get_weight(self, n1, n2):
        """
            calculate the weight where n1 is pre-synaptic and n2 is post-synaptic
        """
        A1 = 0.6
        A2 = 0.2
        sigma1 = 3
        sigma2 = 30
        weight_calc = lambda k: A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2))
        distance = np.abs(n1.distance - n2.distance)
        weight = weight_calc(distance)
        return weight

    def show_weight(self, right = True, steps = 0):
        self.move_distances(right = right, steps = steps)
        while not (self.header.distance == 0):
            self.header = self.header.next
        current_neuron = self.header
        data_dict = {}
        self.header = self.header.next
        for _ in self.neuron_list:
            data_dict[self.header.representation] = listNeuron.get_weight(current_neuron, self.header)
            self.header = self.header.next
        lists = sorted(data_dict.items())
        x, y = zip(*lists)
        plt.plot(np.divide(x, 10),y)
        plt.title("Mexican-hat shaped function on interneuron distance")
        plt.ylabel("Weight")
        plt.xlabel("|x - y|")
        plt.show()
        self.header = current_neuron.representation

    def connect_neurons(self):
        for _ in self.neuron_list:
            current_neuron = self.header
            for _ in self.neuron_list:
                syn_dict = {"weight": self.get_weight(current_neuron, self.header)}
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
        plt.show()
    
    def add_noise(self, representation):
        noise = nest.Create("poisson_generator")
        noise.rate = 10000.0
        for i in range(representation[0], representation[1] + 1):
            while not self.header.representation == i:
                self.header = self.header.next
            nest.Connect(noise, self.header.data, syn_spec={"weight": [[1.2]], "delay": 1.0})
                
listNeuron = NeuronList()
listNeuron.connect_neurons()
# listNeuron.show_weight(steps= 40)
listNeuron.add_noise((20,30))
nest.Simulate(1000)
listNeuron.plot_spikes()