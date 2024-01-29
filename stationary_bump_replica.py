import nest
import numpy as np
import matplotlib.pyplot as plt

class Neuron():

    def __init__(self, params):
        self.neuron = nest.Create("iaf_psc_alpha", params=params)
        self.recorder = nest.Create("spike_recorder")
        nest.Connect(self.neuron, self.recorder)
    
    def get_events(self, time_horizon = 0):
        times = self.recorder.events['times']
        return times[times > time_horizon]        
    
class Bump():

    def __init__(self, connection_type = "stable"):
        self.connection_type = connection_type
        self.num_of_neurons = 100
        self.params = {
            "I_e": 0.9
            }
        self.neurons = self.setup_neurons(self.num_of_neurons)
        self.setup_connections()

    def setup_neurons(self, num_of_neurons):
        """
            The following function determine a list of neurons to be initialised for the attractor
            and set them into an array for reference.
        """
        neurons = []
        for _ in range(num_of_neurons):
            init_neuron = Neuron(self.params)
            neurons= np.append(neurons, init_neuron)
        return neurons
    
    def setup_connections(self):
        """
            The following function determine all the pre-post synaptic connection in the attractor.
        """
        ref_distance = self.calculate_distance()
        ref_pre = 0
        for pre_neuron in self.neurons:
            ref_post = 0
            for post_neuron in self.neurons:
                if self.connection_type == "ring" and self.connection_type == "stable":
                    weight_distance = ref_distance[ref_post]
                else:
                    weight_distance = ref_distance[ref_post] - ref_distance[ref_pre]
                weight = self.calculate_weight(weight_distance)
                syn_dict = {"weight": weight}
                nest.Connect(pre_neuron.neuron, post_neuron.neuron, syn_spec=syn_dict)
                ref_post += 1
            ref_pre += 1

    def calculate_weight(self, distance):
        """
            The following function calculates the weight of the pre-post synaptic connection based
            on the distance between the pre-synaptic and post-synaptic neuron.
        """
        if (self.connection_type == "ring") or (self.connection_type == "line"):
            A1 = 0.6
            A2 = 0.2
            sigma1 = 3
            sigma2 = 30
            # weight_calc = lambda k: A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2)) + np.heaviside(k, 0)*0.2*np.exp(-np.abs(k)/sigma2)
            weight_calc = lambda k: A1*np.exp(-k**2/(2*sigma1**2)) - A2*np.exp(-k**2/(2*sigma2**2))
            weight = weight_calc(distance)
        elif (self.connection_type == "stable"):
            w = lambda a,z:((a * np.pi)**(-1/2))*np.exp(-(z**2)/a)
            weight_calc = lambda z: 5 * (1.1*w(1/28,z) - w(1/20,z))
            weight = self.num_of_neurons * weight_calc(np.abs(50 - distance)/self.num_of_neurons)
        return weight
    
    def calculate_distance(self):
        """
            The following function calculate the reference distance with respect to
            whether its for ring attractor or a line.
        """
        if (self.connection_type == "line") or (self.connection_type == "stable"):
            distance = np.linspace(0, self.num_of_neurons - 1,self.num_of_neurons)
        # for line and stable connection
        else:
            distance = np.abs(np.linspace(-self.num_of_neurons + 2, self.num_of_neurons - 2, self.num_of_neurons))
            distance = np.concatenate((distance[int(np.ceil((self.num_of_neurons - 1)/2)):], distance[:int(np.ceil((self.num_of_neurons - 1)/2))]))
        return distance
    
    def set_noise(self, spike_range=[], level=10000.0):
        """
            The following function is supposed to set a Poisson Generator input to selected neurons
        """
        noise = nest.Create("poisson_generator")
        noise.rate = level
        if len(spike_range) == 0:
            neuron_range = self.neurons
        else:
            neuron_range = self.neurons[spike_range[0]:spike_range[1]]
        for neuron in neuron_range:
            nest.Connect(noise, neuron.neuron, syn_spec={"weight": [[1.2]], "delay": 1.0})
        return noise
    
    def plot_weight(self, bump_neuron = 5):
        """
            The following is a test function for plotting the weight as a sample. The bump will be set to focus on a specific point.
            Thus, the results are the weights for a specific pre-synaptic connection.
        """

        ref_distance = self.calculate_distance()
        ref_pre = bump_neuron
        ref_post = 0
        weight_list = []

        # Calculating the weights for each connection
        for _ in self.neurons:
            if self.connection_type == "ring" or self.connection_type == "stable":
                weight_distance = ref_distance[ref_post]
            else:
                weight_distance = ref_distance[ref_post] - ref_distance[ref_pre]
            weight = self.calculate_weight(weight_distance)
            weight_list = np.append(weight_list, weight)
            ref_post += 1
        
        plt.plot(np.arange(self.num_of_neurons),weight_list)
        plt.title("Mexican-hat shaped function on interneuron distance")
        plt.ylabel("Weight")
        plt.xlabel("Pre-Post synaptic distance")
        plt.show()
    
    def plot_spikes(self):
        """
            The following function plots the spikes of all neurons with respect to events
        """
        plt.figure(2)
        counter = 0
        for neuron in self.neurons:
            plt.plot(neuron.get_events(), np.full(shape=len(neuron.get_events()),fill_value=counter,dtype=np.int64), "k.")
            counter += 1
        plt.ylim([0, self.num_of_neurons + 20])
        plt.xlabel("Timestep (0.1ms/step)")
        plt.ylabel("Neuron Representation")
        plt.show()
    
bump = Bump()
# bump.plot_weight()
beta = 1.5
alpha = lambda t: beta * np.exp(-beta*t)
pulse_time = 10
noise = bump.set_noise(spike_range=[30,40],level=alpha(pulse_time))
pulse = bump.set_noise(level=20)
print(alpha(pulse_time))
nest.Simulate(pulse_time)
bump.plot_spikes()