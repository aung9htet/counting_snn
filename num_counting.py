import nest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

BATCH_ITERATION = 10000
class Num_Counting_Model:

    def __init__(self, sp_rate = 85.0, insen_rate = 85.0, batch = True):
        nest.ResetKernel()
        nest.use_wfr = False

        # weights
        self.SP_SEN_weight = 1.0
        self.SEN_HIGHER_weight = 1.0
        self.SEN_LOWER_weight = -1.0
        self.INSEN_HIGHER_weight = -0.5
        self.INSEN_LOWER_weight = 0.88

        if batch == True:
            self.run_time = BATCH_ITERATION
        else:
            self.run_time = 1

        self.excite = 1.5
        self.inhibit = 0.75

        self.sp_v_m = -65.0
        self.sen_v_m = -65.0
        self.insen_v_m = -65.0
        self.higher_v_m = -65.0
        self.lower_v_m = -65.0
        
        # Sp neuron
        sp_params = {
            "V_m": self.sp_v_m,
            "E_L": -50,
            "C_m": 800000,
            "tau_m": 10,
            "t_ref": 2,
            "V_th": -50,
            "V_reset": -65,
            "tau_syn_ex": self.excite,
            "tau_syn_in": self.inhibit
        }
        self.sp = nest.Create('iaf_psc_alpha', params=sp_params)

        # Sp neuron noise input
        self.sp_noise = nest.Create('poisson_generator')
        self.sp_noise.rate = sp_rate

        # Sensitivity neuron
        sen_params = {
            "V_m": self.sen_v_m,
            "E_L": -50,
            "C_m": 800000,
            "tau_m": 10,
            "t_ref": 2,
            "V_th": -50,
            "V_reset": -65,
            "tau_syn_ex": self.excite,
            "tau_syn_in": self.inhibit
        }
        self.sen = nest.Create('iaf_psc_alpha', params=sen_params)

        # Insensitivity neurons
        insen_params = {
            "V_m": self.insen_v_m,
            "E_L": -50,
            "C_m": 800000,
            "tau_m": 10,
            "t_ref": 2,
            "V_th": -50,
            "V_reset": -65,
            "tau_syn_ex": self.excite,
            "tau_syn_in": self.inhibit
        }
        self.insen = nest.Create('iaf_psc_alpha', params=insen_params)

        # Insensitivity neuron noise input
        self.insen_noise = nest.Create('poisson_generator')
        self.insen_noise.rate = insen_rate

        # Higher/Warm neurons
        higher_params = {
            "V_m": self.higher_v_m,
            "E_L": -50,
            "C_m": 800000,
            "tau_m": 10,
            "t_ref": 2,
            "V_th": -50,
            "V_reset": -65,
            "tau_syn_ex": self.excite,
            "tau_syn_in": self.inhibit
        }
        self.higher = nest.Create('iaf_psc_alpha', params=higher_params)
        
        # Lower/Cold neurons
        lower_params = {
            "V_m": self.lower_v_m,
            "E_L": -50,
            "C_m": 800000,
            "tau_m": 10,
            "t_ref": 2,
            "V_th": -50,
            "V_reset": -65,
            "tau_syn_ex": self.excite,
            "tau_syn_in": self.inhibit
        }
        self.lower = nest.Create('iaf_psc_alpha', params=lower_params)

        # the recorders
        self.sp_srec = nest.Create('spike_recorder')
        self.sen_srec = nest.Create('spike_recorder')
        self.insen_srec = nest.Create('spike_recorder')
        self.higher_srec = nest.Create('spike_recorder')
        self.lower_srec = nest.Create('spike_recorder')

        self.sp_voltmeter = nest.Create("voltmeter")
        self.sen_voltmeter = nest.Create("voltmeter")
        self.insen_voltmeter = nest.Create("voltmeter")
        self.higher_voltmeter = nest.Create("voltmeter")
        self.lower_voltmeter = nest.Create("voltmeter")

    def setup_network(self):
        # noise connection
        nest.Connect(self.sp_noise, self.sp, syn_spec={'weight': 1.0, 'delay': 1.0})
        nest.Connect(self.insen_noise, self.insen, syn_spec={'weight': 1.0, 'delay': 1.0})

        # neuron connection
        nest.Connect(self.sp, self.sen, syn_spec={'weight': self.SP_SEN_weight, 'delay': 1.0})
        nest.Connect(self.sen, self.higher, syn_spec={'weight': self.SEN_HIGHER_weight, 'delay': 1.0})
        nest.Connect(self.sen, self.lower, syn_spec={'weight': self.SEN_LOWER_weight, 'delay': 1.0})
        nest.Connect(self.insen, self.higher, syn_spec={'weight': self.INSEN_HIGHER_weight, 'delay': 1.0})
        nest.Connect(self.insen, self.lower, syn_spec={'weight': self.INSEN_LOWER_weight, 'delay': 1.0})

        # voltmeter connection
        nest.Connect(self.sp_voltmeter, self.sp)
        nest.Connect(self.sen_voltmeter, self.sen)
        nest.Connect(self.insen_voltmeter, self.insen)
        nest.Connect(self.higher_voltmeter, self.higher)
        nest.Connect(self.lower_voltmeter, self.lower)

        # spike recorder connection
        nest.Connect(self.sp, self.sp_srec)
        nest.Connect(self.sen, self.sen_srec)
        nest.Connect(self.insen, self.insen_srec)
        nest.Connect(self.higher, self.higher_srec)
        nest.Connect(self.lower, self.lower_srec)
        
        self.run_network()

    def run_network(self):
        nest.Simulate(self.run_time)

    def count_spikes(self, neuron_low = True):
        if neuron_low == True:
            events = self.lower_srec.get("events")
            senders = events["senders"]
        else:
            events = self.higher_srec.get("events")
            senders = events["senders"]
        return len(senders)

    def plot_individual(self):

        plt.figure(figsize=(28,16))

        # sp voltmeter
        plt.subplot(5,2,1)
        nest.voltage_trace.from_device(self.sp_voltmeter)
        plt.xticks([])
        plt.yticks([])
        plt.title("Sp neuron voltage")

        # sp spikes
        plt.subplot(5,2,2)
        events = self.sp_srec.get("events")
        senders = events["senders"]
        ts = events["times"]
        plt.title("Sp neuron spike timing")
        plt.plot(ts, senders, ".")

        # sensitivity voltmeter
        plt.subplot(5,2,3)
        nest.voltage_trace.from_device(self.sen_voltmeter)
        plt.xticks([])
        plt.yticks([])
        plt.title("Sensitivity neuron voltage")

        # sensitivity spikes
        plt.subplot(5,2,4)
        events = self.sen_srec.get("events")
        senders = events["senders"]
        ts = events["times"]
        plt.title("Sensitivity neuron spike timing")
        plt.plot(ts, senders, ".")

        # insensitivity voltmeter
        plt.subplot(5,2,5)
        nest.voltage_trace.from_device(self.insen_voltmeter)
        plt.xticks([])
        plt.yticks([])
        plt.title("Insensitivity neuron voltage")

        # insensitivity spikes
        plt.subplot(5,2,6)
        events = self.insen_srec.get("events")
        senders = events["senders"]
        ts = events["times"]
        plt.title("Insensitivity neuron spike timing")
        plt.plot(ts, senders, ".")

        # higher voltmeter
        plt.subplot(5,2,7)
        nest.voltage_trace.from_device(self.higher_voltmeter)
        plt.xticks([])
        plt.yticks([])
        plt.title("Higher/Warm neuron voltage")

        # higher spikes
        plt.subplot(5,2,8)
        events = self.higher_srec.get("events")
        senders = events["senders"]
        ts = events["times"]
        plt.title("Higher/Warm neuron spike timing")
        plt.plot(ts, senders, ".")

        # lower voltmeter
        plt.subplot(5,2,9)
        nest.voltage_trace.from_device(self.lower_voltmeter)
        plt.xticks([])
        plt.yticks([])
        plt.title("Lower/Cold neuron voltage")

        # lower spikes
        plt.subplot(5,2,10)
        events = self.lower_srec.get("events")
        senders = events["senders"]
        ts = events["times"]
        plt.title("Lower/Cold neuron spike timing")
        plt.plot(ts, senders, ".")
        plt.show()
        print("high - low = ", self.count_spikes(False) - self.count_spikes(True))

    def plot_seminar(self):

        plt.figure(figsize=(28,16))
        
        plt.suptitle("Spike Timing Results When External Input is similar to Internal Input", fontsize=16)
        # sensitivity spikes
        plt.subplot(2,2,1)
        events = self.sen_srec.get("events")
        ts = events["times"]
        plt.title("Sensitivity External Spike Timing")
        plt.yticks(range(1,2))
        plt.plot(ts, np.full(len(events["times"]), 1), ".", color='mediumblue', linewidth=6)

        # insensitivity spikes
        plt.subplot(2,2,2)
        events = self.insen_srec.get("events")
        ts = events["times"]
        plt.title("Sensitivity Internal Spike Timing")
        plt.yticks(range(1,2))
        plt.plot(ts, np.full(len(events["times"]), 1), ".", color='red', linewidth=6)

        # higher spikes
        plt.subplot(2,2,3)
        events = self.higher_srec.get("events")
        ts = events["times"]
        plt.title("Number Lower Spike Timing")
        plt.yticks(range(1,2))
        plt.plot(ts, np.full(len(events["times"]), 1), ".", color='royalblue', linewidth=6)

        # lower spikes
        plt.subplot(2,2,4)
        events = self.lower_srec.get("events")
        ts = events["times"]
        plt.title("Number Higher Spike Timing")
        plt.yticks(range(1,2))
        plt.plot(ts, np.full(len(events["times"]), 1), ".", color='tomato', linewidth=6)
        plt.show()

def plot_heatmap(iteration = 1):
    insensitivity_list = [x*20 for x in range(1, 10)]
    sp_list = [x*20 for x in range(1, 10)]
    sp_list.reverse()
    print(sp_list)
    results = []
    for row in sp_list:
        row_list = []
        for column in insensitivity_list:
            counter_low = 0
            counter_high = 0
            for i in range(iteration):
                hammel_model = Num_Counting_Model()

                # change simulation value
                hammel_model.insen_noise.rate = column
                hammel_model.sp_noise.rate = row

                hammel_model.setup_network()

                counter_low += hammel_model.count_spikes(neuron_low=True)
                counter_high += hammel_model.count_spikes(neuron_low=False)
            counter = (counter_high/iteration) - (counter_low/iteration)
            row_list.append(counter)
        # results.insert(0, row_list)
        results.append(row_list)
    print(results)
    mpl.rc('xtick', labelsize = 15)
    mpl.rc('ytick', labelsize = 15)
    fig, ax = plt.subplots(figsize=(28,16))
    im = ax.imshow(results)
    
    mpl.rcParams['text.color'] = 'white'
    for x in range(len(sp_list)):
        for y in range(len(insensitivity_list)):
            plt.text(x, y, str(results[y][x]), horizontalalignment='center', verticalalignment='center', weight='bold')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(insensitivity_list)))
    ax.set_yticks(np.arange(len(sp_list)))
    ax.xaxis.label.set_size(17)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_size(17)
    ax.yaxis.label.set_fontweight('bold')
    ax.set_xlabel("Poisson generator to insensitivity neuron firing rate (Hz)")
    ax.set_ylabel("Poisson generator to sp neuron firing rate (Hz)")

    ax.set_xticklabels(insensitivity_list)
    ax.set_yticklabels(sp_list)

    ax.set_title(f"Spike Count for (Higher Neuron - Lower Neuron)", fontdict={'fontsize': 20, 'weight': 'bold'})
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

class NumberLine():
    def __init__(self):
        self.current_rate_rep = {'0' : (0,40), '1' : (40, 80), '2' : (80,120), '3' : (120,160), '4' : (160,200)}
        self.external_num_rep = '2'
        # self.external_num_rate = np.random.uniform(self.current_rate_rep[self.external_num_rep][0], self.current_rate_rep[self.external_num_rep][1])
        self.external_num_rate = 60
        self.current_internal_num_rep = None
        self.current_internal_rate = 0.0
        self.model = Num_Counting_Model(batch=False, sp_rate= self.current_internal_rate, insen_rate=self.external_num_rate)
        self.model.setup_network()

    def set_model(self):
        current_high_spikes = []
        current_low_spikes = []
        for _ in range(BATCH_ITERATION):
            self.model.sp_noise.rate = self.current_internal_rate
            self.model.run_network()
            events = self.model.higher_srec.get("events")
            high_senders = events["senders"]
            events = self.model.lower_srec.get("events")
            low_senders = events["senders"]
            if len(high_senders) - len(current_high_spikes) > 0:
                self.current_internal_rate -= 5 * np.abs(self.current_internal_rate)/100
                current_high_spikes = high_senders
            if len(low_senders) - len(current_low_spikes) > 0:
                self.current_internal_rate += 5 * np.abs(self.current_internal_rate - 200)/100
                current_low_spikes = low_senders
            
            # print(f"Last spike: {high_senders[len(high_senders) - 1]}")
            print(f"Sp_rate: {self.current_internal_rate}, Insen_rate: {self.external_num_rate}, High spike count: {len(high_senders)}, Low spike count: {len(low_senders)}")

# number = NumberLine()
# number.set_model()
# plot_heatmap()
test = Num_Counting_Model()
test.setup_network()
test.plot_seminar()