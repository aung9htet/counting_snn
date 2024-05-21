import numpy as np

def load_data(number=2, data_index = 1):
    """
    Read Data
    Args:
        Number - number representation
        Data Index - Choose which run index(0-19)
    """
    file_name = "model_data/data_spike" + str(number) + ".npy"
    data = np.load(file_name, allow_pickle=True)[data_index]
    high_spike_recorder = data.get("high_spike")
    low_spike_recorder = data.get("low_spike")
    ring_spike_recorders = data.get("ring_spike")
    gain_high_spike_recorders = data.get("gain_high_spike")
    gain_low_spike_recorders = data.get("gain_low_spike")
    perirhinal_fdn_spike_recorders = data.get("per_fdn_spike")
    perihinal_inhibitory_spike_recorder = data.get("per_in_spike")
    perihinal_decision_spike_recorder = data.get("per_de_spike")
    perihinal_vision_decision_spike_recorder = data.get("per_vde_spike")
    print(high_spike_recorder)

test = load_data()