import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import common

CF_CUTOFF = 350
CF_TAPS = 256
CF_GAIN = 1
CF_IMPULSE_RESPONSE = common.quantize(CF_GAIN * signal.firwin(CF_TAPS, CF_CUTOFF, fs=common.sample_rate, scale=True))

# CF2_TRANSITION_WIDTH = 100
# CF2_IMPULSE_RESPONSE = common.quantize(CF_GAIN * signal.remez(CF_TAPS,
#     [0, CF_CUTOFF, CF_CUTOFF + CF2_TRANSITION_WIDTH, 0.5 * common.sample_rate],
#     [1, 0],
#     fs=common.sample_rate
# ))

def apply_cf(s):
    # return np.convolve(s, CF_IMPULSE_RESPONSE)
    return common.quantize(np.convolve(s, CF_IMPULSE_RESPONSE))

def plot_window():
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(f"channel filter, cutoff={CF_CUTOFF}, taps={CF_TAPS}")
    ax1.set_title("impulse response")
    ax1.plot(CF_IMPULSE_RESPONSE)
    
    w, h = signal.freqz(CF_IMPULSE_RESPONSE, fs=common.sample_rate)
    ax2.set_title("freq response")
    ax2.plot(w, 20 * np.log10(np.absolute(h)))
    ax2.plot(w, np.angle(h))

if __name__ == "__main__":
    plot_window()
    plt.show()