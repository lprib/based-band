import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import common

CF_CUTOFF = 350
CF_TAPS = 128
CF_GAIN = 1
CF_IMPULSE_RESPONSE = common.quantize(CF_GAIN * signal.firwin(CF_TAPS, CF_CUTOFF, fs=common.sample_rate, scale=True))

def apply_cf(s):
    # return np.convolve(s, CF_IMPULSE_RESPONSE)
    return common.quantize(np.convolve(s, CF_IMPULSE_RESPONSE))

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(f"channel filter, cutoff={CF_CUTOFF}, taps={CF_TAPS}")
    ax1.set_title("impulse response")
    ax1.plot(CF_IMPULSE_RESPONSE)
    
    w, h = signal.freqz(CF_IMPULSE_RESPONSE, fs=common.sample_rate)
    ax2.set_title("freq response")
    ax2.plot(w, 20 * np.log10(np.absolute(h)))
    ax2.plot(w, np.angle(h))
    plt.show()