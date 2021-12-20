import numpy as np
import matplotlib.pyplot as plt

import common
import rrc

def random_dibits(length):
    return np.floor(np.random.rand(length) * 4).astype(int)

def zero_interpolate(dibits):
    symbols = common.symbols[dibits]
    interp = np.zeros(len(dibits) * common.oversample_rate)
    interp[::common.oversample_rate] = symbols
    return interp

def pulse_shape(dibits):
    interp = zero_interpolate(dibits)
    pulse_shaped = np.convolve(interp, rrc.RRC_IMPULSE_RESPONSE, mode="same")
    return pulse_shaped
    # phase = symbols * common.phase_multiplier

if __name__ == "__main__":
    np.random.seed(0)
    plt.figure()
    test_dibits = random_dibits(20)
    tx_interp = zero_interpolate(test_dibits)
    plt.plot(common.get_t(tx_interp), tx_interp)

    tx_pulse = pulse_shape(test_dibits)
    plt.plot(common.get_t(tx_pulse), tx_pulse)

    rx_pulse = np.convolve(tx_pulse, rrc.RRC_IMPULSE_RESPONSE, mode="same")
    plt.plot(common.get_t(rx_pulse), rx_pulse)

    plt.show()
