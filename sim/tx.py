import numpy as np
import matplotlib.pyplot as plt

import common
import rrc

def random_dibits(length):
    return np.floor(np.random.rand(length) * 4).astype(int)

def zero_interpolate(dibits):
    symbols = common.symbols[dibits]
    # pre-pads zeros as well
    interp = np.zeros((len(dibits) + 1) * common.oversample_rate)
    interp[common.oversample_rate::common.oversample_rate] = symbols
    return interp

def pulse_shape(dibits):
    interp = zero_interpolate(dibits)
    pulse_shaped = np.convolve(interp, rrc.RRC_IMPULSE_RESPONSE)
    return common.quantize(pulse_shaped)
    # phase = symbols * common.phase_multiplier

if __name__ == "__main__":
    np.random.seed(0)
    plt.figure()
    test_dibits = random_dibits(20)
    tx_interp = zero_interpolate(test_dibits)

    # delay in seconds of rrc filter
    rrc_filter_len_s = (len(rrc.RRC_IMPULSE_RESPONSE) - 1) / common.sample_rate / 2

    # delay plot by 2x filter delay so they line up
    plt.plot(common.get_t(tx_interp) + 2 * rrc_filter_len_s, tx_interp, "o-")

    tx_pulse = pulse_shape(test_dibits)
    plt.plot(common.get_t(tx_pulse) + rrc_filter_len_s, tx_pulse, "o-")

    rx_pulse = common.quantize(np.convolve(tx_pulse, rrc.RRC_IMPULSE_RESPONSE))
    plt.plot(common.get_t(rx_pulse), rx_pulse, "o-")

    plt.show()
