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

def get_phase(samples):
    freq = samples * common.max_frequency_dev
    phase = np.cumsum(freq) * common.sample_period
    return phase

def phase_mod(samples):
    freq = samples * common.max_frequency_dev
    phase = np.cumsum(freq) * common.sample_period
    t = np.arange(len(samples)) * common.sample_period
    return common.quantize(np.exp(2*np.pi*1j*(t + phase)))

def phase_demod(samples):
    diff = np.diff(np.angle(samples))
    # handle angle roll-over
    for i in range(len(diff)):
        if diff[i] > np.pi:
            diff[i] -= 2*np.pi
        if diff[i] < -np.pi:
            diff[i] += 2*np.pi
    return diff

if __name__ == "__main__":
    np.random.seed(0)
    plt.figure()
    test_dibits = random_dibits(20)
    tx_interp = zero_interpolate(test_dibits)

    # delay in seconds of rrc filter
    rrc_filter_len_s = (len(rrc.RRC_IMPULSE_RESPONSE) - 1) / common.sample_rate / 2

    # delay plot by 2x filter delay so they line up
    plt.plot(common.get_t(tx_interp) + 2 * rrc_filter_len_s, tx_interp, "x-", alpha=0.3, label="zero interp tx")

    tx_pulse = pulse_shape(test_dibits)
    plt.plot(common.get_t(tx_pulse) + rrc_filter_len_s, tx_pulse, "x-", label="rrc shaped tx")

    # tx_phase = phase_mod(tx_pulse)
    phase = phase_mod(tx_pulse)
    plt.plot(common.get_t(phase) + rrc_filter_len_s, np.real(phase), alpha=0.3, label="freq mod tx real")
    plt.plot(common.get_t(phase) + rrc_filter_len_s, np.imag(phase), alpha=0.3, label="freq mod tx imag")

    phase_demod = phase_demod(phase)
    plt.plot(common.get_t(phase_demod) + rrc_filter_len_s, np.real(phase_demod), label="phase demod")

    rx_pulse = common.quantize(np.convolve(phase_demod, rrc.RRC_IMPULSE_RESPONSE))
    plt.plot(common.get_t(rx_pulse), rx_pulse, "x-", label="rc shaped rx")

    plt.legend()
    plt.tight_layout()
    plt.show()
