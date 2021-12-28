import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import common
import rrc
import cf

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

def get_sampling_instants(dibits):
    # get list of samping instants (seconds) for an interpolated, pulse-shaped signal
    # interpolation zero-pads the beginning of the signal
    interpolate_delay_s = 10 * common.sample_period
    pulse_shape_delay_s = (len(rrc.RRC_IMPULSE_RESPONSE) + 1) / (2 * common.sample_rate)
    return np.arange(len(dibits)) * common.symbol_period + interpolate_delay_s + pulse_shape_delay_s

def get_phase(samples):
    freq = samples * common.max_frequency_dev
    phase = np.cumsum(freq) * common.sample_period
    return phase

def phase_mod(samples):
    """Actually frequency modulation, not phase modulation"""
    # Convert pulse-shaped symbols into frequency/time series
    freq = samples * common.max_frequency_dev
    # Phase is the integral (ie. cumsum times period)
    phase = np.cumsum(freq) * common.sample_period
    t = np.arange(len(samples)) * common.sample_period
    return common.quantize(np.exp(2 * np.pi * 1j * (t + phase)))

def get_sync_taps():
    pulse = pulse_shape(common.sync_pattern_dibits)
    phase = phase_mod(pulse)
    # # chan_filtered = phase
    chan_filtered = common.quantize(np.convolve(phase, cf.CF_IMPULSE_RESPONSE, mode="same"))
    taps = np.flip(np.conjugate(chan_filtered))
    return taps

def phase_demod(samples):
    diff = np.diff(np.angle(samples))
    diff = np.unwrap(diff)
    return diff

def plot_sync():
    pattern = get_sync_taps()
    plt.figure()
    plt.plot(np.angle(pattern))

def plot_psd():
    pattern = random_dibits(1000)
    modulated = cf.apply_cf(phase_mod(pulse_shape(pattern)))
    f, pxx = signal.periodogram(modulated, fs=common.sample_rate)
    plt.figure()
    plt.plot(f, pxx)

def plot_sync_test():
    np.random.seed(0)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    pattern = np.concatenate((random_dibits(10), common.sync_pattern_dibits, random_dibits(200), common.sync_pattern_dibits, random_dibits(100)))
    shaped = pulse_shape(pattern)
    modulated = phase_mod(shaped)
    ax1.plot(np.real(modulated), label="real")
    ax1.plot(np.imag(modulated), label="imag")
    ax1.legend()
    
    ax2.plot(np.angle(modulated))
    
    plt.figure()
    # pattern_with_random = np.concatenate((random_dibits(20), random_dibits(10), random_dibits(20)))
    s = modulated
    # s = cf.apply_cf(modulated)
    
    corr = np.convolve(s, get_sync_taps())
    corr_mag = np.square(np.real(np.absolute(corr)))
    plt.plot(corr_mag)

def plot_bb():
    np.random.seed(0)
    plt.figure()
    test_dibits = random_dibits(20)
    # test_dibits = common.sync_pattern_dibits
    tx_interp = zero_interpolate(test_dibits)

    # delay in seconds of rrc filter
    rrc_filter_len_s = (len(rrc.RRC_IMPULSE_RESPONSE) - 1) / common.sample_rate / 2

    # delay plot by 2x filter delay so they line up
    plt.plot(common.get_t(tx_interp) + 2 * rrc_filter_len_s, tx_interp, "x-", alpha=0.3, label="zero interp tx")

    tx_pulse = pulse_shape(test_dibits)
    # here we delay by an ADDITIONAL rrc_filter_len_s because we're trying to
    # line it up with the RX pulse shape which has been through the filter
    # twice (matched filter)
    tx_sample_instants = get_sampling_instants(test_dibits) + rrc_filter_len_s
    plt.plot(common.get_t(tx_pulse) + rrc_filter_len_s, tx_pulse, "x-", label="rrc shaped tx")
    plt.vlines(tx_sample_instants, -1, 1, label="sampling instants")

    # tx_phase = phase_mod(tx_pulse)
    phase = phase_mod(tx_pulse)
    plt.plot(common.get_t(phase) + rrc_filter_len_s, np.real(phase), alpha=0.3, label="freq mod tx real")
    plt.plot(common.get_t(phase) + rrc_filter_len_s, np.imag(phase), alpha=0.3, label="freq mod tx imag")

    demod = phase_demod(phase)
    plt.plot(common.get_t(demod) + rrc_filter_len_s, np.real(demod), label="phase demod")

    rx_pulse = common.quantize(np.convolve(demod, rrc.RRC_IMPULSE_RESPONSE))
    plt.plot(common.get_t(rx_pulse), rx_pulse, "x-", label="rc shaped rx")

    plt.legend()
    plt.tight_layout()

if __name__ == "__main__":
    # plot_sync()
    # plot_sync_test()
    # plot_psd()
    plot_bb()
    plt.show()