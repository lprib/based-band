import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

import common
import cf
import bb
import rrc

PASSBAND_FS = 40000

BB_INTERPOLATION_TAPS = 128
BB_INTERPOLATION_IMPULSE_RESPONSE = signal.firwin(BB_INTERPOLATION_TAPS, cf.CF_CUTOFF, fs=PASSBAND_FS, scale=True)

# TODO adjust this.... also the baseband CF isn't needed if we have this (?)
UPSAMPLED_CF_CUTOFF = 800
UPSAMPLED_CF_TAPS = 128
UPSAMPLED_CF_IMPULSE_RESPONSE = signal.firwin(UPSAMPLED_CF_TAPS, UPSAMPLED_CF_CUTOFF, fs=PASSBAND_FS, scale=True)

def upsample_baseband(baseband_signal):
    # assert integer interpolation
    assert PASSBAND_FS % common.sample_rate == 0
    upsample_rate = PASSBAND_FS // common.sample_rate
    baseband_zoh = np.repeat(baseband_signal, upsample_rate)
    baseband_interp = np.convolve(baseband_zoh, BB_INTERPOLATION_IMPULSE_RESPONSE)
    return baseband_interp

def downsample_passband(channel_filtered_passband_demod):
    # assert integer interpolation
    assert PASSBAND_FS % common.sample_rate == 0
    downsample_rate = PASSBAND_FS // common.sample_rate
    return channel_filtered_passband_demod[::downsample_rate]

def modulate(carrier_freq, baseband_iq):
    baseband_upsample = upsample_baseband(baseband_iq)
    t = np.arange(len(baseband_upsample)) / PASSBAND_FS
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)
    return np.real(baseband_upsample) * carrier_i + np.imag(baseband_upsample) * carrier_q


# also applies channel filter
def demodulate(carrier_freq, passband_samples):
    t = np.arange(len(passband_samples)) / PASSBAND_FS
    vco_i = np.cos(2 * np.pi * carrier_freq * t)
    vco_q = np.sin(2 * np.pi * carrier_freq * t)
    upsampled_iq = vco_i * passband_samples + 1j * vco_q * passband_samples
    channel_filtered_iq = np.convolve(upsampled_iq, UPSAMPLED_CF_IMPULSE_RESPONSE)
    return channel_filtered_iq

def mod_demod(carrier_freq, baseband_iq):
    mod = modulate(carrier_freq, baseband_iq)
    demod = demodulate(carrier_freq, mod)
    demod_downsample = downsample_passband(demod)
    return demod_downsample

def get_random_passband_signal(carrier_freq):
    baseband = bb.phase_mod(bb.pulse_shape(bb.random_dibits(1000)))
    mod = modulate(carrier_freq, baseband)
    return mod

def write_passband_wav():
    # simulate multiple channels:
    mod = get_random_passband_signal(1000)
    for i in np.arange(1500, 5000, 500):
        mod = mod + get_random_passband_signal(i)

    # single channel
    # mod = get_random_passband_signal(500)

    mod = mod / np.max(np.absolute(mod))

    filename = "passband.wav"
    wavfile.write(filename, PASSBAND_FS, mod)

    f, pxx = signal.periodogram(mod, fs=PASSBAND_FS)
    plt.figure()
    plt.plot(f, pxx)


def plot_upsampled_psd():
    plt.figure()

    baseband = bb.phase_mod(bb.pulse_shape(bb.random_dibits(1000)))
    baseband_upsample = upsample_baseband(baseband)

    bb_f, bb_pxx = signal.periodogram(baseband, fs=common.sample_rate)
    upsample_f, upsample_pxx = signal.periodogram(baseband_upsample, fs=PASSBAND_FS)

    plt.plot(bb_f, bb_pxx)
    plt.plot(upsample_f, upsample_pxx, alpha=0.5)

def plot_psd():
    plt.figure()

    carrier_freq = 12000
    interference_freq = 13600
    awgn_power = 0.0

    # Delays
    # group delay of the baseband->passband interpolation filter
    interp_filter_delay_s = (BB_INTERPOLATION_TAPS + 1) / (2 * PASSBAND_FS)
    # group delay of the passband channel filter
    mod_demod_delay_s = (UPSAMPLED_CF_TAPS + 1) / (2 * PASSBAND_FS)
    # group delay of the pulse shaping filter (full signal passes through this twice, matched filter)
    pulse_shape_delay_s = (len(rrc.RRC_IMPULSE_RESPONSE) + 1) / (2 * common.sample_rate)
    # total incurred delay between transmit and recieve
    total_incurred_delay_s = interp_filter_delay_s + mod_demod_delay_s + pulse_shape_delay_s
    # number of baseband samples of the total incurred delay
    # This is used to disable baseband phase demod until the samples are actually ariving.
    # Otherwise it would attempt to demod noise, which will through the phase completely out of wack
    total_incurred_delay_bb_samples = int(np.ceil(total_incurred_delay_s / common.sample_period))

    pattern = bb.random_dibits(1000)
    tx_pulse_shape = bb.pulse_shape(pattern)
    tx_sample_instants = bb.get_sampling_instants(pattern)
    baseband = bb.phase_mod(tx_pulse_shape)
    passband_modulated = modulate(carrier_freq, baseband)

    interference_pattern = bb.random_dibits(1000)
    interference_baseband = bb.phase_mod(bb.pulse_shape(interference_pattern))
    interference_passband_modulated = modulate(interference_freq, interference_baseband)

    awgn = np.random.normal(0, np.sqrt(awgn_power), len(passband_modulated))

    full_passband = passband_modulated + interference_passband_modulated + awgn

    upsampled_demod = demodulate(carrier_freq, full_passband)

    bb_f, bb_pxx = signal.periodogram(baseband, fs=common.sample_rate)
    mod_f, mod_pxx = signal.periodogram(full_passband, fs=PASSBAND_FS)
    demod_f, demod_pxx = signal.periodogram(upsampled_demod, fs=PASSBAND_FS)
    plt.plot(bb_f, bb_pxx, label="baseband (no upsample) PSD")
    plt.plot(mod_f, mod_pxx, label=f"passband modulated PSD, carrier={carrier_freq}, interference={interference_freq}")
    plt.plot(demod_f, demod_pxx, label=f"passband demodulated PSD, carrier={carrier_freq}")
    plt.legend()

    demod = downsample_passband(upsampled_demod)[total_incurred_delay_bb_samples:]
    rx_phase_demod = bb.phase_demod(demod)
    rx_pulse_shape = np.convolve(rx_phase_demod, rrc.RRC_IMPULSE_RESPONSE)

    # delay the first time base by the two filters' group delay so the plots line up
    t1 = common.get_t(baseband)
    # we delay this by the pulse shaping filter delay so that it will line up
    # with the graphs below, which plot signals that have also been through the
    # RX pulse shaping (PDM filter)
    t2 = common.get_t(demod) + pulse_shape_delay_s

    _, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title("Modulated then demodulated (phase)")
    ax1.plot(t1, np.angle(baseband), label="transmitted baseband phase")
    ax1.plot(t2, np.angle(demod), label="received baseband phase")
    ax1.legend()

    delayed_tx_sample_instants = tx_sample_instants + total_incurred_delay_s
    # no need to delay the tx/rx to line up with eachother, size we have
    # already sliced off the starting samples of the RX so that it doesn't try
    # to demod noise
    t3 = common.get_t(tx_pulse_shape)
    t4 = common.get_t(rx_pulse_shape)
    ax2.set_title("Modulated then demodulated (pulse)")
    ax2.plot(t3, tx_pulse_shape, "o-", label="transmitted baseband pulse")
    ax2.plot(t4, rx_pulse_shape, "o-", label="received baseband pulse")
    ax2.vlines(delayed_tx_sample_instants, -1, 1, label="sampling instants")
    ax2.legend()


if __name__ == "__main__":
    plot_psd()
    # plot_mod_demod()
    # plot_upsampled_psd()
    # plt.show()
    # write_passband_wav()
    plt.show()