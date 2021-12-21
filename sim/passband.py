import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.sparse import base

import common
import cf
import bb

PASSBAND_FS = 40000

BB_INTERPOLATION_TAPS = 128
BB_INTERPOLATION_IMPULSE_RESPONSE = signal.firwin(BB_INTERPOLATION_TAPS, cf.CF_CUTOFF, fs=PASSBAND_FS, scale=True)

UPSAMPLED_CF_CUTOFF = cf.CF_CUTOFF
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

def plot_upsampled_psd():
    plt.figure()

    baseband = bb.phase_mod(bb.pulse_shape(bb.random_dibits(1000)))
    baseband_upsample = upsample_baseband(baseband)
    
    bb_f, bb_pxx = signal.periodogram(baseband, fs=common.sample_rate)
    upsample_f, upsample_pxx = signal.periodogram(baseband_upsample, fs=PASSBAND_FS)
    
    plt.plot(bb_f, bb_pxx)
    plt.plot(upsample_f, upsample_pxx, alpha=0.5)

def plot_mod_demod():
    plt.figure()

    baseband = bb.phase_mod(bb.pulse_shape(bb.random_dibits(100)))
    demod = mod_demod(12000, baseband)
    
    interp_filter_delay_s = (BB_INTERPOLATION_TAPS + 1) / (2 * PASSBAND_FS)
    mod_demod_delay_s = (UPSAMPLED_CF_TAPS + 1) / (2 * PASSBAND_FS)
    # delay the first time base by the two filters' group delay so the plots line up
    t1 = common.get_t(baseband) + interp_filter_delay_s + mod_demod_delay_s
    t2 = common.get_t(demod)
    
    print(baseband.shape, demod.shape)
    plt.plot(t1, np.angle(baseband))
    plt.plot(t2, np.angle(demod))

def plot_psd():
    plt.figure()

    carrier_freq = 12000
    pattern = bb.random_dibits(1000)
    baseband = bb.phase_mod(bb.pulse_shape(pattern))
    passband_modulated = modulate(carrier_freq, baseband)
    
    upsampled_demod = demodulate(carrier_freq, passband_modulated)

    bb_f, bb_pxx = signal.periodogram(baseband, fs=common.sample_rate)
    mod_f, mod_pxx = signal.periodogram(passband_modulated, fs=PASSBAND_FS)
    demod_f, demod_pxx = signal.periodogram(upsampled_demod, fs=PASSBAND_FS)
    plt.plot(bb_f, bb_pxx, label="baseband (no upsample) PSD")
    plt.plot(mod_f, mod_pxx, label=f"passband modulated PSD, carrier={carrier_freq}")
    plt.plot(demod_f, demod_pxx, label=f"passband demodulated PSD, carrier={carrier_freq}")
    plt.legend()
    

if __name__ == "__main__":
    # plot_psd()
    plot_mod_demod()
    # plot_upsampled_psd()
    plt.show()