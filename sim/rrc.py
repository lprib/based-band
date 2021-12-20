import numpy as np
import sys
from scipy import signal
import matplotlib.pyplot as plt

import common

RRC_FILTER_GAIN = 1
RRC_FILTER_NUM_SYMBOLS = 6


def rrc_equation(t, beta):
    if t == 0:
        return common.symbol_rate * (1 + beta * (4/np.pi - 1))
    elif t == (common.symbol_period / (4 * beta)) or t == (-common.symbol_period / (4 * beta)):
        scale = beta / (common.symbol_period * np.sqrt(2))
        sin_term = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        cos_term = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        return scale * (sin_term + cos_term)
    else:
        numerator = np.sin(np.pi * t * common.symbol_rate * (1 - beta)) + 4 * beta * t * common.symbol_rate * np.cos(np.pi * t * common.symbol_rate * (1 + beta))
        denomenator = np.pi * t * common.symbol_rate * (1 - np.square(4 * beta * t * common.symbol_rate))
        return common.symbol_rate * (numerator / denomenator)

# symmetric, type 2
# we only generate one half of the filter, the other half is mirroed (type 2 FIR)
def get_rrc_impulse(beta, num_symbols):
    total_num_taps = common.oversample_rate * num_symbols
    # must be even for type 2
    assert total_num_taps % 2 == 0
    # Time of the impulse response in seconds
    total_filter_time = common.symbol_period * num_symbols
    # time vector centered around zero
    t = np.linspace(-total_filter_time / 2, total_filter_time /2, total_num_taps)
    # only take the top half of the time vector, since its a Type 2 FIR
    t = t[total_num_taps // 2:]

    assert len(t) == total_num_taps // 2, "mirroed segment length is not exactly half of total filter length"
    response = np.array([rrc_equation(tt, beta) for tt in t])
    response = np.concatenate((np.flip(response), response))
    return response

def post_process_response(resp):
    return common.quantize(RRC_FILTER_GAIN * resp / np.sum(resp))

# final filter
RRC_IMPULSE_RESPONSE = post_process_response(get_rrc_impulse(0.75, RRC_FILTER_NUM_SYMBOLS))

if __name__ == "__main__":
    num_symbols = RRC_FILTER_NUM_SYMBOLS
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    fig.suptitle(f"RRC pulse shaping, oversample_rate={common.oversample_rate}, filter_spread_symbols={num_symbols}, filter_gain={RRC_FILTER_GAIN}")
    for beta in [0.01, 0.5, 0.75, 1.0]:
        resp = get_rrc_impulse(beta, num_symbols)
        # sys.exit(1)
        resp = post_process_response(resp)
        resp_2 = common.quantize(np.convolve(resp, resp))

        ax0.set_title("single RRC impulse")
        ax0.plot(common.get_t(resp), resp, label=f"beta = {beta}")

        ax1.set_title("matched filter RC impulse")
        ax1.plot(common.get_t(resp_2), resp_2, label=f"beta = {beta}")

        w, h = signal.freqz(resp, fs=common.sample_rate)

        ax2.set_title("matched filter magnitude, dB")
        ax2.plot(w, 20*np.log10(np.absolute(h)), label=f"mag, beta = {beta}")

        ax3.set_title("matched filter phase, dB")
        ax3.plot(w, np.angle(h), label=f"phase, beta={beta}")

        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()

    plt.tight_layout()
    plt.show()