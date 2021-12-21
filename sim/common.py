import numpy as np

symbols = np.array([1, 3, -1, -3])
phase_multiplier = np.pi / 4

sync_pattern_dibits_half = np.array([1, 3, 1, 1, 3, 1, 3, 1, 3, 3, 3, 1])
# sync_pattern_dibits = sync_pattern_dibits_half
sync_pattern_dibits = np.concatenate((sync_pattern_dibits_half, np.flip(sync_pattern_dibits_half)))

# hz
max_frequency_dev = 400

# hz
symbol_rate = 100
symbol_period = 1 / symbol_rate
oversample_rate = 10

sample_rate = symbol_rate * oversample_rate
sample_period = 1 / sample_rate

# Q7 fixed point
fixed_point_mult = 2**7

def quantize(n):
    return np.around(n * fixed_point_mult) / fixed_point_mult

def clip(n):
    return np.clip(n, -1, 1)

def to_fixed(n):
    return np.around(n * fixed_point_mult)

def from_fixed(n):
    return n / fixed_point_mult

def get_t(n, sample_rate=sample_rate):
    return np.arange(len(n)) / sample_rate

def as_c_string(n):
    return ", ".join([str(x) for x in np.around(n * fixed_point_mult).astype(int)])