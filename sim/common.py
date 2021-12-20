import numpy as np

symbols = np.array([-1, -3, 1, 3])
phase_multiplier = np.pi / 4

# hz
symbol_rate = 1
symbol_period = 1 / symbol_rate
oversample_rate = 10

sample_rate = symbol_rate * oversample_rate

# Q7 fixed point
fixed_point_mult = 2**7

def quantize(n):
    return np.around(n * fixed_point_mult) / fixed_point_mult

def to_fixed(n):
    return np.around(n * fixed_point_mult)

def from_fixed(n):
    return n / fixed_point_mult

def get_t(n, sample_rate=sample_rate):
    return np.arange(len(n)) / sample_rate

def as_c_string(n):
    return ", ".join([str(x) for x in np.around(n * fixed_point_mult).astype(int)])