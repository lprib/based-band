import numpy as np

import common

def encode_phase(dibits):
    symbols = common.symbols[dibits]
    phase = symbols * common.phase_multiplier
