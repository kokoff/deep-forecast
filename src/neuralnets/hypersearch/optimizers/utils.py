import numpy as np
from sobol_seq import i4_sobol_generate


def generate_sobol_sequences(num_sequences, lb, ub):
    sequences = []
    sobol_sequences = i4_sobol_generate(len(lb), num_sequences)
    for i in sobol_sequences:
        sequence = i * np.abs(ub - lb)
        sequence = sequence + lb
        sequences.append(sequence)
    return sequences
