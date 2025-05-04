import sys

import numpy as np
from enum import StrEnum
import seaborn as sns
from matplotlib import pyplot as plt
from pycparser.ply.yacc import MAXINT

sns.set_theme()

class Base(StrEnum):
    A = "A"
    T = "T"
    C = "C"
    G = "G"

def base_to_index(base: str) -> int:
    match base:
        case Base.A: return 0
        case Base.T: return 1
        case Base.C: return 2
        case Base.G: return 3


params = {
    "bases": [Base.A, Base.T, Base.C, Base.G],
    "bias": [0.1, 0.1, 0.4, 0.4],
    "tf_length": 8,
    "promoter_length": 100,
    "amount_tf": 1
}

def ppm_score(ppm: np.array) -> float:
    n, m = ppm.shape
    ic = 0
    for i in range(n):
        for j in range(m):
            ic -= ppm[i, j] * np.log2(ppm[i, j])

    return ic

def get_ppm_pwm() -> (np.array, np.array):
    """
    Get position probability and weight matrix
    :return: ppm and pwm
    """
    # A
    # T
    # C
    # G
    ppm = np.array([
        [0.05, 0.05, 0.1, 0.05, 0.025, 0.05, 0.15, 0.05],
        [0.05, 0.025, 0.05, 0.075, 0.025, 0.05, 0.15, 0.05],
        [0.05, 0.9, 0.8, 0.125, 0.05, 0.8, 0.65, 0.85],
        [0.85, 0.025, 0.05, 0.75, 0.9, 0.1, 0.05, 0.05]
    ])

    assert (np.sum(ppm, axis=0) == 1).all()
    # M = log2(M / b) where b = 0.25 and M is the PPM
    pwm = np.log2(ppm * 4)

    return ppm, pwm

def get_tf_sequence(ppm: np.array) -> np.array:
    """
    Generate TF sequence based on ppm
    :param ppm: position-specific probability matrix
    :return: TF sequence
    """
    n, m = ppm.shape
    tf = ["G", "C", "C", "G", "G", "C", "C", "C"]
    return tf

class SegmentationModel:
    R = []
    R_prime = []
    G = []
    sequence: str = ""
    weight_matrices: np.array

    def __init__(self, sequence: str, weight_matrices):
        self.sequence = sequence
        self.R = np.zeros(len(sequence))
        self.R_prime = np.zeros(len(sequence))
        self.G = np.zeros(len(sequence))

        self.weight_matrices = weight_matrices

    def calc_forward(self, i: int) -> float:
        p_alpha = 0.25
        _, l = self.weight_matrices.shape
        sum = 0
        # First i where promoter seq. of length 8 fits
        if i >= l - 1:
            p = 1
            R = 1
            for k in range(0, l):
                base = self.sequence[i - l + 1 + k]
                p *= self.weight_matrices[base_to_index(base), k]
            for k in range(i - l + 1, i):
                R *= self.R[k]
            sum += p * (1 / R)
        return p_alpha + sum

    def calc_backward(self, i: int) -> float:
        p_alpha = 0.25
        _, l = self.weight_matrices.shape
        sum = 0
        if i <= len(self.sequence) - l:
            p = 1
            R = 1
            for k in range(0, l):
                base = self.sequence[i + k]
                p *= self.weight_matrices[base_to_index(base), k]
            for k in range(i + 1, i + l):
                R *= self.R_prime[k]
            sum += p * (1 / R)
        return p_alpha + sum


    def calc_G(self, i: int, l: int) -> float:
        if i >= l - 1:
            ratio = 1
            for k in range(0, i - l + 1):
                ratio *= self.R[k] / self.R_prime[k]

            denom = 1
            for k in range(i - l + 1, i + 1):
                denom *= self.R_prime[k]
            return ratio / denom
        else:
            return 0

if __name__ == '__main__':
    # Randomly generate promoter sequence
    promoter_sequence = np.random.choice(params["bases"], params["promoter_length"], p=[0.25, 0.25, 0.25, 0.25])

    ppm, pwm = get_ppm_pwm()

    # Insert tf into promoter sequence
    positions = []
    for i in range(params["amount_tf"]):
        start = np.random.choice(range(params["promoter_length"] - params["tf_length"] - 1))
        end = start + params["tf_length"]
        tf = get_tf_sequence(ppm)
        for k in range(len(tf)):
            promoter_sequence[start + k] = tf[k]
        positions.append([start, end])

    promoter_sequence = "".join(promoter_sequence)
    print(f"PPM : {ppm}\n"
          f">Promoter\n{promoter_sequence}\n"
          f"TF positions : {positions}")

    SM = SegmentationModel(promoter_sequence, ppm)
    for i in range(len(promoter_sequence)):
        SM.R[i] = SM.calc_forward(i)
        SM.R_prime[len(promoter_sequence) - 1 - i] = SM.calc_backward(len(promoter_sequence) - 1 - i)

    for i in range(len(promoter_sequence)):
        SM.G[i] = SM.calc_G(i, 8)

    max = 0
    for i in range(len(SM.G)):
        if SM.G[max] == 0 or SM.G[i] > SM.G[max]:
            max = i

    print(
        f"{'Sequence':<8}| {promoter_sequence}\n"
        f"{'True':<8}| {'':<{positions[0][0]}}++++++++{'':>{positions[0][1]}}\n"
        f"{'Pred':<8}| {'':<{max}}--------{'':>{max + params["tf_length"]}}")

    tf_arr_0 = np.zeros((1, params["promoter_length"]))
    for i in range(positions[0][0], positions[0][1]):
        tf_arr_0[0, i] = SM.G[max]

    expanded = np.expand_dims(SM.G, axis=0)
    print(expanded)
    a = np.vstack((tf_arr_0, expanded))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(a)
    ax.grid(False)
    plt.gca().set_aspect(10)
    fig.show()