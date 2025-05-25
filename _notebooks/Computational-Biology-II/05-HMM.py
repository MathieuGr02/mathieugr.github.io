import numpy as np
from enum import StrEnum, IntEnum
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

sns.set_theme()

class Base(StrEnum):
    A = "A"
    T = "T"
    C = "C"
    G = "G"

class BaseInt(IntEnum):
    A = 0
    T = 1
    C = 2
    G = 3

class State(IntEnum):
    Start = 0
    Random = 1
    BindingSite = 2
    End = 3

def base_to_index(base: str) -> int:
    match base:
        case Base.A: return 0
        case Base.T: return 1
        case Base.C: return 2
        case Base.G: return 3

def index_to_base(index: int) -> str:
    match index:
        case 0:
            return Base.A
        case 1:
            return Base.T
        case 2:
            return Base.C
        case 3:
            return Base.G

params = {
    "bases": [Base.A, Base.T, Base.C, Base.G],
    "tf_length": 8,
    "promoter_length": 500,
    "amount_tf": 10,
    "bm_amount_sequences": 30
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

def calc_sequence_probability(sequence: str, WM) -> float:
    """
    Calculate probability of sequence given a probability matrix
    :param sequence:
    :return: Probability
    """
    p = 1
    for pos, base in enumerate(sequence):
        p *= WM[base_to_index(base), pos]
    return p

class SegmentationModel:
    R = []
    R_prime = []
    G = []
    sequence: str = ""
    ppm: np.array

    def __init__(self, sequence: str, weight_matrices):
        self.sequence = sequence
        self.R = np.zeros(len(sequence))
        self.R_prime = np.zeros(len(sequence))
        self.G = np.zeros(len(sequence))
        self.ppm = weight_matrices



    def calc_forward(self, i: int) -> float:
        p_alpha = 0.25
        _, l = self.ppm.shape
        sum = 0
        # First i where promoter seq. of length 8 fits
        if i >= l - 1:
            p = calc_sequence_probability(self.sequence[i - l + 1: i + 1], self.ppm)
            R = np.cumprod(self.R[i - l + 1: i])[-1]
            sum = p * (1 / R)
        return p_alpha + sum

    def calc_backward(self, i: int) -> float:
        p_alpha = 0.25
        _, l = self.ppm.shape
        sum = 0
        if len(self.sequence) - l >= i >= 1:
            p = calc_sequence_probability(self.sequence[i: i + l], self.ppm)
            R_prime = np.cumprod(self.R_prime[i + 1: i + l])[-1]
            sum = p * (1 / R_prime)
        return p_alpha + sum


    def calc_G(self, i: int, l: int) -> float:
        if i >= l - 1:
            ratio = 1
            for k in range(0, i - l + 1):
                ratio *= self.R[k] / self.R_prime[k]

            denom = 1
            for k in range(i - l + 1, i + 1):
                denom *= self.R_prime[k]
            return calc_sequence_probability(self.sequence[i + 1 - l:i + 1], self.ppm) * ratio / denom
        else:
            return 0

    def run(self):
        length: int = len(self.sequence)
        for i in range(length):
            self.R[i] = self.calc_forward(i)
            rev = length - 1 - i
            self.R_prime[rev] = self.calc_backward(rev)

        for i in range(length):
            self.G[i] = self.calc_G(i, 8)

class BaumWelch:
    sequences: list[str]
    WM: np.array

    def __init__(self, sequences: list[str]):
        self.sequences = sequences
        self.WM = np.random.rand(4, params["tf_length"])

    def posterior(self, sequence: str, G: list[float]) -> list[float]:
        posterior = []
        for i in range(params["tf_length"], len(G)):
            posterior.append(calc_sequence_probability(sequence[i - params["tf_length"]: i], self.WM) * G[i])

        return posterior

    def run(self, MAX_ITERATIONS: int = 100):
        for _ in range(MAX_ITERATIONS):
            expected_counts = np.zeros_like(self.WM)

            for sequence in sequences:
                SM = SegmentationModel(sequence, self.WM)
                SM.run()
                posterior = self.posterior(sequence, SM.G)

                for i in range(params["tf_length"] - 1, len(sequence)):
                    post_prob = posterior[i - params["tf_length"] - 1]
                    start = i - (params["tf_length"] - 1)
                    subseq = sequence[start: i]
                    for pos, base in enumerate(subseq):
                        b_index = base_to_index(base)
                        expected_counts[b_index, pos] += post_prob

            expected_counts += 1e-6
            self.WM = expected_counts / expected_counts.sum(axis=0, keepdims=True)

def generate_sequence() -> (np.array, np.array):
    # Randomly generate promoter sequence
    promoter_sequence = np.random.choice(params["bases"], params["promoter_length"])

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
    return promoter_sequence, positions

if __name__ == '__main__':
    ppm, pwm = get_ppm_pwm()

    promoter_sequence, positions = generate_sequence()
    print(f"PPM : {ppm}\n"
          f">Promoter\n{promoter_sequence}\n"
          f"TF positions : {positions}")

    SM = SegmentationModel(promoter_sequence, ppm)
    SM.run()
    max = np.argmax(SM.G)

    tf_arr_0 = np.zeros((1, params["promoter_length"]))
    for (start, end) in positions:
        for i in range(start, end):
            tf_arr_0[0, i] = 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(0, params["promoter_length"]), SM.G)
    ax.plot(range(0, params["promoter_length"]), tf_arr_0.reshape(params["promoter_length"]))
    fig.show()

    sequences = []
    positions = []
    for i in range(params["bm_amount_sequences"]):
        sequence, position = generate_sequence()
        sequences.append(sequence)
        positions.append(position)

    BW = BaumWelch(sequences)
    BW.run(200)
    print(BW.WM)
    print(f'True PPM: \n{ppm}\n'
          f'Pred PPM: \n{BW.WM}')

    error = np.sum(np.sum(np.abs(ppm - BW.WM)))
    print("Error: ", error)

    SM = SegmentationModel(promoter_sequence, BW.WM)
    SM.run()
    max = np.argmax(SM.G)

    tf = ""
    index = np.argmax(BW.WM, axis=0)
    for i in index:
        tf += index_to_base(i)
    print(f'Best: {''.join(get_tf_sequence(ppm))}\n',
          f'Pred: {tf}')

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(0, params["promoter_length"]), SM.G)
    ax.plot(range(0, params["promoter_length"]), tf_arr_0.reshape(params["promoter_length"]))
    fig.show()