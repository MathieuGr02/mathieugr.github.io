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


params = {
    "bases": [Base.A, Base.T, Base.C, Base.G],
    "tf_length": 8,
    "promoter_length": 100,
    "amount_tf": 2,
    "bm_amount_sequences": 10
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
    ppm: np.array

    def __init__(self, sequence: str, weight_matrices):
        self.sequence = sequence
        self.R = np.zeros(len(sequence))
        self.R_prime = np.zeros(len(sequence))
        self.G = np.zeros(len(sequence))
        self.ppm = weight_matrices

    def calc_sequence_probability(self, sub_sequence: str) -> float:
        """
        Calculate probability of sequence given a probability matrix
        :param sub_sequence:
        :return: Probability
        """
        p = 1
        for pos, base in enumerate(sub_sequence):
            p *= self.ppm[base_to_index(base), pos]
        return p

    def calc_forward(self, i: int) -> float:
        p_alpha = 0.25
        _, l = self.ppm.shape
        sum = 0
        # First i where promoter seq. of length 8 fits
        if i >= l - 1:
            p = self.calc_sequence_probability(self.sequence[i - l + 1: i + 1])
            R = np.cumprod(self.R[i - l + 1: i])[-1]
            sum = p * (1 / R)
        return p_alpha + sum

    def calc_backward(self, i: int) -> float:
        p_alpha = 0.25
        _, l = self.ppm.shape
        sum = 0
        if len(self.sequence) - l >= i >= 1:
            p = self.calc_sequence_probability(self.sequence[i: i + l])
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
            return self.calc_sequence_probability(self.sequence[i + 1 - l:i + 1]) * ratio / denom
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
    A: np.array
    E: np.array
    forward: np.array
    backward: np.array
    P = np.array
    sequences: list[str]

    def __init__(self, sequences: list[str]):
        self.sequences = sequences
        self.A = np.random.random((4, 4))
        self.E = np.random.random((4, 4))

    def calc_sequence_probability(self, sub_sequence: str) -> float:
        """
        Calculate probability of sequence given a probability matrix
        :param sub_sequence:
        :return: Probability
        """
        p = 1
        for pos, base in enumerate(sub_sequence):
            p *= self.ppm[base_to_index(base), pos]
        return p

    def emission_prob(self, state: State | int, base: str) ->  float:
        return self.E[state, base_to_index(base)]

    def transition_prob(self, prev: State | int, next: State | int) -> float:
        return self.A[prev, next]


        return paths

    def calc_A(self, k: int, l: int):
        n, m = self.forward
        for j in range(n):
            for i in range(m):
                self.forward[k, i] * self.transition_prob(k, l) * self.emission_prob(l, self.sequences[i + 1]) * self.backward[l, i + 1]

    def calc_E(self, k: int, l: int):
        pass

    def run(self, repeats: int):
        for epoch in range(repeats):
            amount = len(self.sequences)
            length = len(self.sequences[0])
            probabilities = []
            for sequence in sequences:
                SM = SegmentationModel(sequence, self.A)
                SM.run()
            # shape = (sequence, state, base)

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

    print(
        f"{'Sequence':<8}| {promoter_sequence}\n"
        f"{'True':<8}| {'':<{positions[0][0]}}++++++++{'':>{positions[0][1]}}\n"
        f"{'Max':<8}| {'':<{max - params["tf_length"] + 1}}--------{'':>{max + 1}}"
    )

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
    BW.run(100)
    print(BW.A, BW.E)
    print(f'True PPM: \n{ppm}\n'
          f'Pred PPM: \n{BW.A}')