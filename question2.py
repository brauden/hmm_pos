import numpy as np
from typing import List


def viterbi(obs: List[int], pi: List[float], A: np.ndarray, B: np.ndarray) -> List[int]:
    """
    Viterbi algorithm for part of speech tagging.
    :param obs: observations. List of tokens mapped with integers.
    :param pi: initial state probabilities.
    :param A: transition probability matrix
    :param B: observation probability matrix
    :return: list of integers with part of speech. Use map_transition() to get actual POS names.
    """
    pi_log = np.log(pi)
    A_log = np.log(A)
    B_log = np.log(B)
    states = A.shape[0]
    n = len(obs)

    D_log = np.zeros((states, n))
    backtrack = np.zeros((states, n - 1)).astype(int)
    D_log[:, 0] = pi_log + B_log[:, obs[0]]

    for j in range(1, n):
        for i in range(states):
            temp_sum = A_log[:, i] + D_log[:, j - 1]
            D_log[i, j] = np.max(temp_sum) + B_log[i, obs[j]]
            backtrack[i, j - 1] = np.argmax(temp_sum)

    state = np.zeros(n).astype(int)
    state[-1] = np.argmax(D_log[:, -1])
    for n in range(n - 2, -1, -1):
        state[n] = backtrack[int(state[n + 1]), n]
    state = state.tolist()
    return state
