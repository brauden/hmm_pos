import nltk
import numpy as np
from collections import Counter
from typing import Tuple, List


# Transition matrix:
def make_ngram(token, n, pred=False) -> List:
    if len(token) == 0 and pred:
        ngrams = ['' for _ in range(n - 1)]
        return ngrams
    ngrams = zip(*[token[i:] for i in range(n)])
    ngrams = list(ngrams)
    if pred:
        ngrams = [x[1:] for x in ngrams]
        return ngrams
    return ngrams


def transition_matrix(tags) -> np.ndarray:
    tags_bigram = []
    for tag in tags:
        tags_bigram.append(make_ngram(tag, 2))
    tags_bigram_flat = [item for sublist in tags_bigram for item in sublist]
    tags_freq = Counter(tags_bigram_flat)
    tags_freq = tags_freq.most_common(len(tags_freq))

    unique_tags = np.unique([item for sublist in tags for item in sublist])
    trans_matrix = np.zeros(shape=(12, 12))

    for idx, tag in enumerate(unique_tags):
        placeholder = sorted(list(filter(lambda x: x[0][0] == tag, tags_freq)),
                             key=lambda x: x[0][1])
        placeholder = [x[1] for x in placeholder]
        try:
            trans_matrix[:, idx] = placeholder
        except ValueError:
            continue

    x = sorted(list(filter(lambda x: x[0][0] == "X", tags_freq)), key=lambda x: x[0][1])
    x = [x[1] for x in x]
    x.insert(8, 0)
    pron = sorted(list(filter(lambda x: x[0][0] == "PRON", tags_freq)), key=lambda x: x[0][1])
    pron = [x[1] for x in pron]
    pron.append(0)
    trans_matrix[:, 8] = pron
    trans_matrix[:, 11] = x
    trans_matrix = trans_matrix + 1  # smoothing
    norm = np.sum(trans_matrix, axis=1)
    trans_matrix = trans_matrix / norm[:, None]
    return trans_matrix


def map_transition(tags) -> dict:
    unique_tags = np.unique([item for sublist in tags for item in sublist])
    ind_tag_map = {k: v for k, v in enumerate(unique_tags)}
    return ind_tag_map


# Observation matrix:
def map_observation(obs_freq) -> Tuple[dict, dict]:
    unique_words = np.unique([x[0][0] for x in obs_freq])
    ind_word_map = {k: v for k, v in enumerate(unique_words)}
    word_ind_map = {k: v for v, k in ind_word_map.items()}
    return ind_word_map, word_ind_map


def observation_matrix(corpus, tags) -> np.ndarray:
    flat_corpus = [item for sublist in corpus for item in sublist]
    obs_freq = Counter(flat_corpus)
    obs_freq = obs_freq.most_common(len(obs_freq))
    unique_tags = np.unique([item for sublist in tags for item in sublist])
    unique_words = np.unique([x[0][0] for x in obs_freq])
    obs_matrix = np.zeros(shape=(len(unique_words), len(unique_tags)))
    for row, word in enumerate(unique_words):
        for col, tag in enumerate(unique_tags):
            filtered_col = list(filter(lambda x: x[0][1] == tag, obs_freq))
            filtered_row = list(filter(lambda x: x[0][0] == word, filtered_col))
            if len(filtered_row) > 0:
                obs_matrix[row, col] = filtered_row[0][1]
    unk = np.array([0.] * 12)  # Handle OOV
    obs_matrix = np.vstack([obs_matrix, unk])
    obs_matrix = obs_matrix + 1
    norm_obs = np.sum(obs_matrix, axis=0)
    obs_matrix_norm = obs_matrix / norm_obs[None, :]
    return obs_matrix_norm.T


def initial_state(corpus) -> List[float]:
    first_words = [x[0] for x in corpus]
    first_words = [x[1] for x in first_words]
    first_words_freq = Counter(first_words)
    first_words_freq = first_words_freq.most_common(len(first_words_freq))
    first_words_freq = sorted(first_words_freq, key=lambda x: x[0])
    initial_state_dist = [x[1] / sum(y[1] for y in first_words_freq) for x in first_words_freq]
    return initial_state_dist


if __name__ == '__main__':
    corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    tags = [[x[1] for x in y] for y in corpus]
    transition_m = transition_matrix(tags)
    observation_m = observation_matrix(corpus, tags)
    init_state = initial_state(corpus)
