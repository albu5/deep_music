import numpy as np


def decode_rnn(model, seed_seq, seq_len):
    decoded_seq = np.zeros(seed_seq.shape[0], seq_len)
    for i in range(seq_len):
        decoded_seq[:, i] = np.squeeze(model.predict(np.expand_dims(seed_seq, axis=0)))
        seed_seq[:, :-1] = seed_seq[:, 1:]
        seed_seq[:, -1] = decoded_seq[:, i]
    return decoded_seq
