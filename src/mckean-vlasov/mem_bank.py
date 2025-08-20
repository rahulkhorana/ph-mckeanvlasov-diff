# mem_bank.py
import numpy as np

K_BANK = 256  # device slice
Q_BANK = 8192  # host RAM; ~8 MB if D=256


def init_cpu_bank(Q, D):
    return {"buf": np.zeros((Q, D), np.float32), "head": 0, "count": 0, "Q": Q, "D": D}


def bank_enqueue(bank, x_np):  # x_np: (B,D) np.float32
    Q, head, cnt = bank["Q"], bank["head"], bank["count"]
    B = x_np.shape[0]
    end = head + B
    if end <= Q:
        bank["buf"][head:end] = x_np
    else:
        r = end - Q
        bank["buf"][head:] = x_np[: Q - head]
        bank["buf"][:r] = x_np[Q - head :]
    bank["head"] = (head + B) % Q
    bank["count"] = min(Q, cnt + B)


def bank_sample(bank, K):
    cnt = bank["count"]
    if cnt == 0:
        return np.zeros((K, bank["D"]), np.float32), np.zeros((K,), np.bool_)
    take = min(K, cnt)
    idx = np.random.choice(cnt, size=take, replace=False)
    negs = bank["buf"][:cnt][idx]
    if take < K:
        pad = K - take
        negs = np.concatenate([negs, np.zeros((pad, bank["D"]), np.float32)], 0)
        mask = np.concatenate(
            [np.ones((take,), np.bool_), np.zeros((pad,), np.bool_)], 0
        )
    else:
        mask = np.ones((K,), np.bool_)
    return negs, mask
