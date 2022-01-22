import numpy as np


def roulette_wheel(items, attr, number):
    sum_weight = sum([getattr(item, attr) for item in items])
    prob = [getattr(i, attr)/sum_weight for i in items]
    for _ in range(number):
        yield np.random.choice(items, p=prob)
