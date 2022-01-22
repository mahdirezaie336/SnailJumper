import numpy as np


def roulette_wheel(items, attr, number):
    sum_weight = sum([getattr(item, attr) for item in items])
    prob = [getattr(i, attr)/sum_weight for i in items]
    for _ in range(number):
        yield np.random.choice(items, p=prob)


def sus(items, attr, number):
    sum_weight = sum([getattr(item, attr) for item in items])
    step = sum_weight / number
    i = np.random.random() * step
    weight = getattr(items[0], attr)
    current_index = 0
    while i < sum_weight:
        current_item = items[current_index]
        if i < weight:
            yield current_item
            i += step
        else:
            weight += getattr(current_item, attr)
            current_index += 1
