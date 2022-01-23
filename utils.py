import numpy as np
from copy import deepcopy


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
            current_index += 1
            current_item = items[current_index]
            weight += getattr(current_item, attr)


def q_tournament(items, attr, number, q=2):
    for tournament_number in range(number):
        chosen = []
        for i in range(q):
            chosen.append(np.random.choice(items))

        best = chosen[0]
        for item in items:
            if getattr(item, attr) > getattr(best, attr):
                best = item

        yield best


def choose_pairs(items, number):
    if number > len(items):
        raise ValueError('number must be less than len(items).')
    for i in range(0, len(items), 2):
        if i > number:
            break
        yield items[i], items[i+1]
