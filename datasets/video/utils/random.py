import random


def random_bool(p: float) -> bool:
    """
    Return True with probability p
    """
    if p == 0:
        return False
    return random.random() < p
