from scipy.stats import entropy
import numpy as np


class distance:

    def jsd(p, q):
        numerator1 = p
        numerator2 = q
        denominator = 0.5*(p+q)
        dist = 0.5*entropy(numerator1, denominator)+0.5*entropy(numerator2, denominator)
        return dist

    def wassestein(p, q):
        return np.linalg.norm((p-q))