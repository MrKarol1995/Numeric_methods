import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

f1 = lambda x: np.tan(np.pi - x) - x


def bisection2(f, a, b, error):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise(ValueError("Żle dobrany przedział. Miejsce zerowe znajduje się poza nim."))
    else:
        m = (a + b) / 2
        fm = f(m)
        num_iter = int(math.ceil(math.log(abs(a-b)/error, 2)))
        num_add_mult = 9 #f(a) = 2, f(b) = 2 f*a * fb = 1, m = 2, fm = 2
        for _ in range(num_iter):
            if fm * fb < 0:
                a, b = m, b
                fa = fm
            elif fa * fm < 0:
                a, b = a, m
                fb = fm
            m = (a + b) / 2
            fm = f(m)
            num_add_mult += 8 #sum_iter = 1, num_add_mult = 1, fm*fb = 1, fa * f, = 1, m = 2, fm = 2
            if abs(fm) < error:
                return m, fm, num_iter, num_add_mult
        return m, fm, num_iter, num_add_mult

x, fm, num1, num2 = bisection2(f1, 1.7, 2.8, 1e-8)

print(f"Rozwiązanie to: x = {x:6.4f}\nf(m) = {fm}\nLiczba iteracji: {num1}\nLiczba dodawań i mnożeń: {num2}")
