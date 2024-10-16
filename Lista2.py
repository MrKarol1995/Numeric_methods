import numpy as np
from scipy import linalg

# Zad 4
def gauss_elimination(A, b):
    n = len(b)

    # Tworzenie rozszerzonej macierzy układu [A | b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Eliminacja Gaussa
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        if Ab[i, i] == 0:
            raise ValueError("Macierz jest osobliwa (nieodwracalna)!")

        Ab[i] = Ab[i] / Ab[i, i]

        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]

    # Back substitution (wyznaczanie rozwiązania)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1:n] * x[i + 1:n])

    return x


# Zad 4: Porównanie wyników z scipy.linalg.solve
A4 = np.array([[0, 0, 2, 1, 2],
              [0, 1, 0, 2, -1],
              [1, 2, 0, -2, 0],
              [0, 0, 0, -1, 1],
              [0, 1, -1, 1, -1]], dtype=float)

b4 = np.array([1, 1, -4, -2, -1], dtype=float)

x_gauss = gauss_elimination(A4, b4)
x_scipy = linalg.solve(A4, b4)

print("Zad 4: Rozwiązanie eliminacją Gaussa:")
print(x_gauss)
print("Zad 4: Rozwiązanie scipy.linalg.solve:")
print(x_scipy)

# Zad 5: Interpolacja wielomianowa
points = np.array([[0, -1],
                   [1, 1],
                   [3, 3],
                   [5, 2],
                   [6, -2]])

x5 = points[:, 0]
y5 = points[:, 1]

A5 = np.vander(x5, 5)
coefficients_custom = np.linalg.solve(A5, y5)
coefficients_scipy = linalg.solve(A5, y5)

print("\nZad 5: Współczynniki wielomianu (custom):")
print(coefficients_custom)
print("Zad 5: Współczynniki wielomianu (scipy.linalg.solve):")
print(coefficients_scipy)

# Zad 6: Układ równań
A6 = np.array([[3.50, 2.77, -0.76, 1.80],
              [-1.80, 2.68, 3.44, -0.09],
              [0.27, 5.07, 6.90, 1.61],
              [1.71, 5.45, 2.68, 1.71]])

b6 = np.array([7.31, 4.23, 13.85, 11.55])

x6_custom = np.linalg.solve(A6, b6)
x6_scipy = linalg.solve(A6, b6)
det_A6 = np.linalg.det(A6)
Ax6 = np.dot(A6, x6_custom)

print("\nZad 6: Rozwiązanie układu (numpy.linalg.solve):")
print(x6_custom)
print("Zad 6: Rozwiązanie układu (scipy.linalg.solve):")
print(x6_scipy)
print("\nWyznacznik macierzy A (numpy.linalg.det):")
print(det_A6)
print("\nIloczyn A * x (taki jak b_hat):")
print(Ax6)

# Zad 7: Macierz 8x8
A7 = np.array([[10, -2, -1, 2, 3, 1, -4, 7],
              [5, 11, 3, 10, -3, 3, 3, -4],
              [7, 12, 1, 5, 3, -12, 2, 3],
              [8, 7, -2, 1, 3, 2, 2, 4],
              [2, -15, -1, 1, 4, -1, 8, 3],
              [4, 2, 9, 1, 12, -1, 4, 1],
              [-1, 4, -7, -1, 1, 1, -1, -3],
              [-1, 3, 4, 1, 3, -4, 7, 6]])

b7 = np.array([0, 12, -5, 3, -25, -26, 9, -7])

x7_custom = np.linalg.solve(A7, b7)
x7_scipy = linalg.solve(A7, b7)

print("\nZad 7: Rozwiązanie układu (numpy.linalg.solve):")
print(x7_custom)
print("Zad 7: Rozwiązanie układu (scipy.linalg.solve):")
print(x7_scipy)

# Zad 8: Macierz trójdiagonalna
A8 = np.array([[2, -1, 0, 0, 0, 0],
              [-1, 2, -1, 0, 0, 0],
              [0, -1, 2, -1, 0, 0],
              [0, 0, -1, 2, -1, 0],
              [0, 0, 0, -1, 2, -1],
              [0, 0, 0, 0, -1, 5]])

A8_inv = np.linalg.inv(A8)

def is_tridiagonal(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True

print("\nZad 8: Macierz odwrotna A^-1:")
print(A8_inv)
print("\nCzy macierz A^-1 jest trójdiagonalna?")
print(is_tridiagonal(A8_inv))

# Eliminacja Gaussian

import numpy as np


# Funkcja do wykonania eliminacji Gaussa
def gauss_elimination(A, b):
    n = len(b)

    # Łączenie macierzy A i wektora b w macierz rozszerzoną
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Eliminacja Gaussa (przekształcenie do postaci schodkowej)
    for i in range(n):
        # Szukamy największego elementu w kolumnie (dla stabilności numerycznej)
        max_row = np.argmax(abs(Ab[i:, i])) + i
        # Zamieniamy wiersze, jeśli największy element nie jest w wierszu i
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # Przekształcamy wiersze poniżej, aby wyzerować elementy w kolumnie i
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Podstawianie wsteczne
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x


# Przykład użycia
# Definiujemy macierz A i wektor b
A = np.array([[0, 0, 2, 1.0, 2.0],
              [0, 1, 0, 2.0, -1],
              [1, 2.0, 0, -2, 0],
              [0, 0, 0, -1, 1],
              [0, 1, -1, 1, -1]])

bb = np.array([1.0, 1.0, -4.0, -2, -1])

# Rozwiązanie układu równań A * x = b
x = gauss_elimination(A, bb)

# Wyświetlenie rozwiązania
print("Rozwiązanie układu równań (x) Eliminacją Gaussa z Zad 4:")
print(x)
