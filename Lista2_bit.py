import numpy as np
from scipy import linalg


# Zad 4
print("Zad 4")

# Macierz odwrotna
# Funkcja do obliczania macierzy odwrotnej metodą Gaussa-Jordana
def macierz_odwrotna_gauss_jordan(macierz):
    n = len(macierz)

    # Tworzymy rozszerzoną macierz: oryginalna + macierz jednostkowa
    macierz_rozszerzona = np.hstack((macierz, np.identity(n)))

    # Przekształcenie do macierzy jednostkowej
    for i in range(n):
        # Sprawdzamy, czy element główny jest zerem, jeśli tak, zamieniamy wiersze
        if macierz_rozszerzona[i][i] == 0:
            for j in range(i + 1, n):
                if macierz_rozszerzona[j][i] != 0:
                    macierz_rozszerzona[[i, j]] = macierz_rozszerzona[[j, i]]
                    break
            else:
                return "Macierz nie jest odwracalna."

        # Normalizujemy wiersz tak, aby element główny był równy 1
        macierz_rozszerzona[i] = macierz_rozszerzona[i] / macierz_rozszerzona[i][i]

        # Zerujemy pozostałe elementy w kolumnie
        for j in range(n):
            if j != i:
                macierz_rozszerzona[j] -= macierz_rozszerzona[j][i] * macierz_rozszerzona[i]

    # Ostatnie n kolumn zawierają macierz odwrotną
    macierz_odwrotna = macierz_rozszerzona[:, n:]
    return macierz_odwrotna


A4 = np.array([[0, 0, 2, 1, 2],
              [0, 1, 0, 2, -1],
              [1, 2, 0, -2, 0],
              [0, 0, 0, -1, 1],
              [0, 1, -1, 1, -1]], dtype=float)

b4 = np.array([1, 1, -4, -2, -1], dtype=float)

macierz_odwrotna = macierz_odwrotna_gauss_jordan(A4)

# Dot produkt Mnożenie macierzy
wynik4 = np.dot(macierz_odwrotna, b4)
wynik44 = np.linalg.inv(A4)
wynik444 = np.dot(wynik44, b4)
print("Macierz odwrotna metodą Gaussa-Jordana:\n", macierz_odwrotna, "\n Wynik b=A^-1 * x: \n", wynik4,"\n", wynik444)

# Zad 5
print("Zad 5")
# Zad 5: Interpolacja wielomianowa
points = np.array([[0, -1],
                   [1, 1],
                   [3, 3],
                   [5, 2],
                   [6, -2]])

x5 = points[:, 0]
y5 = points[:, 1]

# Tworzy macierz Vandermonde'a na podstawie wektora. Każdy rząd tej macierzy to kolejne potęgi dla wielomianu 4.
A5 = np.vander(x5, 5)
coef = linalg.solve(A5, y5)
odwr = macierz_odwrotna_gauss_jordan(A5)
wynn = np.dot(odwr, coef)
print("Zad 5: Współczynniki wielomianu:")
print(coef)

# Zad 6
print("Zad 6")
# Wyznacznik
# Funkcja do obliczania wyznacznika macierzy rekurencyjnie (metoda Laplace'a)
def wyznacznik(macierz):

    # Sprawdzamy, czy macierz jest kwadratowa
    n = len(macierz)
    if any(len(wiersz) != n for wiersz in macierz):
        raise ValueError("Macierz musi być kwadratowa")

    # Przypadek bazowy: jeśli macierz ma rozmiar 1x1, zwracamy jej element
    if n == 1:
        return macierz[0][0]

    # Dla większych macierzy rozwijamy wyznacznik rekurencyjnie
    det = 0
    for kolumna in range(n):
        # Obliczamy macierz dopełnień przez usunięcie wiersza 0 i kolumny 'kolumna'
        minor = [[macierz[i][j] for j in range(n) if j != kolumna] for i in range(1, n)]
        # Wyznacznik jest sumą elementów pierwszego wiersza pomnożonych przez ich dopełnienia
        det += ((-1) ** kolumna) * macierz[0][kolumna] * wyznacznik(minor)

    return det

A6 = [
    [3.50, 2.77, -0.76, 1.80],
    [-1.80, 2.68, 3.44, -0.09],
    [0.27, 5.07, 6.90, 1.61],
    [1.71, 5.45, 2.68, 1.71]
]

b6 = np.array([7.31, 4.23, 13.85, 11.55])
INV_A6 = macierz_odwrotna_gauss_jordan(A6)
wynik6 = np.dot(INV_A6, b6)
wynik66 = np.linalg.det(A6)
odpo = np.dot(A6, wynik6)
print(f" Wyznacznik: {wyznacznik(A6)}, A * x: {wynik6}", "\n komputerowy wyznacznik: ",wynik66, odpo)

# Zad 7
print("Zad 7")

A7 = [[10, -2, -1, 2, 3, 1, -4, 7],
              [5, 11, 3, 10, -3, 3, 3, -4],
              [7, 12, 1, 5, 3, -12, 2, 3],
              [8, 7, -2, 1, 3, 2, 2, 4],
              [2, -15, -1, 1, 4, -1, 8, 3],
              [4, 2, 9, 1, 12, -1, 4, 1],
              [-1, 4, -7, -1, 1, 1, -1, -3],
              [-1, 3, 4, 1, 3, -4, 7, 6]]

b7 = np.array([0, 12, -5, 3, -25, -26, 9, -7])

macierz_odwrotna7 = macierz_odwrotna_gauss_jordan(A7)
x7_scipy = linalg.solve(A7, b7)

# Dot produkt Mnożenie macierzy
wynik7 = np.dot(macierz_odwrotna7, b7)

print("Wynik b=A^-1 * x: \n", wynik7, " \n komputerowy: ", x7_scipy)

# Zad 8
print("Zad 8")

A8 = [[2, -1, 0, 0, 0, 0],
              [-1, 2, -1, 0, 0, 0],
              [0, -1, 2, -1, 0, 0],
              [0, 0, -1, 2, -1, 0],
              [0, 0, 0, -1, 2, -1],
              [0, 0, 0, 0, -1, 5]]

# Macierz wstęgowa dla 3 "przekątnych"
def is_tridiagonal(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True

wynik8 = macierz_odwrotna_gauss_jordan(A8)
wynik88 = np.linalg.inv(A8)
print("Macierz odwrotna: \n", wynik8," \n komputerowa: \n",wynik88, " \nCzy trójdiagonalna? ", is_tridiagonal(wynik8))

# Zad 9
print("Zad 9")

A9 = [
    [1, 3, -9, 6, 4],
    [2, -1, 6, 7, 1],
    [3, 2, -3, 15, 5],
    [8, -1, 1, 4, 2],
    [11, 1, -2, 18, 7]
]

deter9 = wyznacznik(A9)
wynik9 = macierz_odwrotna_gauss_jordan(A9)
wynik10 = np.linalg.det(A9)
print("Macierz odwrotna: \n", wynik9, " \n Wyznacznik własny:", deter9,"\n Wyznacznik Komputerowy: ", wynik10)
