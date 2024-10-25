import numpy as np
import time


# Funkcja do generowania macierzy Hilberta
def hilbert_matrix(n):
    """Tworzy macierz Hilberta o wymiarach n x n."""
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


# Funkcja do rozwiązywania układu równań metodą iteracyjnego poprawiania
def iterative_improvement(A, b, x0=None, max_iter=1000, tol=1e-6):
    """Rozwiązuje układ równań Ax = b metodą iteracyjnego poprawiania."""
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()

    for iteration in range(max_iter):
        x_new = np.linalg.solve(A, b)
        if np.linalg.norm(x_new - x) < tol:
            print(f"Zatrzymano po {iteration} iteracjach.")
            break
        x = x_new

    return x


# Funkcja do obliczania macierzy odwrotnej metodą Gaussa-Jordana
def macierz_odwrotna_gauss_jordan(macierz):
    """Oblicza macierz odwrotną dla podanej macierzy metodą Gaussa-Jordana."""
    n = len(macierz)
    macierz_rozszerzona = np.hstack((macierz, np.identity(n)))

    for i in range(n):
        # Sprawdzamy, czy element główny jest zerem
        if macierz_rozszerzona[i][i] == 0:
            for j in range(i + 1, n):
                if macierz_rozszerzona[j][i] != 0:
                    macierz_rozszerzona[[i, j]] = macierz_rozszerzona[[j, i]]
                    break
            else:
                return "Macierz nie jest odwracalna."

        # Normalizujemy wiersz
        macierz_rozszerzona[i] = macierz_rozszerzona[i] / macierz_rozszerzona[i][i]

        # Zerujemy pozostałe elementy w kolumnie
        for j in range(n):
            if j != i:
                macierz_rozszerzona[j] -= macierz_rozszerzona[j][i] * macierz_rozszerzona[i]

    macierz_odwrotna = macierz_rozszerzona[:, n:]
    return macierz_odwrotna


# Definiowanie wymiarów i wektora b
n = 5
b = np.array([5, 4, 3, 2, 1])

# Generowanie macierzy Hilberta
A = hilbert_matrix(n)

# Rozwiązywanie układu równań metodą iteracyjnego poprawiania
start_time_iterative = time.time()
solution_iterative = iterative_improvement(A, b)
end_time_iterative = time.time()
time_iterative = end_time_iterative - start_time_iterative

# Rozwiązywanie układu równań metodą Gaussa-Jordana
start_time_gauss = time.time()
inverse_A = macierz_odwrotna_gauss_jordan(A)
if isinstance(inverse_A, str):  # Sprawdzamy, czy nie wystąpił błąd
    print(inverse_A)
else:
    solution_gauss = inverse_A @ b
end_time_gauss = time.time()
time_gauss = end_time_gauss - start_time_gauss

# Wyświetlanie wyników
print("Rozwiązanie układu równań metodą iteracyjnego poprawiania:")
print(solution_iterative)
print(f"Czas wykonania: {time_iterative:.6f} sekund")

print("\nRozwiązanie układu równań metodą Gaussa-Jordana:")
print(solution_gauss)
print(f"Czas wykonania: {time_gauss:.6f} sekund")

# Porównanie dokładności
print("\nPorównanie dokładności:")
print("Różnica między rozwiązaniami:", np.linalg.norm(solution_iterative - solution_gauss))

# Zad2



def create_tridiagonal_matrix(n):
    """Tworzy macierz trójdiagonalną o wymiarach n x n."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i][i] = 4  # Wartość na diagonali
        if i > 0:
            A[i][i - 1] = -1  # Wartość poniżej diagonali
            A[i - 1][i] = -1  # Wartość powyżej diagonali
    return A


def gauss_seidel(A, b, x0=None, max_iter=1000, tol=1e-6):
    """Rozwiązuje układ równań Ax = b metodą Gaussa-Seidla."""
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()

    for iteration in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            # Oblicz nową wartość x[i]
            sum_a = np.dot(A[i], x_new) - A[i][i] * x_new[i]
            x_new[i] = (b[i] - sum_a) / A[i][i]

        # Sprawdź kryterium zatrzymania
        if np.linalg.norm(x_new - x) < tol:
            print(f"Zatrzymano po {iteration + 1} iteracjach.")
            break

        x = x_new

    return x


# Parametr n
n = 20

# Generowanie macierzy A i wektora b
A = create_tridiagonal_matrix(n)
b = np.zeros(n)
b[-1] = 100  # Ustawienie ostatniego elementu na 100

# Rozwiązywanie układu równań metodą Gaussa-Seidla
solution = gauss_seidel(A, b)

# Wyświetlanie wyników
print("Rozwiązanie układu równań:")
print(solution)