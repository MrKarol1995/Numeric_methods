import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy import linalg


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

def gauss_jordan_inverse(matrix):
    n = len(matrix)

    # Tworzymy macierz rozszerzoną: oryginalna macierz po lewej + macierz jednostkowa po prawej
    augmented_matrix = [matrix[i] + [float(i == j) for j in range(n)] for i in range(n)]

    # Przekształcanie macierzy do formy macierzy jednostkowej po lewej stronie
    for i in range(n):
        # Normalizujemy wiersz tak, aby element diagonalny wynosił 1
        diagonal_element = augmented_matrix[i][i]
        if diagonal_element == 0:
            raise ValueError("Macierz nie jest odwracalna.")

        for j in range(2 * n):
            augmented_matrix[i][j] /= diagonal_element

        # Zerowanie pozostałych elementów w kolumnie i
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Wyodrębniamy macierz odwrotną z rozszerzonej macierzy
    inverse_matrix = [row[n:] for row in augmented_matrix]

    return inverse_matrix

print("Zad 1")
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
n1 = 5
b = np.array([5, 4, 3, 2, 1])

# Generowanie macierzy Hilberta
A = hilbert_matrix(n1)

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

# Zad 2

print("Zad 2")
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
n2 = 20

# Generowanie macierzy A i wektora b
A = create_tridiagonal_matrix(n2)
b = np.zeros(n2)
b[-1] = 100  # Ustawienie ostatniego elementu na 100

# Rozwiązywanie układu równań metodą Gaussa-Seidla
solution = gauss_seidel(A, b)

# Wyświetlanie wyników
print("Rozwiązanie układu równań:")
print(solution)


#####################


#Zad 3

print("Zad 3")
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

def gauss_jordan_inverse(matrix):
    n = len(matrix)

    # Tworzymy macierz rozszerzoną: oryginalna macierz po lewej + macierz jednostkowa po prawej
    augmented_matrix = [matrix[i] + [float(i == j) for j in range(n)] for i in range(n)]

    # Przekształcanie macierzy do formy macierzy jednostkowej po lewej stronie
    for i in range(n):
        # Normalizujemy wiersz tak, aby element diagonalny wynosił 1
        diagonal_element = augmented_matrix[i][i]
        if diagonal_element == 0:
            raise ValueError("Macierz nie jest odwracalna.")

        for j in range(2 * n):
            augmented_matrix[i][j] /= diagonal_element

        # Zerowanie pozostałych elementów w kolumnie i
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Wyodrębniamy macierz odwrotną z rozszerzonej macierzy
    inverse_matrix = [row[n:] for row in augmented_matrix]

    return inverse_matrix

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

def create_tridiagonal_matrix(n):
    """Tworzy macierz trójdiagonalną o wymiarach n x n."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i][i] = 4 # Wartość na diagonali
        A[0][n-1] = 1
        A[n-1][0] = 1
        if i > 0:
            A[i][i - 1] = -1  # Wartość poniżej diagonali
            A[i - 1][i] = -1  # Wartość powyżej diagonali
    return A

n = 20
b = np.zeros(n)
b[-1] = 100  # Ustawienie ostatniego elementu na 100
# print("B matrix:\n",b)
# print("Matrix: \n", create_tridiagonal_matrix(n))

A = create_tridiagonal_matrix(n)

# Rozwiązywanie układu równań metodą Gaussa-Seidla
solution = gauss_seidel(A, b)

# Wyświetlanie wyników
print("Rozwiązanie układu równań:")
print(solution)

print("2 rozwiązanie: ")
print(gauss_elimination(A,b))

x = linalg.solve(A, b)
print("3 rozwiązanie: ")
print(x)

# Zad 4
print("Zad 4")


def solve_linear_system(A, b):
    """
    Rozwiązuje układ równań liniowych Ax = b przy użyciu metody dokładnej.

    Parametry:
    - A: macierz współczynników (numpy array)
    - b: wektor wyrazów wolnych (numpy array)

    Zwraca:
    - x: rozwiązanie układu równań (numpy array)
    - czas_rozwiązania: czas wykonania obliczeń w sekundach (float)
    """
    start_time = time.time()  # Rozpocznij pomiar czasu
    x = solve(A, b)
    end_time = time.time()  # Zakończ pomiar czasu
    czas_rozwiązania = end_time - start_time
    return x, czas_rozwiązania

# Zad 4
def create_matrix_B(n):
    """
    Tworzy macierz B o rozmiarze nxn z określonymi wartościami:
    - 0.025, 0.05, 0.075, ..., 0.5 na głównej diagonali
    - 5 na diagonali powyżej głównej

    Zwraca:
    - B: macierz 20x20 (numpy array)
    """
    B = np.zeros((n, n))
    # Wypełnienie głównej diagonali wartościami od 0.025 do 0.5
    diagonal_values = np.linspace(0.025, 0.5, 20)
    np.fill_diagonal(B, diagonal_values)

    # Wypełnienie diagonali powyżej głównej wartością 5
    for i in range(n-1):
        B[i, i + 1] = 5

    return B


def iterative_process(B, num_iterations=100, n=20):
    """
    Wykonuje iteracyjny proces x^(k+1) = B * x^(k) dla num_iterations kroków.
    Oblicza i zapisuje wartość eta_k = ||x^(k)||_2 / ||x^(0)||_2 dla każdego kroku.

    Parametry:
    - B: macierz iteracyjna (numpy array)
    - num_iterations: liczba iteracji (int)

    Zwraca:
    - eta_values: lista wartości eta_k dla każdego k
    - k_min: najmniejsze k, dla którego ||x^(k)||_2 < ||x^(0)||_2
    """
    # Inicjalizacja wektora x^(0) jako wektora jedynek
    x = np.ones(n)
    eta_values = []
    initial_norm = np.linalg.norm(x, 2)  # ||x^(0)||_2

    k_min = None
    for k in range(1, num_iterations + 1):
        # Obliczenie x^(k+1)
        x = B @ x
        # x = np.dot(B, x)
        # Obliczenie eta_k = ||x^(k)||_2 / ||x^(0)||_2
        current_norm = np.linalg.norm(x, 2)
        eta_k = current_norm / initial_norm
        eta_values.append(eta_k)

        # Sprawdzenie warunku, kiedy ||x^(k)||_2 < ||x^(0)||_2
        if k_min is None and eta_k < 1:
            k_min = k

    return eta_values, k_min


def plot_eta_values(eta_values):
    """
    Rysuje wykres wartości eta_k w zależności od liczby iteracji.

    Parametry:
    - eta_values: lista wartości eta_k dla każdego k
    """
    plt.plot(eta_values, marker='o')
    plt.xlabel('Iteracja k')
    plt.ylabel(r'$\eta_k = \frac{||x^{(k)}||_2}{||x^{(0)}||_2}$')
    plt.title('Wartości $\eta_k$ w zależności od liczby iteracji')
    plt.grid(True)
    plt.show()


# Przykład użycia funkcji dla Zadania 4

n = 20
# Zadanie 4: Iteracyjny proces z macierzą B
B = create_matrix_B(n)
eta_values, k_min = iterative_process(B, 100, n)
print("Najmniejsze k, dla którego ||x^(k)||_2 < ||x^(0)||_2:", k_min)

# Rysowanie wykresu eta_k
plot_eta_values(eta_values)

# print(create_matrix_B(n))