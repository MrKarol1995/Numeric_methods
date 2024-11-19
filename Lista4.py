import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, fsolve

# Zad 1
print("Zad 1")


# Definicja funkcji
def f(x):
    return np.tan(np.pi - x) - x

# Zakres wartości x (z ostrożnością, aby uniknąć asymptot w punkcie x = pi)
x = np.linspace(-1.5, 1.5, 1000)
x = x[np.abs(np.mod(x + np.pi, 2 * np.pi) - np.pi) > 0.1]  # Usuwanie wartości zbliżających się do pi

# Obliczanie wartości funkcji f(x)
y = f(x)

# Rysowanie wykresu
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = \tan(\pi - x) - x$", color='blue')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.title(r"Wykres funkcji $f(x) = \tan(\pi - x) - x$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()




# Zad 2
print("Zad 2")

# Definicja funkcji f(x) oraz jej pochodnej f'(x)
def f(x):
    return np.cosh(x) * np.cos(x) - 1

def f_prime(x):
    return np.sinh(x) * np.cos(x) - np.cosh(x) * np.sin(x)

# Wykres funkcji w przedziale [4, 8]
x = np.linspace(4, 8, 1000)
y = f(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = \cosh(x)\cos(x) - 1$", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--", label="y=0")
plt.title("Wykres funkcji f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()

# Implementacja własnej metody Newton-Raphson
def newton_method(func, func_prime, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        f_x = func(x)
        f_prime_x = func_prime(x)
        if abs(f_prime_x) < 1e-10:
            raise ValueError("Pochodna bliska zeru, metoda Newtona niestabilna.")
        x_next = x - f_x / f_prime_x
        if abs(x_next - x) < tol:
            return x_next
        x = x_next
    raise ValueError("Metoda Newtona nie osiągnęła zbieżności.")

# Znalezienie miejsc zerowych metodą własną
x0_manual = 4  # Punkt startowy
root_manual = newton_method(f, f_prime, x0_manual)

# Znalezienie miejsc zerowych metodą wbudowaną (scipy)
x0_scipy = 4  # Punkt startowy
root_scipy = newton(f, x0_scipy, f_prime)

# Wypisywanie wyników
print("========== Wyniki ==========")
if root_manual is not None:
    print(f"Metoda własna: Pierwiastek w x = {root_manual:.6f}, f(x) = {f(root_manual):.6e}")
else:
    print("Metoda własna: Brak zbieżności.")

print(f"Metoda scipy: Pierwiastek w x = {root_scipy:.6f}, f(x) = {f(root_scipy):.6e}")

# Zad 3
print("Zad 3")


# Definicja funkcji prędkości v(t) oraz równania, które rozwiązujemy
def velocity(t, u=2510, M0=2.8e6, m_dot=13.3e3, g=9.81):
    return u * np.log(M0 / (M0 - m_dot * t)) - g * t

def velocity_eq(t, target_velocity=335, u=2510, M0=2.8e6, m_dot=13.3e3, g=9.81):
    return velocity(t, u, M0, m_dot, g) - target_velocity

def velocity_eq_prime(t, u=2510, M0=2.8e6, m_dot=13.3e3, g=9.81):
    denominator = M0 - m_dot * t
    if denominator <= 0:  # Unikamy dzielenia przez zero lub ujemnych wartości masy
        return np.inf
    return -u * m_dot / denominator - g

# Metoda bisekcji
def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("Funkcja musi zmieniać znak na końcach przedziału.")
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
    raise ValueError("Metoda bisekcji nie osiągnęła zbieżności.")


# Parametry problemu
target_velocity = 335  # prędkość dźwięku w m/s
u = 2510  # prędkość spalin
M0 = 2.8e6  # masa początkowa rakiety
m_dot = 13.3e3  # szybkość zużycia paliwa
g = 9.81  # przyspieszenie ziemskie

# Wyznaczenie czasu metodą bisekcji
a, b = 0, 150  # Przedział początkowy (zakładamy, że czas osiągnięcia prędkości dźwięku nie przekroczy 100 s)
t_bisection = bisection_method(lambda t: velocity_eq(t, target_velocity, u, M0, m_dot, g), a, b)

# Wyznaczenie czasu metodą Newton-Raphson
x0 = 1  # Punkt startowy
t_newton = newton_method(lambda t: velocity_eq(t, target_velocity, u, M0, m_dot, g),
                         lambda t: velocity_eq_prime(t, u, M0, m_dot, g), x0)

# Generowanie wykresu trajektorii
t_vals = np.linspace(0, 100, 1000)
v_vals = velocity(t_vals, u, M0, m_dot, g)

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(t_vals, v_vals, label="Prędkość $v(t)$ rakiety", color="blue")
plt.axhline(target_velocity, color="red", linestyle="--", label=f"Prędkość dźwięku ({target_velocity} m/s)")
plt.scatter([t_bisection], [target_velocity], color="green", label=f"Punkt osiągnięcia prędkości dźwięku (t ≈ {t_bisection:.2f} s)", zorder=5)
plt.title("Trajektoria prędkości rakiety Saturn V")
plt.xlabel("Czas [s]")
plt.ylabel("Prędkość [m/s]")
plt.legend()
plt.grid()
#plt.show()

# Wyniki
print("========== Wyniki ==========")
print(f"Czas osiągnięcia prędkości dźwięku metodą bisekcji: {t_bisection:.6f} s")
print(f"Czas osiągnięcia prędkości dźwięku metodą Newton-Raphson: {t_newton:.6f} s")
print(f"Prędkość w t_bisection: {velocity(t_bisection, u, M0, m_dot, g):.6f} m/s")
print(f"Prędkość w t_newton: {velocity(t_newton, u, M0, m_dot, g):.6f} m/s")

# Zad 4 ?
print("Zad 4")


# Definicja funkcji G(T) i jej pochodnej
def gibbs_energy(T, R=8.31441, T0=4.44418):
    return -R * T * np.log((T / T0) ** (5 / 2))

def gibbs_energy_eq(T, target_G=-1e5, R=8.31441, T0=4.44418):
    return gibbs_energy(T, R, T0) - target_G

def gibbs_energy_prime(T, R=8.31441, T0=4.44418):
    if T <= 0:
        raise ValueError("Temperatura musi być większa od zera.")
    return -R * np.log((T / T0) ** (5 / 2)) - (5 / 2) * R

# Metoda Newton-Raphson (implementacja własna)

# Parametry problemu
R = 8.31441  # Stała gazowa w J/(mol*K)
T0 = 4.44418  # Stała T0 w K
target_G = -1e5  # Docelowa energia swobodna Gibbsa w J
x0 = 300  # Punkt startowy dla metody Newtona

# Wyznaczenie temperatury metodą Newton-Raphson (własna implementacja)
T_solution_own = newton_method(lambda T: gibbs_energy_eq(T, target_G, R, T0),
                               lambda T: gibbs_energy_prime(T, R, T0),
                               x0)

# Wyznaczenie temperatury metodą Newton-Raphson z scipy
T_solution_scipy = newton(lambda T: gibbs_energy_eq(T, target_G, R, T0),
                          x0, fprime=lambda T: gibbs_energy_prime(T, R, T0))

# Wyniki
print("========== Wyniki ==========")
print(f"Temperatura (metoda własna): T = {T_solution_own:.6f} K")
print(f"Wartość G w tej temperaturze: G = {gibbs_energy(T_solution_own, R, T0):.6f} J")
print(f"Temperatura (scipy.optimize.newton): T = {T_solution_scipy:.6f} K")
print(f"Wartość G w tej temperaturze: G = {gibbs_energy(T_solution_scipy, R, T0):.6f} J")

# Zad 5
print("Zad 5")

# Definicja układu równań
def equations(vars):
    x, y = vars
    eq1 = np.tan(x) - y - 1
    eq2 = np.cos(x) - 3 * np.sin(y)
    return [eq1, eq2]


# Definicja własnej metody Newtona dla układu równań
def newton_system(equations, jacobian, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        f_val = np.array(equations(x))
        jacobian_matrix = np.array(jacobian(x))
        if np.linalg.det(jacobian_matrix) == 0:
            raise ValueError("Macierz Jacobiego jest osobliwa.")
        delta = np.linalg.solve(jacobian_matrix, -f_val)
        x += delta
        if np.linalg.norm(delta) < tol:
            return x
    raise ValueError("Metoda Newtona nie osiągnęła zbieżności.")


# Definicja Jacobianu dla układu równań
def jacobian(vars):
    x, y = vars
    j11 = 1 / (np.cos(x) ** 2)  # Pochodna eq1 względem x
    j12 = -1  # Pochodna eq1 względem y
    j21 = -np.sin(x)  # Pochodna eq2 względem x
    j22 = -3 * np.cos(y)  # Pochodna eq2 względem y
    return [[j11, j12], [j21, j22]]


# Szukanie rozwiązań w przedziale (0, 1.5)
x_start_points = np.linspace(0.01, 1.5, 5)  # Punkty początkowe dla x
y_start_points = np.linspace(0.01, 1.5, 5)  # Punkty początkowe dla y

results_custom = []
results_scipy = []

for x0 in x_start_points:  # Iteracja po początkowych wartościach dla zmiennej x
    for y0 in y_start_points:  # Iteracja po początkowych wartościach dla zmiennej y
        try:  # Próba wykonania kodu, aby obsłużyć potencjalne błędy podczas iteracji
            # Metoda własna
            solution_custom = newton_system(  # Wywołanie własnej implementacji metody Newtona
                lambda vars: equations(vars),  # Funkcja układu równań jako argument
                lambda vars: jacobian(vars),  # Funkcja Jacobiego jako argument
                [x0, y0]  # Punkt początkowy w postaci [x0, y0]
            )
            # Sprawdzenie, czy znalezione rozwiązanie mieści się w przedziale (0, 1.5) dla obu zmiennych
            if 0 < solution_custom[0] < 1.5 and 0 < solution_custom[1] < 1.5:
                # Dodanie rozwiązania (zaokrąglonego do 6 miejsc po przecinku) do wyników metody własnej
                results_custom.append(tuple(np.round(solution_custom, 6)))

            # Metoda scipy
            solution_scipy = fsolve(  # Wywołanie wbudowanej funkcji fsolve do rozwiązania układu równań
                equations,  # Funkcja układu równań jako argument
                [x0, y0]  # Punkt początkowy w postaci [x0, y0]
            )
            # Sprawdzenie, czy znalezione rozwiązanie mieści się w przedziale (0, 1.5) dla obu zmiennych
            if 0 < solution_scipy[0] < 1.5 and 0 < solution_scipy[1] < 1.5:
                # Dodanie rozwiązania (zaokrąglonego do 6 miejsc po przecinku) do wyników metody scipy
                results_scipy.append(tuple(np.round(solution_scipy, 6)))
        except Exception as e:  # Jeśli wystąpił błąd (np. brak zbieżności metody Newtona)
            continue  # Ignorowanie błędu i przejście do kolejnego punktu startowego


# Usunięcie duplikatów i sortowanie wyników
results_custom = sorted(set(results_custom))
results_scipy = sorted(set(results_scipy))

# Wyniki
print("========== Wyniki metody własnej ==========")
for sol in results_custom:
    print(f"x = {sol[0]:.6f}, y = {sol[1]:.6f}")

print("\n========== Wyniki metody scipy ==========")
for sol in results_scipy:
    print(f"x = {sol[0]:.6f}, y = {sol[1]:.6f}")

# Zad 6 ?
print("Zad 6")


import numpy as np

# Definicja funkcji wielomianu
def polynomial(x, coefficients):
    """Oblicza wartość wielomianu dla zadanego x i współczynników."""
    return sum(c * x**i for i, c in enumerate(reversed(coefficients)))

# Definicja pochodnej wielomianu
def polynomial_derivative(x, coefficients):
    """Oblicza wartość pochodnej wielomianu dla zadanego x i współczynników."""
    derivative_coefficients = [
        i * c for i, c in enumerate(reversed(coefficients)) if i > 0
    ]
    return sum(c * x**(i - 1) for i, c in enumerate(derivative_coefficients, start=1))

# Metoda Newtona do znajdowania pierwiastków
def newton_method1(f, f_prime, x0, coefficients, tol=1e-6, max_iter=100):
    """Znajduje pierwiastek funkcji f(x) za pomocą metody Newtona"""
    x = x0
    for _ in range(max_iter):
        fx = f(x, coefficients)
        fx_prime = f_prime(x, coefficients)
        if abs(fx) < tol:
            return x
        x = x - fx / fx_prime
    raise ValueError("Metoda Newtona nie zbiega się")

# Funkcja do znajdowania wszystkich pierwiastków wielomianu
def find_roots(coefficients, initial_guesses=None):
    """
    Znajduje pierwiastki wielomianu za pomocą metody Newtona oraz funkcji numpy.roots.

    Parameters:
        coefficients (list): Współczynniki wielomianu (od najwyższej do najniższej potęgi).
        initial_guesses (list): Lista punktów początkowych dla metody Newtona (opcjonalnie).

    Returns:
        tuple: Dwie listy - pierwiastki uzyskane metodą Newtona i metodą numpy.roots.
    """
    if initial_guesses is None:
        # Rozszerzona lista punktów początkowych w różnych częściach płaszczyzny zespolonej
        initial_guesses = [
            1 + 1j, -1 - 1j, 2 + 2j, -2 - 2j, 0 + 1j, 0 - 1j,
            0.5 + 1.5j, -1 + 1j, -1.5 - 0.5j, 0.5 - 0.5j
        ]

    roots_custom = []
    for guess in initial_guesses:
        try:
            root = newton_method1(polynomial, polynomial_derivative, guess, coefficients)
            # Dodanie pierwiastka, jeśli nie został już znaleziony
            if not any(np.isclose(root, found_root, atol=1e-6) for found_root in roots_custom):
                roots_custom.append(root)
        except Exception:
            continue  # Jeśli wystąpił błąd, przechodzimy do kolejnego punktu startowego

    # Pierwiastki uzyskane metodą numpy.roots
    roots_numpy = np.roots(coefficients)

    return roots_custom, roots_numpy

# Przykładowe współczynniki wielomianu (od najwyższej potęgi do najniższej)
coefficients = [1, 5 + 1j, -(8 - 5j), 30 - 14j, -84]

# Znajdowanie pierwiastków
roots_custom, roots_numpy = find_roots(coefficients)

# Wyświetlanie wyników
print("========== Wyniki metody Newtona ==========")
for root in roots_custom:
    print(f"Pierwiastek: {root:.6f}")

print("\n========== Wyniki metody numpy.roots ==========")
for root in roots_numpy:
    print(f"Pierwiastek: {root:.6f}")
