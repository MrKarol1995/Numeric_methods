import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, fsolve, brentq

# Zad 1
print("Zad 1")

# Funkcja testowa: f(x) = tan(pi - x) - x
def f(x):
    return np.tan(np.pi - x) - x

# Pochodna funkcji do metody Newtona
def df(x):
    return -1 - (1 / np.cos(np.pi - x))**2


### METODA BRENT'A ###
def brent_method(f, a, b, tol=1e-9, max_iter=100):
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("Funkcja nie zmienia znaku w podanym przedziale [a, b].")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = e = b - a  # Początkowe odległości

    iter_count = 0
    multiplications = 0
    additions = 0

    for iteration in range(max_iter):
        iter_count += 1

        if fb != fc and fa != fc:
            # Interpolacja kwadratowa
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
            multiplications += 9  # Liczymy mnożenia
            additions += 6        # Liczymy dodawania
        else:
            # Interpolacja liniowa
            s = b - fb * (b - a) / (fb - fa)
            multiplications += 3
            additions += 2

        # Warunki ograniczające dla "s"
        if (s < (3 * a + b) / 4 or s > b):
            s = (a + b) / 2
            additions += 1

        fs = f(s)
        multiplications += 1  # Obliczenie fs wymaga jednego mnożenia

        # Aktualizacja punktów
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        # Przełączanie, aby utrzymać "b" jako lepsze przybliżenie
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(fb) < tol or abs(b - a) < tol:
            return b, iter_count, multiplications, additions

    raise RuntimeError("Metoda Brenta nie osiągnęła zbieżności w maksymalnej liczbie iteracji.")


### METODA SIECZNYCH ###
def secant_method(f, x0, x1, tol=1e-9, max_iter=100):
    f0 = f(x0)
    f1 = f(x1)

    iter_count = 0
    multiplications = 0
    additions = 0

    for _ in range(max_iter):
        iter_count += 1

        if f1 - f0 == 0:
            raise ZeroDivisionError("Dzielenie przez zero w metodzie siecznych.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        multiplications += 2
        additions += 2

        f2 = f(x2)
        multiplications += 1

        if abs(f2) < tol or abs(x2 - x1) < tol:
            return x2, iter_count, multiplications, additions

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    raise RuntimeError("Metoda siecznych nie osiągnęła zbieżności w maksymalnej liczbie iteracji.")


### METODA NEWTONA ###
def newton_method(f, df, x0, tol=1e-9, max_iter=100):
    iter_count = 0
    multiplications = 0
    additions = 0

    for _ in range(max_iter):
        iter_count += 1

        fx = f(x0)
        dfx = df(x0)
        multiplications += 1  # Pochodna wymaga obliczenia (1/cos)^2
        additions += 1

        if abs(fx) < tol:
            return x0, iter_count, multiplications, additions

        if dfx == 0:
            raise ZeroDivisionError("Pochodna zerowa - metoda Newtona nie może kontynuować.")

        x1 = x0 - fx / dfx
        multiplications += 1
        additions += 1

        if abs(x1 - x0) < tol:
            return x1, iter_count, multiplications, additions

        x0 = x1

    raise RuntimeError("Metoda Newtona nie osiągnęła zbieżności w maksymalnej liczbie iteracji.")


### METODA RÓWNEGO PODZIAŁU ###
def equal_interval_division(f, x_start, x_end, num_intervals, tol=1e-9):
    step = (x_end - x_start) / num_intervals
    iter_count = 0
    multiplications = 0
    additions = 0

    for i in range(num_intervals):
        iter_count += 1
        x1 = x_start + i * step
        x2 = x1 + step
        additions += 2

        f1 = f(x1)
        f2 = f(x2)
        multiplications += 2

        if f1 * f2 <= 0:
            return refine_interval(f, x1, x2, tol), iter_count, multiplications, additions

    return None, iter_count, multiplications, additions


def refine_interval(f, x1, x2, tol):
    while abs(x2 - x1) > tol:
        x_mid = (x1 + x2) / 2
        f_mid = f(x_mid)

        if f(x1) * f_mid <= 0:
            x2 = x_mid
        else:
            x1 = x_mid

    return (x1 + x2) / 2


# TESTY
x_start = 0.1
x_end = 2
# Wypisywanie wyników
print("========== Wyniki ==========")

# Brent
root_b, it_b, mul_b, add_b = brent_method(f, x_start, x_end)
print(f"Metoda Brenta: root = {root_b}, f(x) = {f(root_b):.6e}, iteracje = {it_b}, mnożenia = {mul_b}, dodawania = {add_b}")

# Sieczne
root_s, it_s, mul_s, add_s = secant_method(f, x_start, x_end)
print(f"Metoda Siecznych: root = {root_s}, f(x) = {f(root_s):.6e}, iteracje = {it_s}, mnożenia = {mul_s}, dodawania = {add_s}")

# Newton
root_n, it_n, mul_n, add_n = newton_method(f, df, x_end)
print(f"Metoda Newtona: root = {root_n}, f(x) = {f(root_n):.6e}, iteracje = {it_n}, mnożenia = {mul_n}, dodawania = {add_n}")

# Równy Podział
root_e, it_e, mul_e, add_e = equal_interval_division(f, x_start, x_end, 100)
print(f"Metoda Równego Podziału: root = {root_e}, f(x) = {f(root_e):.6e}, iteracje = {it_e}, mnożenia = {mul_e}, dodawania = {add_e}")

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

# Implementacja własnej metody Newton-Raphson (Stycznych)
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


# Znalezienie miejsc zerowych metodą własną Newtona
x0_manual = 4  # Punkt startowy
root_manual = newton_method(f, f_prime, x0_manual)

# Znalezienie miejsc zerowych metodą wbudowaną Newtona (scipy)
x0_scipy = 4  # Punkt startowy
root_scipy = newton(f, x0_scipy, f_prime)


# Wypisywanie wyników
print("========== Wyniki ==========")
if root_manual is not None:
    print(f"Metoda własna (Newton): Pierwiastek w x = {root_manual:.6f}, f(x) = {f(root_manual):.6e}")
else:
    print("Metoda własna (Newton): Brak zbieżności.")

if root_scipy is not None:
    print(f"Metoda scipy (Newton): Pierwiastek w x = {root_scipy:.6f}, f(x) = {f(root_scipy):.6e}")
else:
    print("Metoda scipy (Newton): Brak zbieżności.")
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

# Zad 4
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
x_start_points = np.linspace(0, 1.5, 5)  # Punkty początkowe dla x
y_start_points = np.linspace(0, 1.5, 5)  # Punkty początkowe dla y

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
    print(f"x = {sol[0]:.6f}, y = {sol[1]:.6f}", "\n")

# Zad 6
print("Zad 6")
# Definicja funkcji wielomianu
def polynomial(x, coefficients):
    """Oblicza wartość wielomianu dla zadanego x i współczynników."""
    return sum(c * x ** i for i, c in enumerate(reversed(coefficients)))


# Definicja pochodnej wielomianu
def polynomial_derivative(x, coefficients):
    """Oblicza wartość pochodnej wielomianu dla zadanego x i współczynników."""
    derivative_coefficients = [
        i * c for i, c in enumerate(reversed(coefficients)) if i > 0
    ]
    return sum(c * x ** (i - 1) for i, c in enumerate(derivative_coefficients, start=1))


# Metoda Newtona do znajdowania pierwiastków
def newton_method1(f, f_prime, x0, coefficients, tol=1e-6, max_iter=100):
    """Znajduje pierwiastek funkcji f(x) za pomocą metody Newtona."""
    x = x0
    for _ in range(max_iter):
        fx = f(x, coefficients)
        fx_prime = f_prime(x, coefficients)

        if abs(fx) < tol:  # Jeśli wartość funkcji jest bliska zeru, znaleźliśmy pierwiastek
            return x

        if abs(fx_prime) < 1e-10:  # Jeśli pochodna jest zbyt mała, przerywamy
            print(f"Uwaga: Pochodna jest zbyt mała w punkcie x = {x}.")
            break

        x = x - fx / fx_prime  # Zastosowanie wzoru Newtona

    raise ValueError("Metoda Newtona nie zbiega się")


# Dzielimy wielomian przez (x - r)
def synthetic_division(coefficients, root):
    """Dzieli wielomian przez (x - root) za pomocą dzielenia syntetycznego."""
    new_coeffs = [coefficients[0]]
    for i in range(1, len(coefficients)):
        new_coeffs.append(coefficients[i] + new_coeffs[-1] * root)

    # Ostatni element to reszta, powinna być bliska 0 dla pierwiastka
    remainder = new_coeffs[-1]
    new_coeffs = new_coeffs[:-1]  # Usuwamy ostatni element, ponieważ to reszta
    return new_coeffs, remainder


# Metoda do znajdowania wszystkich pierwiastków wielomianu
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
            2 + 0j, 0 + 2j  # Przykładowe pierwiastki, które chcesz znaleźć
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

    # Teraz dzielimy wielomian przez (x - r) dla każdego znalezionego pierwiastka
    remaining_coeffs = coefficients
    for root in roots_custom:
        remaining_coeffs, remainder = synthetic_division(remaining_coeffs, root)

    # Pierwiastki uzyskane metodą numpy.roots
    roots_numpy = np.roots(coefficients)

    return roots_custom, roots_numpy, remaining_coeffs

def process_coefficients(coefficients, threshold=1e-5, decimal_places=1):
    """
    Funkcja do zaokrąglania współczynników wielomianu i traktowania wartości mniejszych
    niż threshold jako zero.

    Parameters:
        coefficients (list): Lista współczynników wielomianu.
        threshold (float): Próg, poniżej którego wartości będą traktowane jako zero.
        decimal_places (int): Liczba miejsc po przecinku do zaokrąglenia wartości.

    Returns:
        list: Nowa lista współczynników po przetworzeniu.
    """
    processed_coeffs = []
    for coeff in coefficients:
        # Jeżeli wartość jest mniejsza niż threshold, traktujemy ją jako zero
        if abs(coeff) < threshold:
            processed_coeffs.append(0)
        else:
            # Jeśli współczynnik jest liczbą zespoloną
            if isinstance(coeff, complex):
                real_part = round(coeff.real, decimal_places)
                imag_part = round(coeff.imag, decimal_places)
                processed_coeffs.append(complex(real_part, imag_part))
            else:
                # Zaokrąglamy liczbę rzeczywistą
                processed_coeffs.append(round(coeff, decimal_places))

    return processed_coeffs

# Przykładowe współczynniki wielomianu (od najwyższej do najniższej potęgi)
coefficients = [1, 5 + 1j, -(8 - 5j), 30 - 14j, -84]

# Znajdowanie pierwiastków
roots_custom, roots_numpy, remaining_coeffs = find_roots(coefficients)



# Współczynniki wielomianu
coefficients1 = [1, (7 + 3j), 21j]

# Pierwiastek metodą Newtona
root_newton_1 = newton_method1(polynomial, polynomial_derivative, 0 + 1j, coefficients1)
root_newton_2 = newton_method1(polynomial, polynomial_derivative, -1 + 0j, coefficients1)

# Znajdowanie pierwiastków
roots_custom1, roots_numpy1, remaining_coeffs1 = find_roots(coefficients1)

# Przykładowe dane wejściowe
remaining_coeffs1 = [1, (7.000000000001375 - 5.355715870791755e-13j)]

# Przetwarzanie współczynników
processed_coeffs = process_coefficients(remaining_coeffs1)



roots_custom3, roots_numpy3, remaining_coeffs3 = find_roots(processed_coeffs)


# Wyświetlanie wyników
print("\n========== Pierwiastki metodą Newtona ==========")
for rooti in roots_custom3:
    print(f"Pierwiastek1: {rooti:.6f}")

for root1 in roots_custom1:
    print(f"Pierwiastek1: {root1:.6f}")

for root in roots_custom:
    print(f"Pierwiastek: {root:.6f}")

print("\n========== Wyniki metody numpy.roots ==========")
for root in roots_numpy:
    print(f"Pierwiastek: {root:.6f}")