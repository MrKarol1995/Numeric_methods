import numpy as np
import matplotlib.pyplot as plt
import scipy
import math, cmath
from scipy.optimize import fsolve, newton, brentq

# Zad 1
print("Zad 1")


# Funkcja testowa: f(x) = tan(pi - x) - x
f1 = lambda x: np.tan(np.pi - x) - x

def bisection(f, a, b, error):
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
        max_iter = int(math.ceil(math.log(abs(a-b)/error, 2)))
        num_add_mult = 11
        num_iter = 1
        for _ in range(max_iter):
            if fm * fb < 0:
                a, b = m, b
                fa = fm
            elif fa * fm < 0:
                a, b = a, m
                fb = fm
            m = (a + b) / 2
            fm = f(m)
            num_iter += 1
            num_add_mult += 1
            if abs(fm) < error:
                return m, fm, num_iter, num_add_mult
        return m, fm, num_iter, num_add_mult

x, fm, num1, num2 = bisection(f1, 1.7, 2.8, 1e-8)


def brent(f, a, b, error):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Źle dobrany przedział. Miejsce zerowe znajduje się poza nim.")
    c = a
    fc = fa
    num_iter = 1
    num_add_mult = 5
    while abs(b - a) > error:
        num_add_mult += 1
        if fa != fc and fb != fc:
            x = (a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) + c * fa * fb / (
                        (fc - fa) * (fc - fb)))
            num_add_mult += 20
        else:
            x = b - fb * (b - a) / (fb - fa)
            num_add_mult += 5

        if not (a < x < b):
            x = (a + b) / 2
            num_add_mult += 2

        fx = f(x)
        num_add_mult += 2
        if abs(fx) < error:
            return x, fx, num_iter, num_add_mult
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        if abs(fa) < abs(fb):
            c, fc = a, fa
        else:
            c, fc = b, fb
        num_iter += 1
        num_add_mult += 1
    return (a + b) / 2, f((a + b) / 2), num_iter, num_add_mult + 6

x1, fm1, num11, num21 = brent(f1, 1.7, 2.8, 1e-8)


def sieczne(f, a, b, max_i_num, tol=1e-8):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Źle dobrany przedział. Miejsce zerowe znajduje się poza nim.")

    x_stare = a
    fxs = f(x_stare)
    x = a - fa / (fb - fa) * (b - a)
    fx = f(x)

    num_iter = 1
    num_add_mult = 12
    for _ in range(max_i_num):
        x_s_temp, fxs_temp = x.copy(), fx.copy()
        x -= fx * (x - x_stare) / (fx - fxs)
        fx = f(x)
        x_stare, fxs = x_s_temp, fxs_temp
        num_add_mult += 5

        if abs(fx) < tol:
            return x, fx, num_iter, num_add_mult

        num_iter += 1

    return x, fx, num_iter, num_add_mult

x2,fx2, num12, num22 = sieczne(f1, 1.7, 2.8,1000)

def newton1(f, f_prime, x_0, max_i_num, tol=1e-8):
    x = x_0
    num_iter = 1
    num_add_mult = 0
    for _ in range(max_i_num):
        fx = f(x)
        num_add_mult += 2
        if abs(fx) < tol:
            return x, fx, num_iter, num_add_mult
        fpx = f_prime(x)
        num_add_mult += 3
        h = - fx / fpx
        x += h
        num_iter += 1
        num_add_mult += 3
    return x, fx, num_iter, num_add_mult

f_prime = lambda x: -1/math.cos(np.pi - x)**2 - 1

x3, fx3, num13, num23 = newton1(f1, f_prime, 2, 10000)
print("========== Wyniki ==========")
print(f"Rozwiązanie  bisekcja to: x = {x:6.4f}\nf(x) = {fm}\nLiczba iteracji: {num1}\nLiczba dodawań i mnożeń: {num2}")
print(" ")
print(f"Rozwiązanie Brent to: x = {x1:6.4f}\nf(x) = {fm1}\nLiczba iteracji: {num11}\nLiczba dodawań i mnożeń: {num21}")
print(" ")
print(f"Rozwiązanie sieczne to: x = {x2:6.4f}\nf(x) = {fx2}\nLiczba iteracji: {num12}\nLiczba dodawań i mnożeń: {num22}")
print(" ")
print(f"Rozwiązanie Newton to: x = {x3:6.4f}\nf(x) = {fx3}\nLiczba iteracji: {num13}\nLiczba dodawań i mnożeń: {num23}")
# Zad 2
print("Zad 2")

# pochodna bardzo mała więc podczas dzielenia przez coś bardzo małego bład staje się duży
#sieczna równoległa, więc ciąg przybliżeń wyskakuje nam koło 10, zamiast 4

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

f2 = lambda x: np.cosh(x) * np.cos(x) - 1
xs = np.linspace(4,8,1000)
f2_prime = lambda x: np.cos(x) * np.sinh(x) - np.sin(x) * np.cosh(x)
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

build_in_result = fsolve(f2, 4)
print(f"Rozwiązanie za pomocą wbudowanej metody fsolve:\nx={build_in_result}")
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
plt.show()

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

# Funkcje układu równań
def eq1(x, y):
    return np.tan(x) - y - 1

def eq2(x, y):
    return np.cos(x) - 3 * np.sin(y)

# Siatka punktów
x = np.linspace(0, 1.5, 400)
y = np.linspace(0, 1.5, 400)
X, Y = np.meshgrid(x, y)

# Obliczenia wartości funkcji
Z1 = eq1(X, Y)
Z2 = eq2(X, Y)

# Wykres poziomic
plt.figure(figsize=(10, 6))
contour1 = plt.contour(X, Y, Z1, levels=[0], colors='blue', linestyles='solid', linewidths=1.5, label='tan(x) - y - 1 = 0')
contour2 = plt.contour(X, Y, Z2, levels=[0], colors='red', linestyles='dashed', linewidths=1.5, label='cos(x) - 3sin(y) = 0')

# Dodanie legendy
plt.clabel(contour1, fmt="tan(x) - y - 1 = 0", colors="blue", fontsize=10)
plt.clabel(contour2, fmt="cos(x) - 3sin(y) = 0", colors="red", fontsize=10)
plt.legend(["tan(x) - y - 1 = 0", "cos(x) - 3sin(y) = 0"], loc="upper right")

# Opisy osi
plt.xlabel("x")
plt.ylabel("y")
plt.title("Poziomice układu równań")
plt.grid(True)
plt.show()


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
    Znajduje pierwiastki wielomianu za pomocą metody Newtona.

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


# słasna metoda


def evaluate_polynomial(coefficients, z):
    """
    Oblicza wartość wielomianu dla liczby zespolonej z na podstawie współczynników.

    Args:
    - coefficients: lista współczynników wielomianu (od najwyższego stopnia do stałej).
    - z: liczba zespolona, dla której obliczany jest wielomian.

    Returns:
    - Wynik obliczenia wielomianu dla danej liczby zespolonej z.
    """
    value = 0
    for i, coeff in enumerate(coefficients):
        value += coeff * z ** (len(coefficients) - i - 1)
    return value


def deflate_polynomial(coefficients, root, tol=1e-6):
    """
    Deflacja wielomianu przez dzielenie przez (z - root), usuwając znaleziony pierwiastek.

    Args:
    - coefficients: lista współczynników wielomianu (od najwyższego stopnia do stałej).
    - root: znaleziony pierwiastek wielomianu, przez który dzielimy.
    - tol: tolerancja do porównań (domyślnie 1e-6).

    Returns:
    - Nowe współczynniki wielomianu po deflacji (o stopniu niższym).
    """
    n = len(coefficients)
    new_coeffs = [0] * (n - 1)
    new_coeffs[0] = coefficients[0]  # Najwyższy współczynnik pozostaje ten sam
    for i in range(1, n - 1):
        new_coeffs[i] = coefficients[i] + new_coeffs[i - 1] * root
    remainder = coefficients[-1] + new_coeffs[-1] * root
    if abs(remainder) > tol:
        raise ValueError("Błąd deflacji: reszta jest zbyt duża.")
    return new_coeffs


def ridders_method_complex(coefficients, z0, dz, tol=1e-6, max_iter=100):
    """
    Zastosowanie metody Riddera do znajdowania pierwiastków wielomianu w przestrzeni zespolonej.

    Args:
    - coefficients: lista współczynników wielomianu (od najwyższego stopnia do stałej).
    - z0: początkowy punkt w przestrzeni zespolonej.
    - dz: krok przy obliczaniu pochodnej numerycznej.
    - tol: tolerancja zbieżności.
    - max_iter: maksymalna liczba iteracji.

    Returns:
    - Pierwiastek wielomianu znaleziony przez metodę Riddera.
    """

    def f(z):
        return evaluate_polynomial(coefficients, z)

    z = z0
    for _ in range(max_iter):
        fz = f(z)
        if abs(fz) < tol:
            return z
        # Przybliżenie pochodnej
        df = (f(z + dz) - f(z - dz)) / (2 * dz)
        if abs(df) < tol:
            raise ValueError("Pochodna bliska zeru, metoda nie zbiega.")
        # Aktualizacja
        z_new = z - fz / df
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    raise ValueError("Nie osiągnięto zbieżności w maksymalnej liczbie iteracji")


def find_all_roots_complex(coefficients, z_start, dz=0.1, tol=1e-6, max_iter=100):
    """
    Znajduje wszystkie pierwiastki wielomianu w przestrzeni zespolonej.
    Po znalezieniu pierwiastka wykonuje deflację wielomianu i kontynuuje szukanie.

    Args:
    - coefficients: lista współczynników wielomianu (od najwyższego stopnia do stałej).
    - z_start: początkowy punkt w przestrzeni zespolonej.
    - dz: krok przy obliczaniu pochodnej numerycznej.
    - tol: tolerancja zbieżności.
    - max_iter: maksymalna liczba iteracji.

    Returns:
    - Lista znalezionych pierwiastków.
    """
    roots = []
    current_coeffs = coefficients

    while len(current_coeffs) > 1:  # Dopóki mamy wielomian wyższego stopnia
        root = ridders_method_complex(current_coeffs, z_start, dz, tol, max_iter)
        roots.append(root)
        current_coeffs = deflate_polynomial(current_coeffs, root, tol)

    # Zaokrąglanie wyników do 3 miejsc po przecinku
    roots_rounded = [round(root.real, 5) + round(root.imag, 3) * 1j for root in roots]

    return roots_rounded


# Współczynniki wielomianu: z^4 + (5+1j)z^3 + (-8+5j)z^2 + (30-14j)z - 84
coefficients = [1, 5 + 1j, -8 + 5j, 30 - 14j, -84]

# Punkt startowy w przestrzeni zespolonej
z_start = 1j +1

# Znajdowanie wszystkich pierwiastków
roots = find_all_roots_complex(coefficients, z_start)

print("\n========== Wyniki metody własnej ==========")
print(f"Znalezione pierwiastki (zaokrąglone): {roots}\n")

