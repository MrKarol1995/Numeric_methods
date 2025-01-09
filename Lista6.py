import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad
import math

# Metoda Simpsona
def simpson_rule(f, a, b, n):
    if n % 2 == 0:
        n += 1  # Simpson wymaga nieparzystej liczby punktów
    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    y = f(x)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:n-1:2]) + 2 * np.sum(y[2:n-2:2]) + y[-1])
    return integral


def trapezoidal_rule(x, y):
    h = np.diff(x)
    return np.sum(h * (y[:-1] + y[1:]) / 2)

# Zadanie 1: Czas potrzebny do rozpędzenia samochodu
def zad1():
    # Dane
    m = 2000  # masa w kg
    v = np.array([0, 1.0, 1.8, 2.4, 3.5, 4.4, 5.1, 6.0])  # prędkość w m/s
    P = np.array([0, 4.7, 12.2, 19.0, 31.8, 40.1, 43.8, 43.2]) * 1000  # moc w W (przeliczenie z kW)

    # Usunięcie punktów, gdzie P = 0
    mask = P > 0  # Tylko tam, gdzie moc jest większa od 0
    v = v[mask]
    P = P[mask]


    # Funkcja do całkowania
    f = v / P
    time = m * trapezoidal_rule(v, f)  # ∆t = m ∫(v/P) dv
    print(f"Zadanie 1: Czas potrzebny do rozpędzenia: {time:.4f} s")


# Zadanie 2: Całka przy użyciu wzoru Simpsona
def zad2():
    # Funkcja do całkowania
    def f(x):
        return np.cos(2 * np.arccos(x))

    # Przedział całkowania
    a, b = -1, 1
    results = {}

    # Obliczenia dla różnych liczby węzłów
    for n in [3, 5, 7]:
        integral = simpson_rule(f, a, b, n)
        results[n] = integral

    for n, integral in results.items():
        print(f"Zadanie 2: Całka dla {n} węzłów: {integral:.6f}")

# wychodzi to samo, bo bład metody simpsona wynosi u nas zero, więc każdy węzeł zwraca to samo



# Zadanie 3: Całka przy użyciu metody trapezów, z podstawieniem x^3 = 1/t
def zad3():
    # Funkcja do całkowania po zmianie zmiennej
    def f(t):
        x = 1/(t)**(1/3)
        return 1/3 * t**(-4/3)/(1+t**(-4/3))

# funkcja niezdefiniowana dla zera, więc  bierzemy coś bardzo małego
    # Przedział całkowania w zmiennej t
    t_values = np.linspace(10**(-6), 1, 6)  # 6 węzłów
    y = f(t_values)

    # Obliczenie całki metodą trapezów
    integral = trapezoidal_rule(t_values, y)
    print(f"Zadanie 3: Całka wynosi: {integral:.6f}")

# Wywołanie funkcji
zad1()
zad2()
zad3()


# Zadanie 4: Wahadło matematyczne
def h(theta0, n=10):
    """
    Oblicza wartość h(theta0) jako całkę:
    h(theta0) = ∫_0^(π/2) dθ / √(1 - sin^2(θ0/2) * sin^2(θ))
    """
    theta0_rad = np.radians(theta0)  # konwersja na radiany
    sin2_theta0_half = np.sin(theta0_rad / 2)**2

    def integrand(theta):
        return 1 / np.sqrt(1 - sin2_theta0_half * np.sin(theta)**2)

    result = simpson_rule(integrand, 0, np.pi / 2, n)
    return result

# Obliczanie wartości h(theta0) dla różnych kątów
angles = [0, 15, 30, 45]
h_values = [h(theta0) for theta0 in angles]
h_approx = np.pi / 2  # Przybliżenie harmoniczne

print("Zadanie 4: Wahadło matematyczne")
for angle, h_val in zip(angles, h_values):
    print(f"h({angle}°) = {h_val:.6f}")
print(f"przybliżenie harmoniczne h(0): {h_approx:.6f}")

# Zadanie 5: Całka metodą Gaussa-Legendre'a
def gauss_legendre_integration(func, a, b, n):
    """
    Oblicza całkę metodą Gaussa-Legendre'a z n węzłami na przedziale [a, b].
    """
    # Węzły i współczynniki
    nodes, weights = roots_legendre(n)
    # Przekształcenie na przedział [a, b]
    t = 0.5 * (nodes + 1) * (b - a) + a
    w = 0.5 * (b - a) * weights
    return np.sum(w * func(t))

# Funkcja podcałkowa
def func(x):
    return np.log(x) / (x**2 - 2*x + 2)

# Przedział całkowania
a, b = 1, np.pi

# Obliczanie całki dla 2 i 4 węzłów
integral_2 = gauss_legendre_integration(func, a, b, 2)
integral_4 = gauss_legendre_integration(func, a, b, 4)

print("Zadanie 5: Całka metodą Gaussa-Legendre'a")
print(f"Całka dla 2 węzłów: {integral_2:.6f}")
print(f"Całka dla 4 węzłów: {integral_4:.6f}")

# Zad 6

print("Zad 6")
# Definicja funkcji
def f1(x):
    return x ** 3 - 2 * x

def f2(x):
    return math.sin(x)

def f3(x):
    return math.exp(x)

# Pochodne analityczne
def df1(x):
    return 3 * x ** 2 - 2

def df2(x):
    return math.cos(x)

def df3(x):
    return math.exp(x)


# Definicje metod różnicowych
def forward(f, x, h):
    return (f(x) - f(x - h)) / h


def central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def central_fourth_order(f, x, h):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)

# Wartości do obliczeń
values = [
    (f1, df1, 1),  # f1'(1)
    (f2, df2, math.pi / 3),  # f2'(pi/3)
    (f3, df3, 0)  # f3'(0)
]

# Różne wartości kroku h
h_values = [0.1, 0.01, 0.001]

# Generowanie wyników
for f, df, x in values:
    print(f"\nDla x = {x}:")
    for h in h_values:
        f_analytical = df(x)
        f_forward = forward(f, x, h)
        f_central = central(f, x, h)
        f_central_fourth = central_fourth_order(f, x, h)

        print(f"h = {h}")
        print(f"f'(x) - Df1 = {abs(f_analytical - f_forward):.6f}")
        print(f"f'(x) - Dc2 = {abs(f_analytical - f_central):.6f}")
        print(f"f'(x) - Dc4 = {abs(f_analytical - f_central_fourth):.6f}")

# Zad 7

def poly_derivative(coeffs):
    n = len(coeffs) - 1
    new_coeffs = []
    for i in range(n):
        new_coeffs.append(coeffs[i] * (n-i))
    return np.poly1d(new_coeffs)


def central_fourth_order2(x, xs, ys):
    sorted_pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    xs_sorted, ys_sorted = zip(*sorted_pairs)
    i = xs_sorted.index(x)
    h = xs[i] - xs[i-1]
    return (-ys_sorted[i+2] + 8*ys_sorted[i+1] - 8*ys_sorted[i-1] + ys_sorted[i-2]) / (12 * h)


# Funkcja do obliczania współczynników Lagrange'a
def lagrange_derivative(x_vals, f_vals, x0, order=10):
    n = len(x_vals)
    result = 0
    for i in range(n):
        term = 0
        for j in range(n):
            if i != j:
                prod = 1 / (x_vals[i] - x_vals[j])
                for k in range(n):
                    if k != i and k != j:
                        prod *= (x0 - x_vals[k]) if order == 1 else 1 / (x_vals[i] - x_vals[k])
                term += prod
        result += f_vals[i] * term * (1 if order == 1 else 2)
    return result

# Dane z tabeli dla funkcji (przykładowe dane dla interpolacji)
x_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
f_points = np.array([0.0, 0.078348, 0.138910, 0.192916, 0.244981])

# Obliczenie różnic centralnych o wysokim rzędzie

def higher_order_central_difference(f_vals, x_vals, x, h):
    i = np.where(x_vals == x)[0][0]  # Znajdujemy indeks odpowiadający x
    return (-f_vals[i + 2] + 8 * f_vals[i + 1] - 8 * f_vals[i - 1] + f_vals[i - 2]) / (12 * h)

# Ustawienie kroku (h) i obliczenia

ffprim = lagrange_derivative(x_points, f_points, 0.2, 1)
h = 0.1
f_prime_0_2 = higher_order_central_difference(f_points, x_points, 0.2, h)
print("\nZadanie 7")
print(f" I: Centralna czwartego rzędu: {central_fourth_order2(0.2, x_points, f_points)}")
print(f"Najdokładniejsze przybliżenie f'(0.2) wynosi: {f_prime_0_2:.6f}")

# Zadanie 8
print("\nZadanie 8")
# Dane
x_vals = np.array([-2.2, -0.3, 0.8, 1.9])
f_vals = np.array([15.180, 10.962, 1.92, -2.04])



def poly_interpol2(x,y):
    n = len(x)
    X = np.poly1d([1,0])
    poly = 0
    for i in range(n):
        w = 1
        for j in range(n):
            if i != j:
                w *= ((X-x[j]) / (x[i]-x[j]))
        poly += w * y[i]
    return poly

poly = poly_interpol2(x_vals,f_vals)
coeffs = poly.coeffs

poly_prime = poly_derivative(coeffs)
poly_bis = poly_derivative(poly_prime.coeffs)
print(f"f'(0) = {poly_prime(0)}")
print(f"f''(0) = {poly_bis(0)}")