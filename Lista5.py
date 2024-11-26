import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

def quadratic_interpolation(h_values, rho_values): ##############
    """
    Aproksymacja kwadratowa dla funkcji rho(h) na podstawie podanych danych.

    Parametry:
        h_values (list or np.ndarray): Wysokości w kilometrach.
        rho_values (list or np.ndarray): Gęstości powietrza dla danych wysokości.

    Zwraca:
        np.poly1d: Funkcja kwadratowa aproksymująca rho(h).
    """
    coeffs = np.polyfit(h_values, rho_values, 2)
    return np.poly1d(coeffs)


def spline_interpolation(x_values, y_values, x_query):
    """
    Interpolacja funkcjami sklejanymi (spline naturalny) dla zadanych punktów.

    Parametry:
        x_values (list or np.ndarray): Znane punkty x.
        y_values (list or np.ndarray): Znane punkty y.
        x_query (float or list): Punkt(y), dla których obliczana jest interpolacja.

    Zwraca:
        float lub np.ndarray: Wartość(y) interpolowane w podanych punktach.
    """
    spline = CubicSpline(x_values, y_values, bc_type='natural')
    return spline(x_query)


def polynomial_interpolation(x_values, y_values, x_query, degree):
    """
    Interpolacja wielomianowa o zadanym stopniu dla punktów zapytania.

    Parametry:
        x_values (list or np.ndarray): Znane punkty x.
        y_values (list or np.ndarray): Znane punkty y.
        x_query (float lub list): Punkt(y), dla których obliczana jest interpolacja.
        degree (int): Stopień wielomianu.

    Zwraca:
        float lub np.ndarray: Wartość(y) interpolowane w podanych punktach.
    """
    poly_coeffs = np.polyfit(x_values, y_values, degree)
    poly = np.poly1d(poly_coeffs)
    return poly(x_query)


# Zadanie 1: Aproksymacja rho(h) jako funkcja kwadratowa
h_values = np.array([0, 3, 6])  # Wysokość w km
rho_values = np.array([1.225, 0.905, 0.652])  # Gęstość powietrza w kg/m^3

rho_quadratic = quadratic_interpolation(h_values, rho_values)
print("Zadanie 1:")
print(f"Funkcja kwadratowa aproksymująca rho(h): {rho_quadratic}")
print()

# Zadanie 2: Interpolacja wartości cD dla Re = 5, 50, i 5000 za pomocą funkcji sklejanych
Re_values = np.array([0.2, 2, 20, 200, 2000, 20000])  # Liczby Reynoldsa
cD_values = np.array([103, 13, 2.72, 0.8, 0.401, 0.433])  # Współczynnik oporu cD

query_points = [5, 50, 5000]
cD_spline_results = spline_interpolation(Re_values, cD_values, query_points)
print("Zadanie 2:")
print(f"Interpolacja funkcjami sklejanymi dla Re = {query_points}: {cD_spline_results}")
print()

# Zadanie 3: Interpolacja wielomianowa dla tych samych wartości
cD_poly_results = polynomial_interpolation(Re_values, cD_values, query_points, degree=5)
print("Zadanie 3:")
print(f"Interpolacja wielomianowa dla Re = {query_points}: {cD_poly_results}")



def exponential_fit(x, y): ####################
    """
    Aproksymacja danych za pomocą funkcji wykładniczej f(x) = a * exp(b * x)
    w sensie najmniejszych kwadratów.

    Parametry:
        x (list or np.ndarray): Dane wejściowe x.
        y (list or np.ndarray): Dane wejściowe y.

    Zwraca:
        tuple: Parametry (a, b) funkcji f(x) = a * exp(b * x).
    """
    log_y = np.log(y)  # Przejście do postaci liniowej: log(y) = log(a) + b * x
    coeffs = np.polyfit(x, log_y, 1)  # Dopasowanie liniowe
    b = coeffs[0]
    a = np.exp(coeffs[1])
    return a, b


def cubic_polynomial_fit(x, y): ####################
    """
    Dopasowanie wielomianu trzeciego stopnia do danych.

    Parametry:
        x (list or np.ndarray): Dane wejściowe x.
        y (list or np.ndarray): Dane wejściowe y.

    Zwraca:
        np.poly1d: Wielomian trzeciego stopnia aproksymujący dane.
    """
    coeffs = np.polyfit(x, y, 3)
    return np.poly1d(coeffs)


def polynomial_fits(x, y):
    """
    Dopasowanie funkcji liniowej i kwadratowej do danych.

    Parametry:
        x (list or np.ndarray): Dane wejściowe x.
        y (list or np.ndarray): Dane wejściowe y.

    Zwraca:
        tuple: Funkcje liniowa (1. stopnia) i kwadratowa (2. stopnia).
    """
    linear_coeffs = np.polyfit(x, y, 1)
    quadratic_coeffs = np.polyfit(x, y, 2)
    return np.poly1d(linear_coeffs), np.poly1d(quadratic_coeffs)


# Zadanie 4: Aproksymacja funkcji wykładniczej f(x) = a * exp(b * x)
x4 = np.array([1.2, 2.8, 4.3, 5.4, 6.8, 7.9])
y4 = np.array([7.5, 16.1, 38.9, 67.0, 146.6, 266.2])
a, b = exponential_fit(x4, y4)
f_exp = lambda x: a * np.exp(b * x)  # Utworzenie funkcji aproksymującej
residuals = y4 - f_exp(x4)
std_dev = np.std(residuals)

print("Zadanie 4:")
print(f"Funkcja wykładnicza: f(x) = {a:.3f} * exp({b:.3f} * x)")
print(f"Odchylenie standardowe: {std_dev:.3f}")

# Wykres dla zadania 4
plt.figure(figsize=(10, 5))
plt.scatter(x4, y4, label="Dane", color="blue")
plt.plot(x4, f_exp(x4), label="Aproksymacja", color="red")
plt.title("Zadanie 4: Aproksymacja funkcją wykładniczą")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Zadanie 5: Dopasowanie wielomianu trzeciego stopnia do danych
T = np.array([0, 21.1, 37.8, 54.4, 71.1, 87.8, 100])  # Temperatura
mu = np.array([1.79, 1.13, 0.696, 0.519, 0.338, 0.321, 0.296])  # Lepkość

cubic_poly = cubic_polynomial_fit(T, mu)
T_query = [10, 30, 60, 90]
mu_approximated = cubic_poly(T_query)

print("Zadanie 5:")
print(f"Wielomian trzeciego stopnia: {cubic_poly}")
print(f"Interpolowane wartości dla T = {T_query}: {mu_approximated}")

# Wykres dla zadania 5
plt.figure(figsize=(10, 5))
plt.scatter(T, mu, label="Dane", color="blue")
T_range = np.linspace(0, 100, 500)
plt.plot(T_range, cubic_poly(T_range), label="Aproksymacja (3 stopnia)", color="red")
plt.title("Zadanie 5: Aproksymacja lepkości wielomianem 3. stopnia")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Lepkość (10^-3 m^2/s)")
plt.legend()
plt.grid()
plt.show()

# Zadanie 6: Dopasowanie funkcji liniowej i kwadratowej do danych
x6 = np.array([1.0, 2.5, 3.5, 4.0, 1.1, 1.8, 2.2, 3.7])
y6 = np.array([6.008, 15.722, 27.13, 33.772, 5.257, 9.549, 11.098, 28.828])

linear_fit, quadratic_fit = polynomial_fits(x6, y6)

print("Zadanie 6:")
print(f"Funkcja liniowa: {linear_fit}")
print(f"Funkcja kwadratowa: {quadratic_fit}")

# Wykres dla zadania 6
plt.figure(figsize=(10, 5))
plt.scatter(x6, y6, label="Dane", color="blue")
x_range = np.linspace(min(x6), max(x6), 500)
plt.plot(x_range, linear_fit(x_range), label="Aproksymacja liniowa", color="green")
plt.plot(x_range, quadratic_fit(x_range), label="Aproksymacja kwadratowa", color="red")
plt.title("Zadanie 6: Dopasowanie liniowe i kwadratowe")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
