import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

def lagrange_interpolation(x_points, y_points):
    """
    Implementacja interpolacji wielomianowej metodą Lagrange'a.

    Parametry:
        x_points (list or np.ndarray): Współrzędne x punktów.
        y_points (list or np.ndarray): Wartości funkcji w punktach x.

    Zwraca:
        funkcja: Funkcja interpolacyjna jako lambda.
    """
    def lagrange_basis(i, x):
        basis = 1
        for j in range(len(x_points)):
            if i != j:
                basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return basis

    def interpolation_function(x):
        result = 0
        for i in range(len(x_points)):
            result += y_points[i] * lagrange_basis(i, x)
        return result

    return interpolation_function

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
    cubic_coeffs = np.polyfit(x, y, 3)
    return np.poly1d(linear_coeffs), np.poly1d(quadratic_coeffs), np.poly1d(cubic_coeffs)


def exponential_fit(x, y):
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


def lagrange_interpolation_with_polynomial(x_points, y_points):
    """
    Implementacja interpolacji wielomianowej metodą Lagrange'a.

    Parametry:
        x_points (list or np.ndarray): Współrzędne x punktów.
        y_points (list or np.ndarray): Wartości funkcji w punktach x.

    Zwraca:
        tuple:
            - Funkcja interpolacyjna jako lambda.
            - Symboliczny wielomian w postaci wyrażenia sympy.
    """
    x = sp.Symbol('x')
    n = len(x_points)
    polynomial = 0

    # Budowanie wielomianu metodą Lagrange'a
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        polynomial += term

    # Konwersja do uproszczonej postaci
    polynomial = sp.simplify(polynomial)

    # Tworzenie funkcji interpolacyjnej jako lambda
    interpolation_function = sp.lambdify(x, polynomial, 'numpy')

    return interpolation_function, polynomial


print("Zadanie 1:")
# Zadanie 1: Aproksymacja rho(h) jako funkcja kwadratowa
# Dane wejściowe
h_values = np.array([0, 3, 6])  # Wysokość w km
rho_values = np.array([1.225, 0.905, 0.652])  # Gęstość powietrza w kg/m^3

# Tworzenie funkcji interpolacyjnej oraz wielomianu
lagrange_rho_function, lagrange_rho_polynomial = lagrange_interpolation_with_polynomial(h_values, rho_values)

# Wyświetlanie wielomianu
print("Interpolowany wielomian metodą Lagrange'a:")
print(lagrange_rho_polynomial)

# Testowanie interpolacji dla przykładowych wartości
test_heights = np.linspace(0, 6, 100)  # Przykładowe wysokości do ewaluacji
interpolated_rho = lagrange_rho_function(test_heights)

# Wykres dla wizualizacji
plt.plot(test_heights, interpolated_rho, label="Zad 1. Interpolacja Lagrange'a", color="blue")
plt.scatter(h_values, rho_values, color="red", label="Punkty danych")
plt.xlabel("Wysokość (km)")
plt.ylabel("Gęstość powietrza (kg/m³)")
plt.title("Interpolacja gęstości powietrza metodą Lagrange'a")
plt.legend()
plt.grid()
plt.show()



# metroda biblioteczna
_, rho_quadratic, _ = polynomial_fits(h_values, rho_values)

# Tworzenie funkcji interpolacyjnej metodą Lagrange'a
lagrange_rho = lagrange_interpolation(h_values, rho_values)

# Testowanie interpolacji dla przykładowych wartości
test_heights = np.linspace(0, 6, 100)  # Przykładowe wysokości do ewaluacji
interpolated_rho = [lagrange_rho(h) for h in test_heights]


print(f"Funkcja kwadratowa aproksymująca rho(h): {[rho_quadratic]}")
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
print("Zadanie 3:")

# Tworzenie funkcji interpolacyjnej i wielomianu metodą Lagrange'a
cD_function, cD_polynomial = lagrange_interpolation_with_polynomial(Re_values, cD_values)

# Znalezienie wartości cD dla Re = 5, 50, 5000
Re_to_interpolate = [5, 50, 5000]
interpolated_cD = [cD_function(Re) for Re in Re_to_interpolate]

# Wyświetlanie wyników
print("Interpolowany wielomian metodą Lagrange'a:")
# print(cD_polynomial) ###########################################################
print("\nWyniki interpolacji:")
for Re, cD in zip(Re_to_interpolate, interpolated_cD):
    print(f"c_D dla Re = {Re}: {cD}")


# Zadanie 4: Aproksymacja funkcji wykładniczej f(x) = a * exp(b * x)
# logarytm z tego == log(a*exp(bx)) = c + bx
x4 = np.array([1.2, 2.8, 4.3, 5.4, 6.8, 7.9])
y4 = np.array([7.5, 16.1, 38.9, 67.0, 146.6, 266.2])
a, b = exponential_fit(x4, y4)
f_exp = lambda x: a * np.exp(b * x)  # Utworzenie funkcji aproksymującej
residuals = y4 - f_exp(x4)
std_dev = np.std(residuals)

#regresja liniowa klasyczna
def linear_regression(x,y):
    b = np.sum((x-np.mean(x)) * (y - np.mean(y)))/np.sum((x - np.mean(x))**2)
    c = np.mean(y) - b * np.mean(x)
    return b, c

ys_log = np.log(y4)

b4, c4 = linear_regression(x4, ys_log)
a4 = np.exp(c4)
f_approx = lambda x: a4 * np.exp(b4 * x)

x_lin = np.linspace(1,9,1000)
y_approx = f_approx(x_lin)

plt.scatter(x4, y4, c="r")
plt.title("Zadanie 4: Aproksymacja funkcją wykładniczą własna")
plt.plot(x_lin, y_approx)
plt.ylim(0,300)
plt.xlim(0,9)
plt.show()
y_approx2 = f_approx(x4)
errors = y4 - y_approx2

print("Zadanie 4:")
print(f"Funkcja wykładnicza: f(x) = {a:.3f} * exp({b:.3f} * x)")
print(f"Odchylenie standardowe: {std_dev:.3f}, własne odchylenie: {np.std(errors)}")

# Wykres dla zadania 4
plt.figure(figsize=(10, 5))
plt.scatter(x4, y4, label="Dane", color="blue")
plt.plot(x4, f_exp(x4), label="Aproksymacja", color="red")
plt.title("Zadanie 4: Aproksymacja funkcją wykładniczą bib")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
# plt.show()

# Zadanie 5: Dopasowanie wielomianu trzeciego stopnia do danych
T = np.array([0, 21.1, 37.8, 54.4, 71.1, 87.8, 100])  # Temperatura
mu = np.array([1.79, 1.13, 0.696, 0.519, 0.338, 0.321, 0.296])  # Lepkość

_, _, cubic_poly = polynomial_fits(T, mu)
T_query = [10, 30, 60, 90]
mu_approximated = cubic_poly(T_query)

cubic_poly1 = np.array(cubic_poly)

# Interpolacja metodą Lagrange'a
lagrange_function, lagrange_polynomial = lagrange_interpolation_with_polynomial(T, mu)
mu_lagrange_approximated = [lagrange_function(t) for t in T_query]

print("Zadanie 5:")
print(f"Wielomian trzeciego stopnia: {cubic_poly}") # chce aby to była lista współczynników
print(f"Interpolowane wartości dla T = {T_query}: {mu_approximated}")

#print(f"Interpolowany wielomian metodą Lagrange'a: {lagrange_polynomial}")
print(f"Interpolowane wartości dla T = {T_query} (metoda Lagrange'a): {mu_lagrange_approximated}")


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


def polynomial_fit(x_points, y_points, degree):
    """
    Dopasowanie wielomianu o zadanym stopniu.

    Parametry:
        x_points (np.ndarray): Współrzędne x punktów.
        y_points (np.ndarray): Wartości funkcji w punktach x.
        degree (int): Stopień wielomianu do dopasowania.

    Zwraca:
        numpy.polynomial.Polynomial: Wielomian dopasowany do danych.
    """
    coeffs = np.polyfit(x_points, y_points, degree)
    return np.poly1d(coeffs)


# Zadanie 6: Dopasowanie funkcji liniowej i kwadratowej do danych
x6 = np.array([1.0, 2.5, 3.5, 4.0, 1.1, 1.8, 2.2, 3.7])
y6 = np.array([6.008, 15.722, 27.13, 33.772, 5.257, 9.549, 11.098, 28.828])

linear_fit, quadratic_fit, _ = polynomial_fits(x6, y6)

print("Zadanie 6:")
print(f"Funkcja liniowa: {[linear_fit]}")
print(f"Funkcja kwadratowa: {[quadratic_fit]}")

# Interpolacja Lagrange'a
lagrange_func, lagrange_poly = lagrange_interpolation_with_polynomial(x6, y6)

# Dopasowanie funkcji liniowej i kwadratowej
linear_fit = polynomial_fit(x6, y6, 1)
quadratic_fit = polynomial_fit(x6, y6, 2)
#print("Wielomian interpolacyjny Lagrange'a:")
#print(lagrange_poly)

print("Wielomiany metodą Lagrangea")
print("\nWspółczynniki funkcji liniowej:")
print(linear_fit)
print("\nWspółczynniki funkcji kwadratowej:")
print(quadratic_fit)

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

# Tworzenie wykresu
x_vals = np.linspace(min(x6), max(x6), 500)
lagrange_vals = lagrange_func(x_vals)
linear_vals = linear_fit(x_vals)
quadratic_vals = quadratic_fit(x_vals)
plt.figure(figsize=(10, 6))

# Punkty danych
plt.scatter(x6, y6, color='red', label='Dane', zorder=5)

# Interpolacja Lagrange'a
plt.plot(x_vals, lagrange_vals, label='Interpolacja Lagrange\'a', color='blue', linestyle='--')

# Dopasowanie liniowe
plt.plot(x_vals, linear_vals, label='Funkcja liniowa', color='green')

# Dopasowanie kwadratowe
plt.plot(x_vals, quadratic_vals, label='Funkcja kwadratowa', color='orange')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Zad 6 bis Interpolacja i dopasowanie wielomianów')
plt.legend()
plt.grid(True)
plt.show()
