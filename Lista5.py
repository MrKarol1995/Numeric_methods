import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def quadratic_interpolation(h_values, rho_values):
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