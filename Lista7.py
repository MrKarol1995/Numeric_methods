from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

# 1. Zadanie pierwsze: y' + 4y = x^2, y(0) = 1
def zad1_analytical_solution(x):
    return (31 / 32) * np.exp(-4 * x) + (1 / 4) * x ** 2 - (1 / 8) * x + (1 / 32)

def euler_method(f, y0, x0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    y_values = [y0]
    for i in range(1, len(x_values)):
        y_next = y_values[-1] + h * f(x_values[i - 1], y_values[-1])
        y_values.append(y_next)
    return x_values, y_values

def rk2_method(f, y0, x0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    y_values = [y0]
    for i in range(1, len(x_values)):
        k1 = f(x_values[i - 1], y_values[-1])
        k2 = f(x_values[i - 1] + h / 2, y_values[-1] + (h / 2) * k1)
        y_next = y_values[-1] + h * k2
        y_values.append(y_next)
    return x_values, y_values

def rk4_method(f, y0, t0, t_end, h, *params):
    t_values = np.arange(t0, t_end + h, h)
    y_values = [y0]
    for i in range(1, len(t_values)):
        k1 = f(t_values[i - 1], y_values[-1], *params)
        k2 = f(t_values[i - 1] + h / 2, y_values[-1] + (h / 2) * k1, *params)
        k3 = f(t_values[i - 1] + h / 2, y_values[-1] + (h / 2) * k2, *params)
        k4 = f(t_values[i - 1] + h, y_values[-1] + h * k3, *params)
        y_next = y_values[-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_values.append(y_next)
    return t_values, np.array(y_values)

# Funkcja dla równania pierwszego
def zad1_function(x, y):
    return x ** 2 - 4 * y

print("Zad 1")
# Warunki początkowe i kroki
y0 = 1
x0 = 0
x_end = 0.03
steps = [1, 2, 4]
for step in steps:
    h = (x_end - x0) / step
    x_euler, y_euler = euler_method(zad1_function, y0, x0, x_end, h)
    x_rk2, y_rk2 = rk2_method(zad1_function, y0, x0, x_end, h)
    x_rk4, y_rk4 = rk4_method(zad1_function, y0, x0, x_end, h)
    print(f"\nWyniki dla h = {h}:")
    print("Euler:", y_euler[-1], "RK2:", y_rk2[-1], "RK4:", y_rk4[-1])

# Zad 2
# 2. Zadanie drugie: y' = sin(y), y(0) = 1
def zad2_function(x, y):
    return np.sin(y)


x0, x_end, h = 0, 0.5, 0.1
y0 = 1
x_euler, y_euler = euler_method(zad2_function, y0, x0, x_end, h)
x_rk4, y_rk4 = rk4_method(zad2_function, y0, x0, x_end, h)

plt.figure(figsize=(10, 5))
plt.plot(x_euler, y_euler, 'r--', label="Euler Method")
plt.plot(x_rk4, y_rk4, 'b-', label="RK4 Method")
plt.title("Zadanie 2: Porównanie metod numerycznych")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Zad 3
# 3. Zadanie trzecie: równanie wahadła
def pendulum(t, y, Q, omega_hat, A_hat):
    theta, v = y
    dydt = [v, - (1 / Q) * v - np.sin(theta) + A_hat * np.cos(omega_hat * t)]
    return np.array(dydt)


# Parametry i warunki początkowe
scenarios = [
    {"Q": 2, "omega_hat": 2 / 3, "A_hat": 0.5, "v0": 0.0, "theta0": 0.01},
    {"Q": 2, "omega_hat": 2 / 3, "A_hat": 0.5, "v0": 0.0, "theta0": 0.3},
    {"Q": 2, "omega_hat": 2 / 3, "A_hat": 1.35, "v0": 0.0, "theta0": 0.3},
]

h = 0.01
t0, t_end = 0, 50
for scenario in scenarios:
    Q, omega_hat, A_hat = scenario["Q"], scenario["omega_hat"], scenario["A_hat"]
    y0 = [scenario["theta0"], scenario["v0"]]
    t_values, y_values = rk4_method(pendulum, y0, t0, t_end, h, Q, omega_hat, A_hat)

    plt.figure(figsize=(10, 5))
    plt.plot(t_values, y_values[:, 0], label=f"Q={Q}, Â={A_hat}, θ0={y0[0]}")
    plt.title("Ruch wahadła - θ(t)")
    plt.xlabel("Czas τ")
    plt.ylabel("Kąt θ")
    plt.legend()
    plt.grid()
    plt.show()


# Parametry zadania
scenarios = [
    {"Q": 2, "w_hat": 2/3, "A_hat": 0.5, "theta0": 0.01, "omega0": 0.0},
    {"Q": 2, "w_hat": 2/3, "A_hat": 0.5, "theta0": 0.3, "omega0": 0.0},
    {"Q": 2, "w_hat": 2/3, "A_hat": 1.35, "theta0": 0.3, "omega0": 0.0},
]

t0, t_end, h = 0, 80, 0.01  # Czas początkowy, końcowy, krok czasowy

# Rozwiązanie dla każdego przypadku
plt.figure(figsize=(12, 6))
for idx, scenario in enumerate(scenarios):
    Q, w_hat, A_hat = scenario["Q"], scenario["w_hat"], scenario["A_hat"]
    theta0, omega0 = scenario["theta0"], scenario["omega0"]

    # Rozwiązanie równań ruchu za pomocą nowej metody
    y0 = [theta0, omega0]
    t_values, y_values = rk4_method(pendulum, y0, t0, t_end, h, Q, w_hat, A_hat)

    # Wykresy w przestrzeni fazowej
    plt.plot(y_values[:, 0], y_values[:, 1], label=f"Scenariusz {idx+1}: Q={Q}, A_hat={A_hat}, theta0={theta0}")

# Konfiguracja wykresu
plt.title("Wykres w przestrzeni fazowej wahadła: θ vs dθ/dτ")
plt.xlabel("θ (kąt)")
plt.ylabel("dθ/dτ (prędkość kątowa)")
plt.legend()
plt.grid()
plt.show()

# ZAd4
# 4.1 Trajektoria bez oporów powietrza
def projectile_no_air_resistance(v0, angle, g=9.81, t_max=10, dt=0.01):
    angle_rad = np.radians(angle)
    t = np.arange(0, t_max, dt)
    x = v0 * np.cos(angle_rad) * t
    y = v0 * np.sin(angle_rad) * t - 0.5 * g * t**2
    y = np.maximum(y, 0)  # Piłka nie spada poniżej ziemi
    return x, y

# 4.2 Trajektoria z oporem powietrza
def projectile_with_air_resistance(v0, angle, cw, rho, A, m, g=9.81, dt=0.01):
    angle_rad = np.radians(angle)
    vx, vy = v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)
    x, y = [0], [0]

    while y[-1] >= 0:
        v = np.sqrt(vx**2 + vy**2)
        Fx = -0.5 * cw * rho * A * v * vx
        Fy = -0.5 * cw * rho * A * v * vy - m * g

        ax = Fx / m
        ay = Fy / m

        vx += ax * dt
        vy += ay * dt

        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)

    return np.array(x), np.array(y)


# Parametry dla czterech zestawów
params = [
    {"v0": 20, "angle": 45, "cw": 0.35, "rho": 1.2, "A": 0.03, "m": 0.145},  # Zestaw 1
    {"v0": 25, "angle": 40, "cw": 0.35, "rho": 1.2, "A": 0.03, "m": 0.145},  # Zestaw 2
    {"v0": 20, "angle": 30, "cw": 0.5, "rho": 1.2, "A": 0.05, "m": 0.2},  # Zestaw 3
    {"v0": 15, "angle": 60, "cw": 0.5, "rho": 1.2, "A": 0.05, "m": 0.2}  # Zestaw 4
]

# Tworzenie wykresu
plt.figure(figsize=(12, 8))

for idx, p in enumerate(params):
    v0, angle, cw, rho, A, m = p["v0"], p["angle"], p["cw"], p["rho"], p["A"], p["m"]

    # Obliczanie trajektorii bez oporu powietrza
    x_no_air, y_no_air = projectile_no_air_resistance(v0, angle)

    # Obliczanie trajektorii z oporem powietrza
    x_with_air, y_with_air = projectile_with_air_resistance(v0, angle, cw, rho, A, m)

    # Wykres
    plt.plot(x_no_air, y_no_air, linestyle="--", label=f"Zestaw {idx + 1} - Bez oporu powietrza")
    plt.plot(x_with_air, y_with_air, label=f"Zestaw {idx + 1} - Z oporem powietrza")

# Konfiguracja wykresu
plt.title("Porównanie trajektorii piłki rzuconej ukośnie (4 zestawy parametrów)")
plt.xlabel("Odległość [m]")
plt.ylabel("Wysokość [m]")
plt.legend()
plt.grid(True)
plt.show()

# Zad 5
# Definiowanie równania różniczkowego (zamiana na układ równań pierwszego rzędu)
def fun(x, y):
    dydx = np.zeros_like(y)
    dydx[0] = y[1]
    dydx[1] = -(1 - 0.2 * x) * y[0]**2
    return dydx

# Warunki brzegowe
def bc(ya, yb):
    return np.array([ya[0], yb[0] - 1])

# Siatka początkowa i zgadywanie rozwiązania
x = np.linspace(0, np.pi / 2, 50)
y_guess = np.zeros((2, x.size))
y_guess[0] = x / (np.pi / 2)  # Przybliżenie liniowe dla warunku y(pi/2) = 1

# Rozwiązanie równania przy użyciu metody solve_bvp
sol = solve_bvp(fun, bc, x, y_guess)

# Funkcja do rozwiązania przy użyciu metody Rungego-Kutty czwartego rzędu (RK4)
def rk4_method(f, y0, t0, t_end, h, *params):
    t_values = np.arange(t0, t_end + h, h)
    y_values = [y0]
    for i in range(1, len(t_values)):
        k1 = f(t_values[i - 1], y_values[-1], *params)
        k2 = f(t_values[i - 1] + h / 2, y_values[-1] + (h / 2) * k1, *params)
        k3 = f(t_values[i - 1] + h / 2, y_values[-1] + (h / 2) * k2, *params)
        k4 = f(t_values[i - 1] + h, y_values[-1] + h * k3, *params)
        y_next = y_values[-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_values.append(y_next)
    return t_values, np.array(y_values)

# Warunki początkowe i parametry dla RK4
y0_rk4 = [0, 0]  # Warunki początkowe dla y(0) i y'(0)
t0, t_end, h = 0, np.pi / 2, 0.01  # Początkowy i końcowy czas oraz krok czasowy
params_rk4 = []  # Brak dodatkowych parametrów w tym przypadku

# Rozwiązanie przy użyciu metody RK4
t_values_rk4, y_values_rk4 = rk4_method(fun, y0_rk4, t0, t_end, h)

# Wykres
x_plot = np.linspace(0, np.pi / 2, 100)
y_plot_bvp = sol.sol(x_plot)[0]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot_bvp, label="Rozwiązanie numeryczne (solve_bvp)", color='b', linestyle='--')
plt.plot(t_values_rk4, y_values_rk4[:, 0], label="Rozwiązanie RK4", color='r')
plt.title("Porównanie rozwiązań zagadnienia brzegowego")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid()
plt.show()
