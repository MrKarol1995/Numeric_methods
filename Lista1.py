import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cProfile

# Zad1


# Definicje funkcji przybliżeń Padégo
def z1(x):
    return (6 - 2 * x) / (6 + 4 * x + x ** 2)


def z2(x):
    return (6 - 4 * x + x ** 2) / (6 + 2 * x)

def funkcja(x):
    return np.exp(-x)

# Tworzenie danych
x = np.linspace(-8, 8, 300)
wartości_1z= z1(x)
wartości_2z = z2(x)
wartości_funkcji=funkcja(x)

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(x, wartości_1z, label="przybliżenie z1", color="darkcyan", linewidth=1.3)
plt.plot(x, wartości_2z, label="przybliżenie z2", color="plum", linewidth=1.3)
plt.plot(x, wartości_funkcji, label="funkcja exp(-x)", color="gold", linewidth=1.3)
plt.title("Przybliżenie Padégo funkcji exp(-x)", fontsize=12, fontweight='bold', color="dimgray")
plt.xlabel("x", fontsize=10, fontweight='bold', color="dimgray")
plt.ylabel("wartość przybliżenia", fontsize=10, fontweight='bold', color="dimgray")
plt.gca().set_facecolor('snow')
plt.legend()
plt.grid(True)

# Zapis wykresu jako plik PNG
plt.savefig('przybliżenie_padego.png', dpi=300, bbox_inches='tight')

# Wyświetlenie wykresu
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(12, 8))

# Pierwszy wykres - przybliżenie z1
axs[0].plot(x, wartości_1z, label="przybliżenie z1", color="darkcyan", linewidth=1.3)
axs[0].plot(x, wartości_funkcji, label="funkcja exp(-x)", color="gold", linewidth=1.3)
axs[0].set_title("Przybliżenie z1 vs. exp(-x)", fontsize=12, fontweight='bold', color="dimgray")
axs[0].set_xlabel("x", fontsize=10, fontweight='bold', color="dimgray")
axs[0].set_ylabel("wartość przybliżenia", fontsize=10, fontweight='bold', color="dimgray")
axs[0].legend()
axs[0].grid(True)
axs[0].set_facecolor('snow')

# Drugi wykres - przybliżenie z2
axs[1].plot(x, wartości_2z, label="przybliżenie z2", color="plum", linewidth=1.3)
axs[1].plot(x, wartości_funkcji, label="funkcja exp(-x)", color="gold", linewidth=1.3)
axs[1].set_title("Przybliżenie z2 vs. exp(-x)", fontsize=12, fontweight='bold', color="dimgray")
axs[1].set_xlabel("x", fontsize=10, fontweight='bold', color="dimgray")
axs[1].set_ylabel("wartość przybliżenia", fontsize=10, fontweight='bold', color="dimgray")
axs[1].legend()
axs[1].grid(True)
axs[1].set_facecolor('snow')

# Trzeci wykres - exp(-x)
axs[2].plot(x, wartości_funkcji, label="funkcja exp(-x)", color="gold", linewidth=1.3)
axs[2].set_title("Funkcja exp(-x)", fontsize=12, fontweight='bold', color="dimgray")
axs[2].set_xlabel("x", fontsize=10, fontweight='bold', color="dimgray")
axs[2].set_ylabel("wartość funkcji", fontsize=10, fontweight='bold', color="dimgray")
axs[2].legend()
axs[2].grid(True)
axs[2].set_facecolor('snow')

# Dostosowanie układu i zapis wykresu jako plik PNG
plt.tight_layout()
plt.savefig('wielowykres_padego.png', dpi=300, bbox_inches='tight')

# Wyświetlenie wykresów
plt.show()

# Obliczanie błędów
błąd_bezwzględny_z1 = np.abs(wartości_funkcji - wartości_1z)
błąd_bezwzględny_z2 = np.abs(wartości_funkcji - wartości_2z)

# Średni błąd bezwzględny (MAE)
mae_z1 = np.mean(błąd_bezwzględny_z1)
mae_z2 = np.mean(błąd_bezwzględny_z2)

# Średni błąd kwadratowy (MSE)
mse_z1 = np.mean((wartości_funkcji - wartości_1z)**2)
mse_z2 = np.mean((wartości_funkcji - wartości_2z)**2)

# Wyświetlenie błędów
print(f"Średni błąd bezwzględny dla z1: {mae_z1}")
print(f"Średni błąd bezwzględny dla z2: {mae_z2}")
print(f"Średni błąd kwadratowy dla z1: {mse_z1}")
print(f"Średni błąd kwadratowy dla z2: {mse_z2}")

# Tworzenie wykresu błędów
plt.figure(figsize=(10, 6))

plt.plot(x, błąd_bezwzględny_z1, label="Błąd bezwzględny z1", color="darkcyan", linewidth=1.3)
plt.plot(x, błąd_bezwzględny_z2, label="Błąd bezwzględny z2", color="plum", linewidth=1.3)

plt.title("Błędy bezwzględne przybliżeń Padégo", fontsize=12, fontweight='bold', color="dimgray")
plt.xlabel("x", fontsize=10, fontweight='bold', color="dimgray")
plt.ylabel("Wartość błędu bezwzględnego", fontsize=10, fontweight='bold', color="dimgray")
plt.gca().set_facecolor('snow')
plt.legend()
plt.grid(True)

# Zapis wykresu jako plik PNG
plt.savefig('bledy_przybliżeń_padego.png', dpi=300, bbox_inches='tight')

# Wyświetlenie wykresu
plt.show()


# Zad2
def machine_epsilon():
    """
    Funkcja wyznacza dokładność maszynową (epsilon maszynowy),
    która jest najmniejszą liczbą dodatnią, która po dodaniu do 1.0 daje wynik różny od 1.0.

    Zwraca:
    float: epsilon maszynowy.
    """
    epsilon = 1.0  # Epsilon jako 1
    # Pętla zmniejszająca epsilon aż do momentu, kiedy 1.0 + epsilon == 1.0
    while (1.0 + epsilon) != 1.0:
        epsilon /= 2  # Zmniejszenie epsilon o połowę
    return epsilon * 2  # Zwracamy epsilon pomnożone przez 2, ponieważ ostatnia zmiana była za duża


# Wyliczenie dokładności maszynowej
epsilon = machine_epsilon()
print("Zad 2: ", "Dokładność maszynowa:", epsilon)

# Zad3
odp1 = []
for i in range(1, 51, 1):
    print(i, i/100 * 100 - i)
    if i / 100 * 100 - i != 0.0:
        odp1.append(i)

print("Zad 3", odp1)
# zad 4
print("Zad 4")
def bl_bez(x, x0):
    return fabs(x - x0)

def bl_wzgl(x, x0):
    d = bl_bez(x, x0)
    return d/x
x=1.7
# Dokładna wartość liczby
dokladna_wartosc = 1.7

# Przybliżona wartość liczby w standardzie IEEE
przyblizona_wartosc = 1.6999999284744263

# Obliczanie błędu bezwzględnego
blad_bezwzgledny = abs(dokladna_wartosc - przyblizona_wartosc)

# Obliczanie błędu względnego
blad_wzgledny = blad_bezwzgledny / dokladna_wartosc

# Wyświetlenie wyników
print(f'Błąd bezwzględny: {blad_bezwzgledny}')
print(f'Błąd względny: {blad_wzgledny}')
# zad 5
print("Zad 5")
# Funkcja wielomianu
def wiel(x):
    return 6*x**4 + 5*x**3 - 13*x**2 + x + 1

# Funkcja, którą chcemy profilować
def main():
    odp = []
    for i in np.arange(-10, 10, 0.0001):
        odp.append(wiel(i))

# Uruchomienie profilowania
cProfile.run('main()')


def W(x):
    return 6 * x ** 4 + 5 * x ** 3 - 13 * x ** 2 + x + 1


def obliczenia():
    start_time = time.time()
    x_wartosci = np.arange(-10, 10, 0.0001)
    w_wartosci = [W(x) for x in x_wartosci]

    with open("wielomian_wyniki.txt", "w") as file:
        for x, w in zip(x_wartosci, w_wartosci):
            file.write(f'W({x:.4f}) = {w:.4f}\n')

    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time:.4f} sekundy")


obliczenia()
# Zad6
print("Zad 6")
# Funkcja Hornera
def horner(poly, x):
    """
    Funkcja oblicza wartość wielomianu w punkcie x za pomocą schematu Hornera.

    Parametry:
    - poly (list): Współczynniki wielomianu w kolejności od najwyższego stopnia.
    - x (float): Punkt, w którym obliczamy wartość wielomianu.

    Zwraca:
    - result (float): Wartość wielomianu dla danego x.
    """
    result = poly[0]
    for i in range(1, len(poly)):
        result = result * x + poly[i]
    return result


# Funkcja główna wykonująca obliczenia dla przedziału -10 do 10
def main1():
    """
    Funkcja iteruje po wartościach z przedziału [-10, 10] ze skokiem 0.0001 i oblicza wartości wielomianu
    za pomocą funkcji horner(). Wyniki są przechowywane w liście 'odp'.
    """
    odp = []
    poly = [6, 5, -13, 1, 1]  # Współczynniki wielomianu

    # Pętla obliczająca wartości wielomianu dla każdego x z przedziału [-10, 10]
    for i in np.arange(-10, 10, 0.0001):
        odp.append(horner(poly, i))


# Profilowanie wydajności
cProfile.run('main1()')


# Funkcja wektorowa obliczająca wielomian w wielu punktach jednocześnie
def W_horner(x_wartosci):
    """
    Funkcja oblicza wartości wielomianu dla wielu punktów jednocześnie za pomocą schematu Hornera.

    Parametry:
    - x_wartosci (np.array): Tablica NumPy z wartościami x, dla których chcemy obliczyć wartość wielomianu.

    Zwraca:
    - result (np.array): Tablica NumPy z obliczonymi wartościami wielomianu dla każdego punktu z x_wartosci.
    """
    a = np.array([6, 5, -13, 1, 1])  # Współczynniki wielomianu
    result = a[0] * np.ones_like(x_wartosci)
    for coefficient in a[1:]:
        result = result * x_wartosci + coefficient
    return result


# Funkcja obliczająca wartości wielomianu dla przedziału [-10, 10] i zapisująca wyniki do pliku
def obliczenia_horner():
    """
    Funkcja oblicza wartości wielomianu dla przedziału od -10 do 10 ze skokiem 0.0001,
    zapisuje wyniki do pliku oraz mierzy czas wykonania obliczeń.
    """
    start_time = time.time()  # Rozpoczęcie pomiaru czasu

    # Tablica wartości x z przedziału [-10, 10] ze skokiem 0.0001
    x_wartosci = np.arange(-10, 10, 0.0001)

    # Obliczanie wartości wielomianu dla każdego x
    w_wartosci = W_horner(x_wartosci)

    # Zapis wyników do pliku
    with open("wielomian_wyniki_horner.txt", "w") as file:
        for x, w in zip(x_wartosci, w_wartosci):
            file.write(f'W({x:.4f}) = {w:.4f}\n')

    end_time = time.time()  # Zakończenie czasu
    print(f"Czas wykonania działania sposobem schematu Hornera: {end_time - start_time:.4f} sekundy")


# Uruchomienie obliczeń
obliczenia_horner()
