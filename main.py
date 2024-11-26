import matplotlib
import matplotlib.pyplot as plt

from interpolator import NumericalIntegrator

matplotlib.use('TkAgg')

PARTICLE_MASS = 4.65e-26

# Данные
ENERGY_EV = [
    6.728, 7.000, 8.000, 8.012, 8.378, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40,
    45, 50, 60, 70, 80, 90, 100, 120, 140, 165, 200, 250, 300, 400, 500, 600, 700,
    800, 900, 1000, 1500, 2000, 3000, 4000, 5000
]

CROSS_SECTION = [
    0, 0, 0, 0, 0, 0.4082, 1.0473, 3.0237, 4.5666, 5.7550, 6.6567, 7.3377, 8.3879,
    8.8769, 9.0630, 9.0790, 8.9971, 8.8587, 8.5228, 8.1615, 7.7992, 7.4527, 7.1281,
    6.5483, 6.0533, 5.5328, 4.9448, 4.3043, 3.8208, 3.1372, 2.6746, 2.3391, 2.0837,
    1.8822, 1.7189, 1.5835, 1.1475, 0.9085, 0.6498, 0.5105, 0.4227
]

if __name__ == "__main__":
    integrator = NumericalIntegrator(ENERGY_EV, CROSS_SECTION)

    temperature_range = [1 + i * (10000 - 0) / 1000 for i in range(1000)]

    results = [integrator.integrate(temperature, PARTICLE_MASS) for temperature in temperature_range]

    plt.figure(figsize=(8, 6))
    plt.plot(temperature_range, results, label='R(T)', color='green')

    plt.xlabel('Температура (eV)')
    plt.ylabel('R(T)')
    plt.title('Зависимость R(T) от температуры')
    plt.grid(True)
    plt.legend()
    plt.show()  # Открываем Шампанское
