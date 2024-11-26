from typing import Callable

from interpolation import Interpolator

import math

K_B = 1  # Постоянная Больцмана, Дж/К
EV_TO_J = 1  # Перевод из eV в Дж

maxwell_energy_distribution: Callable[[float, float], float] = lambda energy, temperature: (
        (2 / (K_B * temperature)) ** 1.5 *
        math.sqrt(energy / math.pi) *
        math.exp(-energy / (K_B * temperature))
)

velocity: Callable[[float, float], float] = lambda energy, mass: (
    math.sqrt(2 * energy * EV_TO_J / mass)
)


class NumericalIntegrator:
    """
    Класс для численного интегрирования функции для вычисления R(T).
    """

    def __init__(self, energies: list[float], cross_sections: list[float], max_energy: float = 10000,
                 steps: int = 10000) -> None:
        """
        Инициализация интегратора.

        :param energies: Список энергий (в эВ).
        :param cross_sections: Список значений сечений (в å²).
        :param max_energy: Максимальная энергия для интегрирования.
        :param steps: Количество шагов интегрирования.
        """
        self.energies = energies
        self.cross_sections = cross_sections
        self.max_energy = max_energy
        self.steps = steps
        self.interpolator = Interpolator(energies, cross_sections)

    def integrate(self, temperature: float, mass: float) -> float:
        """
        Выполняет численное интегрирование для вычисления R(T).

        :param temperature: Температура в К.
        :param mass: Масса частицы в кг.
        :return: Результат численного интеграла для R(T).
        """
        energies = [1 + i * (self.max_energy - 0) / (self.steps - 1) for i in range(self.steps)]
        energy_step = energies[1] - energies[0]

        integral = sum(
            self.interpolator.interpolate(energy) *
            velocity(energy, mass) *
            maxwell_energy_distribution(energy, temperature) *
            energy_step
            for energy in energies
        )

        return integral

