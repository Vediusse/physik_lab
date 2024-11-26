import math
import matplotlib.pyplot as plt


class Interpolator:
    """
    Класс для интерполяции значений с использованием линейной интерполяции и экстраполяции.
    """

    def __init__(self, energies: list[float], cross_sections: list[float]) -> None:
        """
        Инициализация интерполятор.

        :param energies: Список энергий (в эВ).
        :param cross_sections: Список значений сечений (в å²).
        """
        self.energies = energies
        self.cross_sections = cross_sections
        self.cache = {energy: cs for energy, cs in zip(energies, cross_sections)}

    def interpolate(self, energy: float) -> float:
        if energy in self.cache:
            return self.cache[energy]

        if energy < min(self.cache.keys()):
            result = self.extrapolate_left(energy)

        elif energy > max(self.cache.keys()):
            result = self.extrapolate_right(energy)
        else:
            result = self.linear_interpolation(energy)

        self.cache[energy] = result
        return result

    def linear_interpolation(self, energy: float) -> float:
        """
        Осуществляет линейную интерполяцию сечений.

        :param energy: Энергия для интерполяции.
        :return: Интерполированное значение сечения.
        """
        sorted_keys = sorted(self.cache.keys())
        lower_idx, upper_idx = self.find_closest_indices(sorted_keys, energy)
        x1, x2 = sorted_keys[lower_idx], sorted_keys[upper_idx]
        y1, y2 = self.cache[x1], self.cache[x2]

        return y1 + (y2 - y1) * (energy - x1) / (x2 - x1)

    def extrapolate_left(self, energy: float) -> float:
        """
        Экстраполирует значение сечения влево от минимальной энергии.

        :param energy: Энергия для экстраполяции.
        :return: Экстраполированное значение сечения.
        """
        min_energy = min(self.cache.keys())
        min_cs = self.cache[min_energy]
        factor = math.exp(-(min_energy - energy) / min_energy)
        return min_cs * factor

    def extrapolate_right(self, energy: float) -> float:
        """
        Экстраполирует значение сечения вправо от максимальной энергии.

        :param energy: Энергия для экстраполяции.
        :return: Экстраполированное значение сечения.
        """
        max_energy = max(self.cache.keys())
        max_cs = self.cache[max_energy]
        factor = math.exp(-(energy - max_energy) / max_energy)
        return max_cs * factor

    @staticmethod
    def find_closest_indices(sorted_keys: list[float], energy: float) -> tuple[int, int]:
        """
        Находит индексы ближайших значений энергии в отсортированном списке.

        :param sorted_keys: Отсортированный список энергий.
        :param energy: Энергия для поиска ближайших значений.
        :return: Индексы ближайших значений энергии.
        """
        for idx in range(len(sorted_keys) - 1):
            if sorted_keys[idx] <= energy < sorted_keys[idx + 1]:
                return idx, idx + 1
        raise ValueError("Энергия находится вне диапазона доступных значений.")


if __name__ == "__main__":
    ENERGY_EV = [
        6.738, 7.000, 8.000, 8.012, 8.378, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40,
        45, 50, 60, 70, 80, 90, 100, 120, 140, 165, 200, 250, 300, 400, 500, 600, 700,
        800, 900, 1000, 1500, 2000, 3000, 4000, 5000
    ]

    CROSS_SECTION = [
        0, 0, 0, 0, 0, 0.4182, 1.0473, 3.0237, 4.5666, 5.7550, 6.6567, 7.3377, 8.3879,
        8.8769, 9.0630, 9.0790, 8.9971, 8.8587, 8.5228, 8.1615, 7.7992, 7.4527, 7.1281,
        6.5483, 6.0533, 5.5328, 4.9448, 4.3043, 3.8208, 3.1372, 2.6746, 2.3391, 2.0837,
        1.8822, 1.7189, 1.5835, 1.1475, 0.9085, 0.6498, 0.5105, 0.4227
    ]

    interpolator = Interpolator(ENERGY_EV, CROSS_SECTION)

    energy_range = [i for i in range(1, 10001)]
    cross_sections_interpolated = [interpolator.interpolate(e) for e in energy_range]

    plt.figure(figsize=(10, 6))
    plt.plot(energy_range, cross_sections_interpolated, label="Interpolated/Extrapolated Cross-Section")
    plt.xscale('log')
    plt.yscale('log', base=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Cross Section (Å²)')
    plt.title('Extrapolation and Interpolation of Cross Section')
    plt.legend()
    plt.grid(True)
    plt.show()
