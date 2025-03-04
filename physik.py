import numpy as np
from typing import Tuple, Dict, List
import subprocess



def generate_grid(A: float, B: float, div1: int, div2: int) -> Tuple[List[Tuple[float, float]], Dict[Tuple[int, int], int], float, float]:
    """
    Генерирует сетку точек внутри прямоугольной области.

    :param float A: Ширина области
    :param float B: Высота области
    :param int div1: Количество делений по горизонтали
    :param int div2: Количество делений по вертикали
    :return: Кортеж, содержащий список точек (x, y), словарь индексов точек, шаг по x и шаг по y
    :rtype: Tuple[List[Tuple[float, float]], Dict[Tuple[int, int], int], float, float]
    """
    dx = A / (div1 + 1)
    dy = B / (div2 + 1)
    points = []
    index_map = {}
    eq_index = 0

    for i in range(1, div1 + 1):
        for j in range(1, div2 + 1):
            x = i * dx
            y = j * dy
            points.append((x, y))
            index_map[(i, j)] = eq_index
            eq_index += 1

    return points, index_map, dx, dy


def apply_boundary_conditions(idx: int, neighbor_idx: int, value: float, coef: float,
                              A_matrix: np.ndarray, b_vector: np.ndarray):
    """
    Обрабатывает граничные условия в матрице системы.

    :param int idx: Индекс текущей точки
    :param int neighbor_idx: Индекс соседней точки (или None, если граница)
    :param float value: Значение граничного условия
    :param float coef: Коэффициент при разностной схеме
    :param np.ndarray A_matrix: Матрица коэффициентов уравнения
    :param np.ndarray b_vector: Вектор правой части
    """
    if neighbor_idx is not None:
        A_matrix[idx, neighbor_idx] = coef
    else:
        b_vector[idx] -= value * coef


def construct_equation_matrix(n: int, div1: int, div2: int, A: float, dx: float, dy: float, points: List[Tuple[float, float]], index_map: Dict[Tuple[int, int], int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создаёт матрицу системы и вектор правой части для уравнения теплопроводности.

    :param int n: Общее число узлов сетки
    :param int div1: Количество делений по горизонтали
    :param int div2: Количество делений по вертикали
    :param float A: Ширина области
    :param float dx: Шаг по x
    :param float dy: Шаг по y
    :param List[Tuple[float, float]] points: Список координат узлов сетки
    :param Dict[Tuple[int, int], int] index_map: Словарь соответствия индексов точек
    :return: Кортеж (матрица коэффициентов, вектор правой части)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    A_matrix = np.zeros((n, n))
    b_vector = np.zeros(n)

    coef_x = 1 / dx ** 2
    coef_y = 1 / dy ** 2
    diag_coef = -2 * (coef_x + coef_y)

    for (i, j), idx in index_map.items():
        x, _ = points[idx]
        A_matrix[idx, idx] = diag_coef

        apply_boundary_conditions(idx, index_map.get((i - 1, j)), 0, coef_x, A_matrix, b_vector)
        apply_boundary_conditions(idx, index_map.get((i + 1, j)), A, coef_x, A_matrix, b_vector)
        apply_boundary_conditions(idx, index_map.get((i, j - 1)), x, coef_y, A_matrix, b_vector)
        apply_boundary_conditions(idx, index_map.get((i, j + 1)), (x ** 2) / A, coef_y, A_matrix, b_vector)

    return A_matrix, b_vector


def solve_heat_equation(A: float, B: float, n: int, div1: int, div2: int) -> np.ndarray:
    """
    Решает уравнение теплопроводности на сетке методом конечных разностей.

    :param float A: Ширина области
    :param float B: Высота области
    :param int n: Общее число узлов сетки
    :param int div1: Количество делений по горизонтали
    :param int div2: Количество делений по вертикали
    :return: Матрица температурных значений в узлах сетки
    :rtype: np.ndarray
    :raises AssertionError: Если n не равно div1 * div2
    """
    assert div1 * div2 == n, "Число n должно быть произведением div1 и div2"

    points, index_map, dx, dy = generate_grid(A, B, div1, div2)
    A_matrix, b_vector = construct_equation_matrix(n, div1, div2, A, dx, dy, points, index_map)
    T_vector = np.linalg.solve(A_matrix, b_vector)

    T_matrix = np.zeros((div1, div2))
    for i in range(1, div1 + 1):
        for j in range(1, div2 + 1):
            idx = index_map[(i, j)]
            T_matrix[i - 1, j - 1] = T_vector[idx]

    return T_matrix


def generate_latex(A: float, B: float, div1: int, div2: int, T: np.ndarray, points: List[Tuple[float, float]]):
    matrix_str = " \\\\\n".join(" & ".join(f"{T[j, div1-i-1]:.2f}" for j in range(div2)) for i in range(div1))

    colors = [
        "violet",  # Фиолетовый (минимальная температура)
        "blue",  # Синий
        "cyan",  # Голубой
        "green",  # Зелёный
        "yellow",  # Жёлтый
        "orange",  # Оранжевый
        "red"  # Красный (максимальная температура)
    ]


    points_tikz = []
    for idx, (x, y) in enumerate(points):
        i = idx // div2
        j = idx % div2
        temp = T[i, j]

        color_index = int((temp / A) * (len(colors) - 1))
        color_index = min(max(color_index, 0), len(colors) - 1)
        color = colors[color_index]

        points_tikz.append(
            "\t\t\t"+ rf"\filldraw[{color}] ({x},{y}) circle (2pt);" #node[above] {{${temp:.2f}^\circ C$}}
        )
    points_tikz = "\n".join(points_tikz)

    tex_content = rf"""
    \documentclass[10pt]{{article}}
    \usepackage[T1,T2A]{{fontenc}}
    \usepackage[english, russian]{{babel}}
    \usepackage[left=1cm, right=1cm, top=1cm]{{geometry}}
    \usepackage{{amsmath}}
    \usepackage{{graphicx}}
    \usepackage{{tikz}}
    \usepackage{{enumitem}} % Подключаем пакет для кастомизации списков

    \title{{Задание по физике номер 1}}
    \author{{Рублёв Валерий}}

    \begin{{document}}
    \maketitle

    \section{{Условие задачи}}

    \subsection{{Задача}}
    Есть двухмерное тело прямоугольной формы:

    \begin{{enumerate}}
        \item Пользователь задает числа $A$ и $B$ - размеры прямоугольника.
        \item Пользователь задает число $n$.
        \item Пользователь задает два числа, которые являются делителями числа $n$ и при умножении дают $n$.
    \end{{enumerate}}
    
    Тело по краям имеет такое значение температуры:\
    
    \begin{{enumerate}}[label=\textbullet] % Указываем маркер (•)
        \item Слева: температура $0^\circ C$.
        \item Справа: температура $A^\circ C$.
        \item Снизу: температура $x$.
        \item Сверху: температура $\frac{{x^2}}{{A}}$.
    \end{{enumerate}}

    Задача: вычислить температуру в $n$ точках внутри тела.

    \subsection{{Аналитичесое описание тела}}
    Дано прямоугольное тело размерами $A={A}$ м, $B={B}$ м, создадим равномерное разбиение по осям: {div1} по $x$, {div2} по $y$.

    \subsection{{Построение тела}}
    Используется сетка конечных разностей.

    \begin{{center}}
        \begin{{tikzpicture}}[scale=1]
            % Координатные оси
            \draw[->] (-0.5, 0) -- ({A + 2}, 0) node[below] {{$x$}};
            \draw[->] (0, -0.5) -- (0, {B + 2}) node[left] {{$y$}};

            % Прямоугольник
            \draw[thick] (0,0) rectangle ({A}, {B});

            % Подписи вершин
            \node[below left] at (0,0) {{$ (0,0) $}};
            \node[below] at ({A},0) {{$ ({A},0) $}};
            \node[left] at (0,{B}) {{$ (0,{B}) $}};
            \node[above right] at ({A},{B}) {{$ ({A},{B}) $}};\

            % Подписи температур
            \node[left] at (-0.25,{B / 2}) {{$ 0^\circ C $}}; % Слева
            \node[right] at ({A + 0.25},{B / 2}) {{$ A^\circ C $}}; % Справа
            \node[below] at ({A / 2},-0.25) {{$ x $}}; % Снизу
            \node[above] at ({A / 2},{B + 0.25}) {{$ \frac{{x^2}}{{A}} $}}; % Сверху

            
            
        \end{{tikzpicture}}
    \end{{center}}

    \section{{Решение задачи}}
    \subsection{{Описание решения и алгоритма}}
    Разностная схема основана на уравнении $\frac{{dT}}{{dx}} + \frac{{dT}}{{dy}} = 0$. Используется метод решения СЛАУ.

    \subsection{{Подстановка значений}}
    Полученная температурная матрица:
    \begin{{equation}}
    T =
    \begin{{bmatrix}}
    {matrix_str}
    \end{{bmatrix}}
    \end{{equation}}

    \subsection{{Ответ}}
    
    
    \begin{{center}}
        \begin{{tikzpicture}}[scale=1]
            % Координатные оси
            \draw[->] (-0.5, 0) -- ({A + 2}, 0) node[below] {{$x$}};
            \draw[->] (0, -0.5) -- (0, {B + 2}) node[left] {{$y$}};

            % Прямоугольник
            \draw[thick] (0,0) rectangle ({A}, {B});

            % Подписи вершин
            \node[below left] at (0,0) {{$ (0,0) $}};
            \node[below] at ({A},0) {{$ ({A},0) $}};
            \node[left] at (0,{B}) {{$ (0,{B}) $}};
            \node[above right] at ({A},{B}) {{$ ({A},{B}) $}};\

            % Подписи температур
            \node[left] at (-0.25,{B / 2}) {{$ 0^\circ C $}}; % Слева
            \node[right] at ({A + 0.25},{B / 2}) {{$ A^\circ C $}}; % Справа
            \node[below] at ({A / 2},-0.25) {{$ x $}}; % Снизу
            \node[above] at ({A / 2},{B + 0.25}) {{$ \frac{{x^2}}{{A}} $}}; % Сверху

            % Точки сетки
            {points_tikz}
        \end{{tikzpicture}}
    \end{{center}}

    \end{{document}}
    """

    with open("report.tex", "w", encoding="utf-8") as file:
        file.write(tex_content)





A = 7
B = 2
n = 1024
div1 = 32
div2 = 32
points, index_map, dx, dy = generate_grid(A, B, div1, div2)
T = solve_heat_equation(A, B, n, div1, div2)
generate_latex(A, B, div1, div2, T, points)

print(T)
