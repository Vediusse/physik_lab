import numpy as np
from typing import Tuple, Dict, List
from numpy.linalg import LinAlgError
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import cholesky, cho_factor, cho_solve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc
from typing import Callable, Dict, Any
import numpy as np
import subprocess


class SolverMetrics:
    def __init__(self, matrix: np.ndarray, execution_time: float, memory_usage: int, error: bool = False):
        self.matrix = matrix
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.error = error


## Методы СЛАУ

def gauss_elimination(A, b):
    n = len(b)
    # Augment matrix A with vector b
    Augmented = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Augmented[i:n, i])) + i
        Augmented[[i, max_row]] = Augmented[[max_row, i]]

        for j in range(i + 1, n):
            factor = Augmented[j, i] / Augmented[i, i]
            Augmented[j, i:] -= factor * Augmented[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Augmented[i, -1] - np.dot(Augmented[i, i + 1:n], x[i + 1:])) / Augmented[i, i]

    return x


def gauss_elimination_pivoting(A, b):
    n = len(b)
    Augmented = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        max_row = np.argmax(np.abs(Augmented[i:n, i])) + i
        Augmented[[i, max_row]] = Augmented[[max_row, i]]

        for j in range(i + 1, n):
            factor = Augmented[j, i] / Augmented[i, i]
            Augmented[j, i:] -= factor * Augmented[i, i:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Augmented[i, -1] - np.dot(Augmented[i, i + 1:n], x[i + 1:])) / Augmented[i, i]

    return x


def cholesky_algorithm(A, b):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)

    return np.linalg.solve(L.T, y).flatten()


import numpy as np


def thomas_algorithm(A, b):
    """Метод Томаса для трехдиагональной матрицы"""
    n = len(b)
    a = np.array([A[i, i - 1] for i in range(1, n)], dtype=np.float64)
    b_diag = np.array([A[i, i] for i in range(n)], dtype=np.float64)
    c = np.array([A[i, i + 1] for i in range(n - 1)], dtype=np.float64)
    d = np.array(b, dtype=np.float64)

    
    for i in range(1, n):
        factor = a[i - 1] / b_diag[i - 1]
        b_diag[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    
    x = np.zeros(n, dtype=np.float64)
    x[-1] = d[-1] / b_diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b_diag[i]

    return x


def simple_iteration_jacobi(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Solve the system Ax = b using the Jacobi (simple iteration) method.

    Parameters:
        A       : numpy.ndarray, coefficient matrix (should be diagonally dominant for convergence)
        b       : numpy.ndarray, right-hand side vector
        x0      : numpy.ndarray, initial guess for the solution (default is zero vector)
        tol     : float, tolerance for convergence (stopping criterion on infinity norm of the change)
        max_iter: int, maximum number of iterations allowed

    Returns:
        x       : numpy.ndarray, approximate solution vector
        k       : int, number of iterations performed
    """
    n = A.shape[0]

    # Initial guess (if not provided)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Extracting Diagonal D and remainder R from A
    D = np.diag(A)
    if np.any(np.isclose(D, 0)):
        raise ValueError("Zero detected on the diagonal. The Jacobi method requires non-zero diagonal entries.")

    # Pre-calculate the inverse of the diagonal for efficient vectorized updates
    invD = 1.0 / D
    R = A - np.diagflat(D)

    # Iterative update
    for k in range(max_iter):
        # Update rule: x_new = invD * (b - R*x)
        x_new = invD * (b - np.dot(R, x))

        # Check for convergence based on infinity norm
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new  # converged

        x = x_new.copy()

    # If convergence was not reached within max_iter iterations, return the last iterate.
    return x


def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Solve the system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
        A        : numpy.ndarray, coefficient matrix (n x n), assumed to be diagonally dominant or SPD.
        b        : numpy.ndarray, right-hand side vector (n,)
        x0       : numpy.ndarray, initial guess (default: zero vector)
        tol      : float, tolerance for convergence (infinity norm)
        max_iter : int, maximum number of iterations

    Returns:
        x        : numpy.ndarray, approximate solution vector

    """
    n = A.shape[0]

    # Initial guess for the solution vector
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Iterative update
    for k in range(max_iter):
        x_new = x.copy()  # Create a copy for the new iterate

        for i in range(n):
            # Update each x[i] using the latest available values:
            # Use the new values for x[0:i] and the old values for x[i+1:n].
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        # Check for convergence using the infinity norm (max absolute change)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new.copy()

    return x  # If convergence is not reached within max_iter iterations


## ДАЛЬШЕ ИДЕТ ПРИКЛАДНОЕ РЕШЕНИЕ ФИЗИЧЕСКОЙ ЗАДАЧИ
def generate_grid(A: float, B: float, div1: int, div2: int) -> Tuple[
    List[Tuple[float, float]], Dict[Tuple[int, int], int], float, float]:
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


def construct_equation_matrix(n: int, div1: int, div2: int, A: float, dx: float, dy: float,
                              points: List[Tuple[float, float]], index_map: Dict[Tuple[int, int], int]) -> Tuple[
    np.ndarray, np.ndarray]:
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


def solve_heat_equation(
        A: float, B: float, n: int, div1: int, div2: int,
        solvers: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]
) -> Dict[str, SolverMetrics]:
    """
    Решает уравнение теплопроводности на сетке методом конечных разностей.

    :param float A: Ширина области
    :param float B: Высота области
    :param int n: Общее число узлов сетки
    :param int div1: Количество делений по горизонтали
    :param int div2: Количество делений по вертикали
    :param list solvers: Список функций для решения СЛАУ
    :return: Словарь с решениями, временем выполнения и потреблением памяти для каждого метода
    :rtype: Dict[str, SolverMetrics]
    :raises AssertionError: Если n не равно div1 * div2
    """
    assert div1 * div2 == n, "Число n должно быть произведением div1 и div2"

    points, index_map, dx, dy = generate_grid(A, B, div1, div2)
    A_matrix, b_vector = construct_equation_matrix(n, div1, div2, A, dx, dy, points, index_map)

    print("First 5 rows of A:\n", A_matrix[:5, :5])
    print("First 5 elements of b:\n", b_vector[:5])

    results = {}

    for solver_name, solver in solvers.items():
        A_matrix, b_vector = construct_equation_matrix(n, div1, div2, A, dx, dy, points, index_map)
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            T_vector = np.array(solver(A_matrix, b_vector)).astype(float).flatten()

            end_time = time.perf_counter()
            mem_usage = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            T_matrix = np.zeros((div1, div2))
            for i in range(1, div1 + 1):
                for j in range(1, div2 + 1):
                    idx = index_map[(i, j)]
                    T_matrix[i - 1, j - 1] = T_vector[idx]

            results[solver_name] = SolverMetrics(
                matrix=T_matrix,
                execution_time=end_time - start_time,
                memory_usage=mem_usage[1]
            )
            print(T_matrix)
            print("\n\n\n\n")
        except LinAlgError:

            end_time = time.perf_counter()
            mem_usage = tracemalloc.get_traced_memory()
            results[solver_name] = SolverMetrics(
                matrix=np.zeros((div1, div2)),
                execution_time=end_time - start_time,
                memory_usage=mem_usage[1],
                error=True
            )

    return results


def generate_latex(A: float, B: float, div1: int, div2: int, T: np.ndarray, points: List[Tuple[float, float]]):
    matrix_str = " \\\\\n".join(" & ".join(f"{T[j, div1 - i - 1]:.2f}" for j in range(div2)) for i in range(div1))

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
            "\t\t\t" + rf"\filldraw[{color}] ({x},{y}) circle (2pt);"  # node[above] {{${temp:.2f}^\circ C$}}
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


def plot_solver_metrics(results: Dict[str, SolverMetrics]):
    num_methods = len(results)


    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5))

    if num_methods == 1:
        axes = [axes]

    for ax, (solver_name, metrics) in zip(axes, results.items()):
        rotated_matrix = np.rot90(metrics.matrix)
        sns.heatmap(rotated_matrix, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title(f"Heatmap: {solver_name}")

    plt.tight_layout()
    plt.show()

    
    solver_names = list(results.keys())
    execution_times = [metrics.execution_time for metrics in results.values()]

    plt.figure(figsize=(6, 4))
    plt.bar(solver_names, execution_times, color="skyblue")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time by Solver")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.5)
    plt.show()

    memory_usages = [metrics.memory_usage for metrics in results.values()]

    plt.figure(figsize=(6, 4))
    plt.bar(solver_names, memory_usages, color="lightcoral")
    plt.ylabel("Memory Usage (bytes)")
    plt.title("Memory Usage by Solver")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.5)
    plt.show()


A = 30
B = 2
n = 256
div1 = 16
div2 = 16
points, index_map, dx, dy = generate_grid(A, B, div1, div2)
solvers = {gauss_elimination.__name__: gauss_elimination, thomas_algorithm.__name__: thomas_algorithm}
T = solve_heat_equation(A, B, n, div1, div2, solvers=solvers)
plot_solver_metrics(T)
generate_latex(A, B, div1, div2, T.get("gauss_elimination").matrix, points)
