import numpy as np

EPS = 1e-10


def pivot_tableau(tableau, pivot_row_index, pivot_col_index):
    pivot_value = tableau[pivot_row_index, pivot_col_index]
    tableau[pivot_row_index, :] = tableau[pivot_row_index, :] / pivot_value

    row_count = tableau.shape[0]
    for row_index in range(row_count):
        if row_index != pivot_row_index:
            factor = tableau[row_index, pivot_col_index]
            if abs(factor) > EPS:
                tableau[row_index, :] = tableau[row_index, :] - factor * tableau[pivot_row_index, :]


def choose_entering_variable_max(tableau):
    objective_row = tableau[-1, :-1]
    entering_col_index = int(np.argmin(objective_row))
    min_value = float(objective_row[entering_col_index])

    is_optimal = True
    if min_value < -1e-12:
        is_optimal = False

    return {
        "is_optimal": is_optimal,
        "entering_col_index": entering_col_index
    }


def choose_leaving_variable(tableau, entering_col_index):
    constraint_rows = tableau[:-1, :]
    rhs_column = constraint_rows[:, -1]
    entering_column = constraint_rows[:, entering_col_index]

    leaving_row_index = -1
    best_ratio = None

    for row_index in range(constraint_rows.shape[0]):
        coefficient = entering_column[row_index]
        if coefficient > EPS:
            ratio = rhs_column[row_index] / coefficient
            if (best_ratio is None or
                    ratio < best_ratio - 1e-12 or
                    (abs(ratio - best_ratio) <= 1e-12 and row_index < leaving_row_index)):
                best_ratio = ratio
                leaving_row_index = row_index

    is_unbounded = leaving_row_index == -1
    return {
        "is_unbounded": is_unbounded,
        "leaving_row_index": leaving_row_index
    }


def simplex_maximize(tableau, basis_variable_indices, max_iterations=1000):
    status = "iter_limit"
    iteration = 0

    while iteration < max_iterations:
        entering_info = choose_entering_variable_max(tableau)

        if entering_info["is_optimal"]:
            status = "optimal"
            break

        entering_col_index = entering_info["entering_col_index"]
        leaving_info = choose_leaving_variable(tableau, entering_col_index)

        if leaving_info["is_unbounded"]:
            status = "unbounded"
            break

        leaving_row_index = leaving_info["leaving_row_index"]
        pivot_tableau(tableau, leaving_row_index, entering_col_index)
        basis_variable_indices[leaving_row_index] = entering_col_index

        iteration += 1

    return {
        "status": status,
        "tableau": tableau,
        "basis": basis_variable_indices
    }


def extract_solution(tableau, basis_variable_indices, variable_count):
    solution = np.zeros(variable_count, dtype=float)
    constraint_count = tableau.shape[0] - 1

    for row_index in range(constraint_count):
        basis_var = basis_variable_indices[row_index]
        if basis_var < variable_count:
            solution[basis_var] = tableau[row_index, -1]

    return {
        "x": solution,
        "z": float(tableau[-1, -1])
    }


def solve_big_m(A_matrix, rhs_vector, objective_cost, big_m_value=1e6):
    status = "unknown"
    solution_vector = None
    objective_value = None

    constraint_count, variable_count = A_matrix.shape
    artificial_identity = np.eye(constraint_count)

    augmented_matrix = np.hstack([A_matrix.astype(float), artificial_identity])
    augmented_cost = np.concatenate([
        objective_cost.astype(float),
        -big_m_value * np.ones(constraint_count)
    ])

    total_variable_count = variable_count + constraint_count

    tableau = np.zeros((constraint_count + 1, total_variable_count + 1), dtype=float)
    tableau[:constraint_count, :total_variable_count] = augmented_matrix
    tableau[:constraint_count, -1] = rhs_vector.astype(float)

    basis_variable_indices = [variable_count + i for i in range(constraint_count)]

    basis_costs = augmented_cost[basis_variable_indices]
    tableau[-1, :total_variable_count] = (basis_costs @ tableau[:constraint_count, :total_variable_count]) - augmented_cost
    tableau[-1, -1] = basis_costs @ tableau[:constraint_count, -1]

    simplex_result = simplex_maximize(tableau, basis_variable_indices)

    status = simplex_result["status"]
    final_tableau = simplex_result["tableau"]
    final_basis = simplex_result["basis"]

    if status == "optimal":
        extended_solution = np.zeros(total_variable_count, dtype=float)
        for row_index in range(constraint_count):
            extended_solution[final_basis[row_index]] = final_tableau[row_index, -1]

        artificial_part = extended_solution[variable_count:]
        infeasible = np.max(np.abs(artificial_part)) > 1e-8

        if infeasible:
            status = "infeasible"
        else:
            solution_vector = extended_solution[:variable_count]
            objective_value = float(objective_cost @ solution_vector)

    return {
        "status": status,
        "x": solution_vector,
        "z": objective_value
    }


def main():
    # Вариант 8
    A_matrix = np.array([
        [1, 5, -3, -4, 2, 1],
        [2, 9, -5, -7, 4, 2]
    ], dtype=float)

    rhs_vector = np.array([14, 32], dtype=float)

    # z = x1 - 3x2 + 4x3 + 5x4 - x5 + 8x6 -> max
    objective_cost = np.array([1, -3, 4, 5, -1, 8], dtype=float)

    result = solve_big_m(A_matrix, rhs_vector, objective_cost, big_m_value=1e6)

    print("Метод больших штрафов")
    print("Статус:", result["status"])
    if result["status"] == "optimal":
        print("x:", np.round(result["x"], 6))
        print("z:", round(result["z"], 6))


if __name__ == "__main__":
    main()