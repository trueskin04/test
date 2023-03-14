from math import asin
from math import pow
from math import floor
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROUND_CONSTANT = 6

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', ('{:.' + str(ROUND_CONSTANT) + 'f}').format)

a = -0.5
b = 0.5

MAX_F_DIFF_2 = 3.47
MAX_F_DIFF_3 = 8.56
MAX_F_DIFF_4 = 47.05

m = 54
E = pow(2, -m)


def f(x):
    return pow(asin(x), 2)


f = np.vectorize(f)


def f_diff(x):
    return (2 * asin(x)) / (sqrt(1 - pow(x, 2)))


f_diff = np.vectorize(f_diff)


def f_diff_2(x):
    return 2 / (1 - pow(x, 2)) + (2 * x * asin(x)) / ((1 - pow(x, 2)) * sqrt(1 - pow(x, 2)))


f_diff_2 = np.vectorize(f_diff_2)


def x_array_by_number_of_steps(n):
    res_array = np.array([a])
    d = (b - a) / n
    for i in range(n):
        res_array = np.append(res_array, a + d * (i+1))
        pass
    return d, res_array


def x_array_by_length_of_steps(d):
    res_array = np.array([a])
    n = floor((b - a) / d)
    for i in range(n):
        res_array = np.append(res_array, a + d * (i+1))
        pass
    return d, res_array


def generate_data_table(d, array):
    data = pd.DataFrame()
    data["x"] = array
    data["f(x)"] = np.array(f(array))
    data["f'(x)"] = np.array(f_diff(array))
    data["lkra"] = [None] + [(data.iloc[i, 1] - data.iloc[i - 1, 1]) / d for i in range(1, len(data.index))]
    data["pkra"] = [(data.iloc[i + 1, 1] - data.iloc[i, 1]) / d for i in range(len(data.index) - 1)] + [None]
    data["ckra"] = [None] + [(data.iloc[i + 1, 1] - data.iloc[i - 1, 1]) / (2 * d) for i in
                             range(1, len(data.index) - 1)] + [None]
    data["f''(x)"] = np.array(f_diff_2(array))
    data["ckra_2"] = [None] + [(data.iloc[i - 1, 1] - 2 * data.iloc[i, 1] + data.iloc[i + 1, 1]) / pow(d, 2) for i in
                               range(1, len(data.index) - 1)] + [None]
    return data


def plot_table(table):
    plt.subplot(2, 1, 1)
    plt.plot(table["x"], table["f'(x)"], color=(0, 1, 0), label="f '(x)")
    plt.plot(table["x"], table["lkra"], color=(0.84, 0.76, 0.27), label="ЛКРА")
    plt.plot(table["x"], table["pkra"], color=(0.5, 0.75, 0.87), label="ПКРА")
    plt.plot(table["x"], table["ckra"], color=(1, 0, 0), label="ЦКРА")
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    plt.plot(table["x"], table["f''(x)"], color=(0, 1, 0), label="f ''(x)")
    plt.plot(table["x"], table["ckra_2"], color=(1, 0, 0), label="ЦКРА")
    plt.legend(loc="upper left")
    plt.savefig("shedule.png")


def generate_errors_table():
    table = pd.DataFrame(index=["LKRA", "PKRA", "CKRA", "CKRA2"])
    table["Errors"] = [round(i, ROUND_CONSTANT) for i in errors.values()]
    table["Taylor errors"] = [round(i, ROUND_CONSTANT) for i in taylor_errors.values()]
    table["Computational errors"] = [i for i in computational_errors.values()]
    return table


mode = input("0 - количество шагов, 1 - длина интервала: ")

x_array = np.array([])

h = None

precision = None

if mode == "0":
    number_of_steps = int(input("Количество шагов: "))
    h, x_array = x_array_by_number_of_steps(number_of_steps)
elif mode == "1":
    length_of_step = float(input("Длина интервала: "))
    h, x_array = x_array_by_length_of_steps(length_of_step)
else:
    print("Неверный ввод")
    raise Exception()
    pass

precision = float(input("Введите точность: "))

ROUND_CONSTANT = int(input("Введите количество знаков после запятой: "))

first_table = generate_data_table(h, x_array)

log_file = open("log.log", "w")
log_file.write("Step value (h): " + str(round(h, ROUND_CONSTANT)) + "\n\n")
log_file.write("X values:\n")
for i in x_array:
    log_file.write(str(round(i, ROUND_CONSTANT)) + "\n")
log_file.write("\n")
log_file.write("Precision: " + str(precision) + "\n\n")

is_finished = {"lkra": False, "pkra": False, "ckra": False, "ckra_2": False}

steps = {"lkra": h, "pkra": h, "ckra": h, "ckra_2": h}

errors = {"lkra": None, "pkra": None, "ckra": None, "ckra_2": None}

taylor_errors = {"lkra": MAX_F_DIFF_2 * h / 2, "pkra": MAX_F_DIFF_2 * h / 2, "ckra": MAX_F_DIFF_3 * h * h / 6,
                 "ckra_2": MAX_F_DIFF_4 * h * h / 12}

computational_errors = {"lkra": 2 * E / h, "pkra": 2 * E / h, "ckra": E / h, "ckra_2": 4 * E / h / h}

log_file.write("Max 2d: " + str(MAX_F_DIFF_2) + "\t")
log_file.write("Max 3d: " + str(MAX_F_DIFF_3) + "\t")
log_file.write("Max 4d: " + str(MAX_F_DIFF_4) + "\n\n\n")

log_file.write("-" * 20 + "\nStart loop\n" + "-" * 20 + "\n\n\n")

number_of_iteration = 0

while True:

    number_of_iteration = number_of_iteration + 1

    log_file.write("Number of iteration: " + str(number_of_iteration) + "\n\n")

    log_file.write("Current table:\n\n")
    log_file.write(first_table.to_string())
    log_file.write("\n\n")

    h, x_array = x_array_by_length_of_steps(h / 2)
    current_table = generate_data_table(h, x_array)
    current_table = current_table.loc[current_table["x"].isin(first_table["x"])]

    log_file.write("Table with h=" + str(round(h, ROUND_CONSTANT)) + ":\n\n")
    log_file.write(current_table.to_string())
    log_file.write("\n\n")

    if not is_finished["lkra"]:

        d_lkra_list = abs(first_table["lkra"].to_numpy() - current_table["lkra"].to_numpy())
        d_lkra_list = [i for i in d_lkra_list if not np.isnan(i)]
        errors["lkra"] = max(d_lkra_list)

        if errors["lkra"] <= precision:
            is_finished["lkra"] = True
        else:
            first_table["lkra"] = current_table["lkra"].to_numpy()
            steps["lkra"] = h
            taylor_errors["lkra"] = MAX_F_DIFF_2 * h / 2
            computational_errors["lkra"] = 2 * E / h

    if not is_finished["pkra"]:

        d_pkra_list = abs(current_table["pkra"].to_numpy() - first_table["pkra"].to_numpy())
        d_pkra_list = [i for i in d_pkra_list if not np.isnan(i)]
        errors["pkra"] = max(d_pkra_list)

        if errors["pkra"] <= precision:
            is_finished["pkra"] = True
        else
            first_table["pkra"] = current_table["pkra"].to_numpy()
            steps["pkra"] = h
            taylor_errors["pkra"] = MAX_F_DIFF_2 * h / 2
            computational_errors["pkra"] = 2 * E / h

    if not is_finished["ckra"]:

        d_ckra_list = abs(current_table["ckra"].to_numpy() - first_table["ckra"].to_numpy())
        d_ckra_list = [i for i in d_ckra_list if not np.isnan(i)]
        errors["ckra"] = max(d_ckra_list) / 3

        if errors["ckra"] <= precision:
            is_finished["ckra"] = True
        else:
            first_table["ckra"] = current_table["ckra"].to_numpy()
            steps["ckra"] = h
            taylor_errors["ckra"] = MAX_F_DIFF_3 * h * h / 6
            computational_errors["ckra"] = E / h

    if not is_finished["ckra_2"]:

        d_ckra_2_list = abs(current_table["ckra_2"].to_numpy() - first_table["ckra_2"].to_numpy())
        d_ckra_2_list = [i for i in d_ckra_2_list if not np.isnan(i)]
        errors["ckra_2"] = max(d_ckra_2_list) / 3

        if errors["ckra_2"] <= precision:
            is_finished["ckra_2"] = True
        else:
            first_table["ckra_2"] = current_table["ckra_2"].to_numpy()
            steps["ckra_2"] = h
            taylor_errors["ckra_2"] = MAX_F_DIFF_4 * h * h / 12
            computational_errors["ckra_2"] = 4 * E / h / h

    log_file.write("Error table:\n\n")
    log_file.write(generate_errors_table().to_string())
    log_file.write("\n\n\n")

    if is_finished["lkra"] and is_finished["pkra"] and is_finished["ckra"] and is_finished["ckra_2"]:
        log_file.write("\n\nResult table:\n\n")
        log_file.write(first_table.to_string())
        log_file.write("\n")
        log_file.close()



        out_file = open("output.txt", "w")
        out_file.write("Значение ЛКРА\n\n")
        lkra_table = pd.DataFrame()
        _, x_lkra = x_array_by_length_of_steps((steps["lkra"]))
        lkra_table["x"] = generate_data_table(steps["lkra"], x_lkra)["x"].to_numpy()
        lkra_table["lkra"] = generate_data_table(steps["lkra"], x_lkra)["lkra"].to_numpy()
        out_file.write(lkra_table.to_string() + "\n\n")

        _, x_pkra = x_array_by_length_of_steps((steps["pkra"]))
        out_file.write("Значение ПКРА\n\n")
        pkra_table = pd.DataFrame()
        pkra_table["x"] = generate_data_table(steps["pkra"], x_pkra)["x"].to_numpy()
        pkra_table["pkra"] = generate_data_table(steps["pkra"], x_pkra)["pkra"].to_numpy()
        out_file.write(pkra_table.to_string() + "\n\n")

        _, x_ckra = x_array_by_length_of_steps((steps["ckra"]))
        out_file.write("Значение ЦКРА\n\n")
        ckra_table = pd.DataFrame()
        ckra_table["x"] = generate_data_table(steps["ckra"], x_ckra)["x"].to_numpy()
        ckra_table["ckra"] = generate_data_table(steps["ckra"], x_ckra)["ckra"].to_numpy()
        out_file.write(ckra_table.to_string() + "\n\n")

        _, x_ckra_2 = x_array_by_length_of_steps((steps["ckra_2"]))
        out_file.write("Значение ЦКРА для второй производной\n\n")
        ckra_2_table = pd.DataFrame()
        ckra_2_table["x"] = generate_data_table(steps["ckra_2"], x_ckra_2)["x"].to_numpy()
        ckra_2_table["ckra"] = generate_data_table(steps["ckra_2"], x_ckra_2)["ckra_2"].to_numpy()
        out_file.write(ckra_2_table.to_string() + "\n\n")

        min_step = min(steps.values())
        _, x_array_by_min_step = x_array_by_length_of_steps(min_step)
        min_step_table = generate_data_table(min_step,x_array_by_min_step)
        plot_table(min_step_table)

        out_file.write("Точность метода, рассчитанная на основе принципа Рунге: \n")
        out_file.write("Для ЛКРА: " + str(round(errors["lkra"], ROUND_CONSTANT)) + "\n")
        out_file.write("Для ПКРА: " + str(round(errors["pkra"], ROUND_CONSTANT)) + "\n")
        out_file.write("Для ЦКРА: " + str(round(errors["ckra"], ROUND_CONSTANT)) + "\n")
        out_file.write("Для ЦКРА 2 производной: " + str(round(errors["ckra_2"], ROUND_CONSTANT)) + "\n\n")

        out_file.write("Точность метода, рассчитанная на основе оценки остаточного члена формулы численного дифференцирования: \n")
        out_file.write("Для ЛКРА: " + str(round(taylor_errors["lkra"], ROUND_CONSTANT)) + "\n")
        out_file.write("Для ПКРА: " + str(round(taylor_errors["pkra"], ROUND_CONSTANT)) + "\n")
        out_file.write("Для ЦКРА: " + str(round(taylor_errors["ckra"], ROUND_CONSTANT)) + "\n")
        out_file.write("Для ЦКРА 2 производной: " + str(round(taylor_errors["ckra_2"], ROUND_CONSTANT)) + "\n\n")

        out_file.write("Вычислительная погрешность: \n")
        out_file.write("Для ЛКРА: " + str(computational_errors["lkra"]) + "\n")
        out_file.write("Для ПКРА: " + str(computational_errors["pkra"]) + "\n")
        out_file.write("Для ЦКРА: " + str(computational_errors["ckra"]) + "\n")
        out_file.write("Для ЦКРА 2 производной: " + str(computational_errors["ckra_2"]) + "\n\n")

        out_file.close()
        break
