'''

check the additivability of A response on stress/strain

'''

import json

import numpy as np


def is_linear_leading(fit_res):
    # x=0.01
    return abs(fit_res["fit"][0] / fit_res["fit"][1]) < 10 and abs(fit_res["fit"][2] / fit_res["fit"][1]) < 1000


strain_dict = {"xx": np.array([1, 0, 0, 0, 0, 0]),
               "yy": np.array([0, 1, 0, 0, 0, 0]),
               "zz": np.array([0, 0, 1, 0, 0, 0]),
               "yz": np.array([0, 0, 0, 1, 0, 0]),
               "zx": np.array([0, 0, 0, 0, 1, 0]),
               "xy": np.array([0, 0, 0, 0, 0, 1])}

atom_name = ["$^{14}N$", "$^{13}C_1$", "$^{13}C_2$", "$^{13}C_3$", "$^{13}C_5$", "$^{13}C_4$", "energy"]
work_dir = "C:/Users/dugue/OneDrive/117_520"

fit_res_colle = json.load(open(f"{work_dir}/dontread_fit_res_colle.json", "r"))
for i in fit_res_colle:
    for atom in fit_res_colle[i]:
        fit_res_colle[i][atom]["linear_leading"] = is_linear_leading(fit_res_colle[i][atom])
        fit_res_colle[i][atom]["fit"] = np.array(fit_res_colle[i][atom]["fit"])
        fit_res_colle[i][atom]["error"] = np.array(fit_res_colle[i][atom]["error"])
        fit_res_colle[i][atom]["relative"] = np.abs(fit_res_colle[i][atom]["error"] / fit_res_colle[i][atom]["fit"])

basis_fit = {i: fit_res_colle.pop(i) for i in ["xx", "yy", "zz", "yz", "zx", "xy"]}
basis = {atom: np.array([basis_fit[j][atom]["fit"][1] for j in basis_fit]) for atom in atom_name}
basis["energy"] = np.array([basis_fit[j]["energy"]["fit"][0] for j in basis_fit])

for i in fit_res_colle:
    factor_list = np.zeros(6)
    for j in i.replace("-", "+-").split("+"):
        if j:
            if "*" in j:
                factor, strain = j.split("*")
                factor = float(factor)
            elif "-" in j:
                factor = -1
                strain = j.replace("-", "")
            else:
                factor = 1
                strain = j
            factor_list += strain_dict[strain] * factor
    for atom in atom_name:
        fit_res_colle[i][atom]["add"] = factor_list.dot(basis[atom])
    fit_res_colle[i]["energy"]["add"] = np.power(factor_list, 2).dot(basis["energy"])

# json_tricks.dump(fit_res_colle, open(f"{work_dir}/dontread_fit_res_colle2.json", "w"), indent=4)

# display
# for i in fit_res_colle:
#     print(i)
#     print(*[fit_res_colle[i][atom]["fit"][0] if atom == "energy" else fit_res_colle[i][atom]["fit"][1] for atom in
#             fit_res_colle[i]])
#     print(*[fit_res_colle[i][atom]["add"] for atom in fit_res_colle[i]])
#     print(*[round((fit_res_colle[i][atom]["add"]) /
#                   (fit_res_colle[i][atom]["fit"][0] if atom == "energy" else fit_res_colle[i][atom]["fit"][1]), 2) for
#             atom in fit_res_colle[i]])

for i in basis_fit:
    print(i)
    print(
        *[basis_fit[i][atom]["fit"][0] if atom == "energy" else basis_fit[i][atom]["fit"][1] for atom in basis_fit[i]])
