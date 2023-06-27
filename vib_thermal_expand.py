'''
%freq changes when cell expanding
we assume the effects of thermal expansion and lattice vibration on $A$ parameter are individual,
the phonon properties vary when lattice volume changing at finite temperature
since the crystal potential is an anharmonic function of volume\cite{togo_2015_first}.
Typically phonon frequency decreases by increasing volume,
and the slope of each phonon mode is nearly constant in a wide volume range.
The normalized slope is called mode-Grüneisen parameter that is defined as
\begin{equation}
\gamma_q(V) = -\frac{V}{\omega_q(V)} \cdots \frac{\partial\omega_q(V)}{\partial V}
\end{equation},
Usually, the mode-Grüneisen parameter is in the order of several.
According to the thermal expansion of diamond,
the relative variations of volume and phonon frequency are about five orders of magnitude
smaller than temperature variation indicating the effect of thermal expansion on vibration calculations is limited.
The deduction also supported by the phonon calculation at gamma point in different lattice parameters
showing in Fig.~\ref{}.
'''

import json
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

work_dir = r"C:\Users\dugue\OneDrive\freq_thermal_expand"
plt.rcParams["font.size"] = 15

lattice_parameter = np.array(range(83)[70:]) / 100 + 10
temp_list = []
for vib_json in os.listdir(work_dir):
    temp_list.append([int(vib_json[1:3]), json.load(open(work_dir + "/" + vib_json, "r"))["freq"]])
temp_list.sort(key=lambda x: x[0])
freq_volume = np.array(list(zip(*temp_list))[1])
freq_volume /= freq_volume[0]
freq_volume -= 1
freq_volume = -freq_volume.T

lattice_parameter = lattice_parameter ** 3
lattice_parameter /= lattice_parameter[0]
lattice_parameter -= 1

coef_list = []
cov_list = []
for fv in freq_volume:
    coef, cov = np.polyfit(lattice_parameter, fv, 1, cov=True)
    coef_list.append(coef[0])
    cov_list.append(cov[0][0])
cov_list = np.sqrt(cov_list)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(len(coef_list) - 3)), coef_list[3:], label="fitting")
ax.plot(list(range(len(cov_list) - 3)), cov_list[3:], label="error")
ax.set_xlabel("phonon mode")
ax.set_ylabel("Grüneisen parameter")
ax.yaxis.set_major_locator(MaxNLocator(5))
plt.legend()
plt.tight_layout()
plt.savefig("freq_thermal_expand.png", dpi=600)
