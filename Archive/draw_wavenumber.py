import csv
import matplotlib.pyplot as plt
import numpy as np

# import smplotlib
import sys

arr = []
with open(sys.argv[1], newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar="|")
    for row in reader:
        arr.append(row)
arr = np.array(arr, dtype="float")

arr_lit = []
with open("literature_polystyrene.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar="|")
    for row in reader:
        arr_lit.append(row)
arr_lit = np.array(arr_lit, dtype="float")

low_ind = 0
high_ind = 0
for ind, i in enumerate(arr):
    if i[0] < 550:
        low_ind = ind
    elif i[0] < 700:
        high_ind = ind

arr = arr[low_ind:high_ind]
laser_wavenumber = 10000000 / 532
arr[:, 0] = laser_wavenumber - 10000000 / arr[:, 0]
arr[:, 1] /= np.max(arr[:, 1])


fig, ax = plt.subplots()
ax.set_ylim(0, 1.1)
ax.set_xlim(arr[0, 0], arr[-1, 0])
ax.set_ylabel("Relative Intensity (arb.)")
ax.set_xlabel("Relative Wavenumber (1/cm)")
ax.ticklabel_format(useOffset=False, style="plain")

ax.fill_between(arr[:, 0], arr[:, 1], ec="#1F77B4", fc="#28BFFF")

# ax.plot(arr_lit[:,0],arr_lit[:,1]/17000,ls='--',c='red',alpha=0.5)
plt.show()
