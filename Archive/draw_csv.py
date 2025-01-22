import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

arr = []
with open(sys.argv[1], newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in reader:
        arr.append(row)
arr = np.array(arr,dtype='float')
arr[:,1] /= np.max(arr[:,1])


fig,ax = plt.subplots()
ax.set_ylim(0, 1.1)
ax.set_xlim(300,1200)
ax.set_ylabel("Relative Intensity (arb.)")
ax.set_xlabel("Wavelength (nm)")
ax.ticklabel_format(useOffset=False, style='plain')

ax.fill_between(arr[:,0],arr[:,1], ec='#1F77B4',fc='#28BFFF')
plt.show()
