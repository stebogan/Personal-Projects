import matplotlib.pyplot as plt

y1 = "S14621-01-104-LongBaby1-CVsweep.dat"
y2 =
y3 =
y4 =
y5 =
y6 =
y7 =
y8 =
y9 =
...
title = "Capacitance vs. Frequency Sweep (1K-1000K Hz)"

plt.figure(1)
plt.subplot(211)
plt.plotfile(y1, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y2, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y3, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y4, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y5, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y6, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y7, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y8, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.plotfile(y9, delimiter = ' ', cols=(0, 1),
                names=('Bias Voltage (V)', 'Leakage Current'))

plt.title(title)

plt.title(title)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


plt.show()
