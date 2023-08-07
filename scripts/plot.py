import sys
import matplotlib.pyplot as plt
import numpy as np

data = {}
with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith('##'):
            method = line.strip('#').strip()
            data[method] = []
        elif line[0].isdigit():
            ppd, *seeds = map(float, line.split())
            # ppd > 1900 doesn't make sense, as it's bigger than the vocab
            if ppd > 1900:
                continue
            data[method].append((ppd, seeds))

plt.figure()
for method, values in data.items():
    x, y = zip(*values)
    y_median = [np.median(triple) for triple in y]
    y_lower = [min(triple) for triple in y]
    y_upper = [max(triple) for triple in y]

    plt.plot(x, y_median, label=method)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2)

plt.legend()
plt.xlabel('Params/dim')
plt.xscale('log')
plt.ylabel('BCE')
plt.title('ml100k best of 10 epochs')
plt.show()
