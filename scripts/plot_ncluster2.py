import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()



series = {64: ([16, 32], [0.649, 0.667]),
 128: ([1, 2, 4, 8, 16, 32], [0.581, 0.607, 0.637, 0.657, 0.67, 0.679]),
 256: ([1, 2, 4, 8, 16, 32], [0.608, 0.634, 0.657, 0.672, 0.682, 0.686]),
 512: ([1, 2, 4, 8, 16, 32], [0.628, 0.655, 0.672, 0.682, 0.69, 0.698]),
 1024: ([1, 2, 4, 8, 16, 32], [0.645, 0.67, 0.683, 0.694, 0.71, 0.731]),
 2048: ([1, 2, 4, 8, 16, 32], [0.664, 0.684, 0.7, 0.725, 0.753, 0.769]),
 4096: ([1, 2, 4, 8, 16, 32], [0.681, 0.701, 0.726, 0.759, 0.781, 0.786]),
 8192: ([1, 2, 4, 8, 16, 32], [0.698, 0.723, 0.756, 0.787, 0.797, 0.797]),
 16384: ([1, 2, 4, 8, 16, 32], [0.719, 0.75, 0.782, 0.802, 0.805, 0.804]),
 32768: ([1, 2, 4, 8, 16, 32], [0.744, 0.777, 0.801, 0.81, 0.81, 0.809]),
 65536: ([1, 2, 4, 8, 16, 32], [0.774, 0.799, 0.811, 0.815, 0.814, 0.813]),
 131072: ([1, 2, 4, 8, 16], [0.802, 0.815, 0.821, 0.819, 0.82])}

for ppd, (xs, ys) in series.items():

    plt.plot(xs, ys, label=f'{ppd} rows')
    plt.scatter(xs, ys)

    # Find the y-maximum value and its corresponding x value
    y_max = max(ys)
    x_max = xs[ys.index(y_max)]


    # Plot a black dot at the (x_max, y_max) position
    plt.scatter(x_max, y_max, color='black')

    y_pos = ys[0]  # Get the y value of the first point for the line
    label_str = '$2^{' + str(int(math.log2(ppd))) + '}$'
    if ppd == 131072:
        label_str = 'ppd=' + label_str
    plt.annotate(label_str, (xs[0], y_pos), textcoords="offset points", xytext=(0,3), ha='center', fontsize=9)


plt.xscale('log')
plt.xlabel("N Chunks")
plt.xticks([2**i for i in range(6)], [2**i for i in range(6)])
plt.ylabel('AUC')
plt.title("Movielens 25M, CCE, dim=32, best of 10 epochs")
#plt.legend()
plt.show()
