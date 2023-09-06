import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

s = sys.stdin.read()

rows = [r.split() for r in s.split('\n') if r and not r.startswith('#')]
cols = len(rows[0])
print(rows)
xs = [2**i for i in range(0, cols-1)]
for j, row in enumerate(rows[1:]):
    ys = [float(y) for y in row[1:]]

    print(ys)
    print(xs)

    plt.plot(xs, ys, label=f'{row[0]} rows')
    plt.scatter(xs, ys)

    # Find the y-maximum value and its corresponding x value
    y_max = max(ys)
    x_max = xs[ys.index(y_max)]


    # Plot a black dot at the (x_max, y_max) position
    plt.scatter(x_max, y_max, color='black')

    y_pos = ys[0]  # Get the y value of the first point for the line
    label_str = '$2^{' + str(j+2) + '}$'
    if j == len(rows)-2:
        label_str = 'ppd=' + label_str
    plt.annotate(label_str, (xs[0], y_pos), textcoords="offset points", xytext=(0,1), ha='center', fontsize=9)


plt.xscale('log')
plt.xlabel("N Chunks")
plt.ylabel('AUC')
plt.title("Movielens 25M, CE-concat, dim=32, best of 10 epochs")
#plt.legend()
plt.show()
