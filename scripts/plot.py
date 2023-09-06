import sys
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("--plotly", action="store_true")
parser.add_argument("--title", default=None)
args = parser.parse_args()

names = {
    "cce": "CCE",
    "ce": "CE/QR",
    "tt": "TT-Rec",
    "dhe": "Deep Hash Embeddings",
    "simple": "Hashing Trick",
    "hnet": "Hashed Net",
    "hash": "Bloom / Sum Pooling",
    "bloom": "Bloom / Sum Pooling",
    "whemb": "Hash Embedding v2",
    "hemb": "Hash Embedding",
    "robe": "Robe",
    "cce_robe": "CCE Robe Hybrid",
    "full": "Baseline",
    "baseline": "Baseline",
    "ldim": "Low dimension",
}

ignore = {
        'hemb2', 'rhemb', 'hnet', 'ldim'
        }

main_title = None
data = {}
with open(args.file) as f:
    for line in f:
        if line.startswith("# "):
            main_title = line.strip("#").strip()
        if line.startswith("##"):
            method = line.strip("#").strip()
            if method not in data:
                data[method] = defaultdict(list)
        elif line[0].isdigit():
            ppd, *vals = map(float, line.split())
            if "auc" in args.file:
                vals = [max(v, 0.5) for v in vals]
            data[method][ppd] += vals

if args.title is not None:
    main_title = args.title

for method, md in data.items():
    if 2**18 not in md:
        print(method, end=" ")
print()

# Determine the fixed y-axis range
all_y_values = [
    y
    for series in data.values()
    for triple in series.values()
    for y in triple
    if y < 0.99
]
min_y = min(all_y_values)
max_y = max(all_y_values)

backend = "pyplot"
if args.plotly:
    backend = "plotly"

if backend == "pyplot":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,7))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', '--', '-.', ':']
    styles = ((color, linestyle) for linestyle in linestyles for color in default_colors)

    for method, series in data.items():
        if method in ignore:
            continue
        color, linestyle = next(styles)

        x = series.keys()
        y = series.values()
        # y_median = [np.median(triple) for triple in y]
        y_median = [np.mean(triple) for triple in y]
        y_lower = [min(triple) for triple in y]
        y_upper = [max(triple) for triple in y]

        plt.plot(x, y_median, label=names.get(method, method), color=color, linestyle=linestyle)
        plt.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

    plt.legend()
    plt.xlabel("Params/dim")
    plt.xscale("log")
    plt.ylabel("BCE" if args.file.endswith('ll') else 'AUC')
    plt.ylim(min_y * 0.99, max_y * 1.01)
    plt.title(main_title)
    plt.savefig('output.png')
    plt.show()

if backend == "plotly":
    import plotly.graph_objects as go

    fig = go.Figure()

    # This will retrieve plotly's default color cycle
    colorway = fig.layout.template.layout.colorway

    for index, (method, values) in enumerate(data.items()):
        x, y = zip(*values)
        y_median = [np.median(triple) for triple in y]
        y_lower = [min(triple) for triple in y]
        y_upper = [max(triple) for triple in y]
        color = colorway[index % len(colorway)]

        # Adding the fill with a similar color but with transparency
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                legendgroup=method,
                showlegend=False,
            )
        )

        # Adding the main line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_median,
                mode="lines",
                name=method,
                legendgroup=method,
                line=dict(color=color),
            )
        )

    fig.update_layout(
        title=main_title,
        xaxis=dict(type="log", title="Params/dim"),
        yaxis=dict(
            title="BCE" if args.file.endswith('ll') else 'AUC',
            range=[min_y * 0.99, max_y * 1.01],  # Set the fixed y-axis range
        ),
        showlegend=True,
    )

    # Saving the figure to an HTML file
    fig.write_html("output_plot.html")
