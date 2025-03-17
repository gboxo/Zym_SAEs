import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create synthetic data
np.random.seed(42)
n_positions = 20
data_length = 100
data = []

for i in range(n_positions):
    # Generate two different sets of data for each position
    data1 = np.random.normal(loc=i, scale=0.5, size=data_length)
    data2 = np.random.normal(loc=i + 0.5, scale=0.5, size=data_length)
    for d in data1:
        data.append((d, i, 'Type A'))
    for d in data2:
        data.append((d, i, 'Type B'))

# Convert to a DataFrame
df = pd.DataFrame(data, columns=['Value', 'Position', 'Type'])

# Initialize the FacetGrid object
g = sns.FacetGrid(df, row="Position", hue="Type", aspect=15, height=0.4, palette="husl")

# Draw the densities in a few steps
g.map(sns.kdeplot, "Value",
      bw_adjust=1, clip_on=False,
      fill=True, alpha=0.6, linewidth=1.5)

# Draw a horizontal line at each kernel density estimate
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plots in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)



# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-0.8)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

plt.show()

