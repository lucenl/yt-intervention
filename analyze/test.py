import numpy as np
import matplotlib.pyplot as plt

def make_cdf(x, ax, label=None, color=None):
    x = np.sort(x)
    y = np.linspace(0, 1, len(x))
    kwargs = {}
    if label:
        kwargs['label'] = label
    if color:
        kwargs['color'] = color
        
    ax.plot(x, y, **kwargs)
    fig.savefig('test.png', bbox_inches='tight')
    
fig, ax = plt.subplots(dpi=300)


# random data to plot
data = [np.random.rand(100) for _ in range(3)]

# plot three lines
make_cdf(data[0], ax=ax, label='perc=0.00')
make_cdf(data[1], ax=ax, label='perc=0.25')
make_cdf(data[2], ax=ax, label='perc=0.50')

ax.legend()
ax.set_xlabel('data')
ax.set_ylabel('CDF')

