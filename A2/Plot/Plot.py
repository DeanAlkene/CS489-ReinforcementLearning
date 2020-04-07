import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib import axes
 
def draw_heatmap(data, title):
    cmap=cm.Blues    
    #cmap = cm.get_cmap('rainbow',1000)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax.set_title(title)
    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    # map = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    plt.savefig(title + '.jpg')

def main():
    col_names = [str(i) for i in range(6)]
    titles = [
        'First Visit MC',
        'Every Visit MC',
        'TD(0), alpha=0.10',
        'TD(0), alpha=0.20',
        'TD(0), alpha=0.30',
        'TD(0), alpha=0.40',
        'TD(0), alpha=0.50',
        'TD(0), alpha=0.60',
        'TD(0), alpha=0.70',
        'TD(0), alpha=0.80',
        'TD(0), alpha=0.90',
        'True Value'
    ]
    for i in range(len(titles)):
        data = pd.read_csv(str(i)+'.txt', sep='\t', names=col_names)
        data = data.values
        draw_heatmap(data, titles[i])

if __name__ == '__main__':
    main()
