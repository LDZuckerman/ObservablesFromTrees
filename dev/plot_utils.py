import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def multi_base(ys, pred, targets):
    ''' 
    ys/pred should be the same dimensionality, targets should be numerical indexed, not boolean
    '''
    n_t = len(targets)
    figs=[]
    for n in range(n_t):
        fig, ax =plt.subplots(1,2, figsize=(12,6))
        ax=ax.flatten()
        ax[0].plot(ys[:,n],pred[:,n], 'ro', alpha=0.3)
        ax[0].plot([min(ys[:,n]),max(ys[:,n])],[min(ys[:,n]),max(ys[:,n])], 'k--', label='Perfect correspondance')
        ax[0].set(xlabel='Simulation Truth',ylabel='GNN Prediction', title=targets[n])
        yhat=r'$\hat{y}$'
        ax[0].text(0.6,0.15, f'Bias (mean(y-{yhat})) : {np.mean(ys[:,n]-pred[:,n]):.3f}', transform=ax[0].transAxes)
        ax[0].text(0.6,0.1, r'$\sigma$ :  '+f'{np.std(ys[:,n]-pred[:,n]):.3f}', transform=ax[0].transAxes)
        ax[0].legend()
        vals, x, y, _ =ax[1].hist2d(ys[:,n],pred[:,n],bins=50, norm=mpl.colors.LogNorm(), cmap=mpl.cm.magma)
        X, Y = np.meshgrid((x[1:]+x[:-1])/2, (y[1:]+y[:-1])/2)
        ax[1].contour(X,Y, np.log(vals.T+1), levels=10, colors='black')
        ax[1].plot([min(ys[:,n]),max(ys[:,n])],[min(ys[:,n]),max(ys[:,n])], 'k--', label='Perfect correspondance')
        ax[1].set(xlabel='Truth',ylabel='GNN Prediction', title=targets[n])
        ax[1].legend()
        fig.tight_layout()
        figs.append(fig)
    return figs