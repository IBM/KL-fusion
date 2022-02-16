from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import time
plt.rcParams["figure.figsize"] = (6,6)

def scale_margins(margins, contract=0.8):
    scale = contract*max([margins[1][0] - margins[0][0], margins[1][1] - margins[0][1]])
    coord_x = [margins[0][0], margins[1][0]]
    mid_x = (coord_x[0] + coord_x[1])/2
    coord_y = [margins[0][1], margins[1][1]]
    mid_y = (coord_y[0] + coord_y[1])/2
    new_margins = [[mid_x-scale/2, mid_y-scale/2], [mid_x+scale/2, mid_y+scale/2]]
    return new_margins

def draw_ellipse(position, covariance, ax=None, color=None, hatch=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        if hatch is not None:
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, zorder=0, hatch=hatch, **kwargs))
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, zorder=1, color=color, **kwargs))
        
        
def plot_gmm(means, covs, margins, X=None, marker='*', hatch=None, color = ['red', 'green', 'blue', 'orange']):
    ax = plt.gca()
    plt.xlim((margins[0][0],margins[1][0]))
    plt.ylim((margins[0][1],margins[1][1]))
    if covs is not None:
        for pos, covar, c in zip(means, covs, color):
            draw_ellipse(pos, covar, alpha=0.5, color=c, hatch=hatch)
        
    ax.scatter(means[:, 0], means[:, 1], s=100, color='black', alpha=1., zorder=3, marker=marker)
    
    if X is not None:
        ax.scatter(X[:, 0], X[:, 1], s=20, zorder=2, color='orange')

def plot_estimation(true, est, margins, X=None):
    color = ['red', 'blue']
    markers = ['*', 'o']
    hatch = ['/', None]
    for i, (means, covs) in enumerate([true, est]):
        plot_gmm(means, covs, margins, X=X, marker = markers[i], color = [color[i]]*means.shape[0],
                 hatch = hatch[i])
        
    plt.show()
    time.sleep(1)