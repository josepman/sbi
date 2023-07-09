def density_scatter( x , y, ax = None, sort = True, bins = 20, cmap='viridis', colorbar=True,  **kwargs )   :
    """
    Args:
        x:
        y:
        ax:
        sort:
        bins:
        cmap:
        colorbar:
        **kwargs:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from scipy.interpolate import interpn
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, cmap=cmap, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    if colorbar:
        cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax)
        # cbar = plt.colorbar(plot1, ax=axes['A'], shrink=0.5, aspect=15, fraction=.12, pad=.02)
        cbar.ax.set_ylabel('Density')

    return ax