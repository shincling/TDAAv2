import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import *
import numpy as np
def relitu(x=None,y=None,z=None):
    if not x and not y and not z:
        x, y = np.arange(0,10), np.arange(20)
        z = (np.random.rand(9000000)+np.linspace(0,1, 9000000)).reshape(3000, 3000)
    else:

        x, y = np.arange(0,x), np.arange(0,y)
    # plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
    #         cmap=cm.hot, norm=LogNorm())
    # plt.imshow(np.sum(z+10,1).reshape((626,1)).repeat(10,1), extent=(np.amin(x), np.amax(x), 0, 10),
    #         cmap=cm.hot, norm=LogNorm())
    plt.imshow((z+10).reshape(-1,626),
               cmap=cm.hot, norm=LogNorm())
    # plt.imshow(z[1]+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
    #            cmap=cm.hot, norm=LogNorm())
    plt.colorbar()
    plt.show()

def relitu_line(x=None,y=None,z=None):
    if not x and not y and not z:
        x, y = np.arange(0,10), np.arange(30)
        z = (np.random.rand(9000000)+np.linspace(0,1, 9000000)).reshape(3000, 3000)
    else:

        x, y = np.arange(0,x), np.arange(0,10)
    plt.imshow((z).reshape(-1,626),extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
               cmap=cm.hot, norm=LogNorm())
    # plt.imshow(z[1]+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
    #            cmap=cm.hot, norm=LogNorm())
    plt.colorbar()
    plt.show()
# relitu()
