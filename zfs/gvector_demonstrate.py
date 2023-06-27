import matplotlib.pyplot as plt
import numpy as np

def gvector_demonstrate():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x_range, y_range, z_range = [np.arange(-5, 6, 1) for _ in range(3)]
    x_grid, y_grid, z_grid = np.array(np.meshgrid(x_range, y_range, z_range, indexing="ij")).reshape(3, -1)
    kgrid = np.array([x_grid, y_grid, z_grid]).T
    plane=True
    if plane:
        considered = ((x_grid == 0) & (y_grid > 0)) | ((x_grid == 0) & (y_grid == 0) & (z_grid >= 0))
        regardless = ((x_grid == 0) & (y_grid < 0)) | ((x_grid == 0) & (y_grid == 0) & (z_grid < 0))
    else:
        considered = (x_grid > 0) | ((x_grid == 0) & (y_grid > 0)) | ((x_grid == 0) & (y_grid == 0) & (z_grid >= 0))
        regardless = np.invert(considered)

    kgrid_considered = kgrid[considered]
    kgrid_regardless = kgrid[regardless]
    print(kgrid.shape, kgrid_considered.shape, kgrid_regardless.shape)
    ax.scatter(*kgrid_considered.T, color="green")
    ax.scatter(*kgrid_regardless.T, color="red")
    plt.show()
