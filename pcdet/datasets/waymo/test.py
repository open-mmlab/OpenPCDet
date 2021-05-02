import numpy as np
# from waymo_utils import plot_pointcloud
import mayavi.mlab as mlab


def plot_pointcloud(pointcloud, vals='height'):
    import mayavi.mlab as mlab
    print(pointcloud.shape)
    print(type(pointcloud))
    x = pointcloud[:, 0]  # x position of point
    xmin = np.amin(x, axis=0)
    xmax = np.amax(x, axis=0)
    y = pointcloud[:, 1]  # y position of point
    ymin = np.amin(y, axis=0)
    ymax = np.amax(y, axis=0)
    z = pointcloud[:, 2]  # z position of point
    zmin = np.amin(z, axis=0)
    zmax = np.amax(z, axis=0)
    print(xmin, xmax, ymin, ymax, zmin, zmax)
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    if vals == "height":
        col = z
    else:
        col = d
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mlab.points3d(x, y, z,
                  col,  # Values used for Color
                  mode="point",
                  # 灰度图的伪彩映射
                  colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                  # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                  figure=fig,
                  )
    # 绘制原点
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=1)
    # 绘制坐标
    axes = np.array(
        [[20.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 0.0]],
        dtype=np.float64,
    )
    # x轴
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    # y轴
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    # z轴
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    return fig


point = np.fromfile('0004.npy')
point = point.reshape(-1, 6)
print(point)
# fig = plot_pointcloud(point)
# mlab.show()
