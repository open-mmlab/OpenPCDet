import numpy as np
# from waymo_utils import plot_pointcloud
import mayavi.mlab as mlab
import matplotlib.pyplot as plt


def plot_bev(pointcloud, res=(0.1, 0.1, 2), pc_range=(-80, -80, -10.0, 80, 80, 10.0)):
    print(pointcloud.shape)
    print(type(pointcloud))
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]
    xmin = max(np.amin(x, axis=0),pc_range[0])
    xmax = min(np.amax(x, axis=0),pc_range[3])

    ymin = max(np.amin(y, axis=0),pc_range[1])
    ymax = min(np.amax(y, axis=0),pc_range[4])

    zmin = max(np.amin(z, axis=0),pc_range[2])
    zmax = min(np.amax(z, axis=0),pc_range[5])
    print(xmin, xmax, ymin, ymax, zmin, zmax)
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor


    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_dim = int((xmax-xmin) / res[0])
    y_dim = int((ymax-ymin) / res[1])
    z_dim = int((zmax-zmin) / res[2])
    print(x_dim,y_dim,z_dim)
    top = np.zeros([ x_dim + 1,y_dim + 1,z_dim+1], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x > xmin), (x < xmax))
    s_filt = np.logical_and(
        (y > ymin), (y < ymax))
    filt = np.logical_and(f_filt, s_filt)

    for i, height in enumerate(np.arange(zmin, zmax, res[2])):
        z_filt = np.logical_and((z >= height),
                                (z < height + res[2]))
        zfilter = np.logical_and(filt, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x[indices]
        yi_points = y[indices]
        zi_points = z[indices]
        ri = r[indices]
        di = d[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (yi_points / res[1]).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (xi_points / res[0]).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(ymin / res[1]))
        y_img -= int(np.floor(xmin / res[0]))

        pixel_values = ri

        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img,x_img,   i] = zi_points

    # z_filt = np.logical_and((z >= zmin),
    #                         (z < zmax))
    # zfilter = np.logical_and(filt, z_filt)
    # indices = np.argwhere(zfilter).flatten()
    #
    # # KEEPERS
    # xi_points = x[indices]
    # yi_points = y[indices]
    # zi_points = z[indices]
    # ri = r[indices]
    # di = d[indices]
    #
    # # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    # x_img = (yi_points / res[1]).astype(np.int32)  # x axis is -y in LIDAR
    # y_img = (xi_points / res[0]).astype(np.int32)  # y axis is -x in LIDAR
    #
    # # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # # floor & ceil used to prevent anything being rounded to below 0 after
    # # shift
    # x_img -= int(np.floor(ymin / res[1]))
    # y_img -= int(np.floor(xmin / res[0]))
    #
    # pixel_values = di
    #
    # # FILL PIXEL VALUES IN IMAGE ARRAY
    # top[x_img, y_img] = pixel_values



    top = (top / np.max(top) * 255).astype(np.uint8)
    return top

point = np.load('0004.npy')
fig, ax = plt.subplots(figsize=(10, 10))

top = plot_bev(point)
top = np.array(top, dtype = np.float32)
print(top.shape)
ax.imshow(top, aspect='equal')
plt.axis('off')
plt.tight_layout()
plt.draw()
plt.show()