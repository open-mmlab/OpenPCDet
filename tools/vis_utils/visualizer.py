import cv2, copy
import matplotlib.pyplot as plt
import numpy as np

import mayavi.mlab as mlab

def compute_colors_for_labels(label):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = label * palette
    colors = (colors % 255).astype("uint8")
    return colors


def draw_bbox_2d(image, cls_id, cls_name, box, color=None):
    # box: x1y1x2y2
    img = copy.deepcopy(image)
    (text_width, text_height) = cv2.getTextSize(str(cls_name), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]

    if color is None:
        color = compute_colors_for_labels(cls_id).tolist()
    img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), tuple(color), 2)
    img = cv2.rectangle(img, (int(box[0]), int(box[1])-3-text_height), (int(box[0]+text_width), int(box[1])-3), (0,0,0), -1)
    cv2.putText(img, str(cls_name),(int(box[0]), int(box[1])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return img

def draw_bbox_3d_img(image, points_img, color=None):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    camera coordinate:
      z
     /
     --> x
     |
     y
    """
    if color is None:
        color = (0,0,255)
    canvas = copy.deepcopy(image)
    for p in points_img:
        cv2.circle(canvas, (int(p[0]), int(p[1])), 2, color, -1)

    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        cv2.line(canvas, (int(points_img[e[0]][0]),int(points_img[e[0]][1])),
                 (int(points_img[e[1]][0]),int(points_img[e[1]][1])), color, 1)
    return canvas


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig

