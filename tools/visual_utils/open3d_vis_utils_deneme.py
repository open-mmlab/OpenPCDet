"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
from PIL import Image, ImageFont, ImageDraw
#from pyquaternion import Quaternion
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, class_names = None, save_outputs=False, filename=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    if save_outputs: # dont visualize 1 by 1, save them to an output folder
        vis.create_window(visible=False)
    else:
        vis.create_window(visible=True)   

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    
    
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    print("POINTS SHAPE",points.shape)

    vis.add_geometry(pts)
      
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        print("gt boxes are drawing")        
        vis = draw_box(vis, gt_boxes, (0, 0, 1), class_names=class_names)
        

    if ref_boxes is not None:
        print("ref boxes are drawing")
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores,class_names=class_names)
        
        
        
        
        

    if not save_outputs: 
        vis.run()
        vis.destroy_window()
    else:
        savename = save_outputs + filename.split(".")[0] + ".jpg"
        depth = vis.capture_depth_float_buffer(True) 
        plt.imsave(savename,np.asarray(depth))
        # img = vis.capture_screen_float_buffer(True) 
        # plt.imsave(savename,np.asarray(img))
               


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

def text_3d(text, pos, direction=None, degree=0.0, font="/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size=10, density=50000):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: open3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)
    
    print("font_dim", font_dim)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
   
    img.save('pil_text.png')

    im = open3d.io.read_image("./pil_text.png")
    
    return im

def draw_box(vis, boxes, color=(0, 1, 0), ref_labels=None, score=None, class_names=None):
    for i in range(boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
        
        print(class_names[ref_labels[i].item()-1])
        print("center points", boxes[i][0:3])
        print("3d box center points", box3d.extent)
        
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        
        
        text = class_names[ref_labels[i].item()-1]
        x,y,z = box3d.extent[0:3]
        
        pcd_10 = text_3d(text, pos=(z,y,x), font_size=10, density=10)
        vis.add_geometry(pcd_10)
        
        # img = Image.new('RGB', font_dim, color=(255, 255, 255))
        # draw = ImageDraw.Draw(img)
        # draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        # img = np.asarray(img)
        # img_mask = img[:, :, 0] < 128
        # indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T
                
        

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
