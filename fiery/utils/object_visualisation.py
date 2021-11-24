import math
# from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import transforms
# import matplotlib.colors as colors
# from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle


def vis_score(score, grid, cmap='cividis', ax=None):
    score = score.cpu().float().detach().numpy()
    grid = grid.cpu().detach().numpy()
    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 1], score, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Format axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    return ax


def draw_bbox2d(objects, color='k', ax=None):

    limits = ax.axis()

    for obj in objects:
        #         print("obj.classname: ", obj.classname);
        #         print("obj.rec:, ", obj.rec)
        #         print("obj.anglez: ", obj.angle)
        x, y, z = obj.position
        w, l, h = obj.dimensions

        # Setup transform
        t = transforms.Affine2D().rotate(-obj.angle + math.pi * 2.5)
        t = t.translate(x, y) + ax.transData

        # Draw 2D object bounding box
        rect = Rectangle((-w / 2, -l / 2), w, l,
                         edgecolor=color,
                         transform=t,
                         fill=False)
        ax.add_patch(rect)

        # Draw dot indicating object center
        center = Circle((x, y), 0.5, facecolor='k')
        ax.add_patch(center)

    ax.axis(limits)
    return ax


def vis_uncertainty(logvar, objects, grid, cmap='cividis_r', ax=None):
    var = logvar.cpu().float().detach().numpy()
    grid = grid.cpu().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 1], var, cmap=cmap)
    ax.set_aspect('equal')

    # Draw object positions
    draw_bbox2d(objects, ax=ax)

    # Format axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    return ax


def visualize_score(scores, heatmaps, grid):
    # Visualize score
    fig_score = plt.figure(num='score', figsize=(8, 6))
    fig_score.clear()

    vis_score(scores[0, 0], grid[0], ax=plt.subplot(121))
    vis_score(heatmaps[0, 0], grid[0], ax=plt.subplot(122))

    return fig_score


def visualize_bev(gt_objects, gt_heatmaps, pred_objects, pred_heatmaps, grid):

    # Visualize score
    fig_score = plt.figure(num='score', figsize=(10, 8))
    fig_score.clear()

    vis_uncertainty(gt_heatmaps[0, 0], gt_objects[0], grid[0], ax=plt.subplot(121))
#     vis_score(heatmaps[0, 0], grid[0], ax=plt.subplot(122))
    vis_uncertainty(pred_heatmaps[0, 0], pred_objects[0], grid[0], ax=plt.subplot(122))

    return fig_score
