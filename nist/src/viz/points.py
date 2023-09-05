import matplotlib.pyplot as plt
import numpy as np

from .wandb_fig import wandb_img


def draw_histogram(labels, fig_path, num_class=10, wandb_banner="ps_histogram"):
    # labels: (n,)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2), facecolor="w")
    ax.hist(labels, bins=num_class)
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    wandb_img(wandb_banner, fig_path, fig_path)


def draw_classifier(feat, label, fig_path, num_labels, wandb_banner="train_data"):
    feat = feat.detach().cpu()
    label = label.detach().cpu()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), facecolor="w")
    cmap = plt.get_cmap("jet")
    ax.scatter(feat[:, 0], feat[:, 1], color=cmap(label / num_labels), alpha=0.7)
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    wandb_img(wandb_banner, fig_path, fig_path)


def draw_source_target(
    source,
    target,
    fig_path,
    num_source_class,
    num_target_class,
    wandb_banner="train_data",
    plot_size=3,
):
    source = source.detach().cpu()
    target = target.detach().cpu()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), facecolor="w")
    cmap = plt.get_cmap("jet")
    ax[0].scatter(
        source[:, 0],
        source[:, 1],
        color=cmap(source[:, 2] / (num_source_class + num_target_class)),
        alpha=0.7,
    )
    ax[1].scatter(
        target[:, 0],
        target[:, 1],
        color=cmap(
            (target[:, 2] + num_source_class) / (num_source_class + num_target_class)
        ),
        alpha=0.7,
    )
    lims = (-plot_size, plot_size)
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    wandb_img(wandb_banner, fig_path, fig_path)


def draw_trajectory(
    trajectory,
    target,
    fig_path,
    num_source_class,
    num_target_class,
    wandb_banner="geodesic",
    plot_size=3,
):
    num_intpl = len(trajectory)
    target = target.detach().cpu()
    fig, ax = plt.subplots(
        nrows=1, ncols=num_intpl + 1, figsize=(4 * (num_intpl + 1), 4), facecolor="w"
    )
    cmap = plt.get_cmap("jet")
    lims = (-plot_size, plot_size)
    for idx in range(num_intpl):
        trajectory[idx] = trajectory[idx].detach().cpu()
        ax[idx].scatter(
            trajectory[idx][:, 0],
            trajectory[idx][:, 1],
            color=cmap(trajectory[idx][:, 2] / (num_source_class + num_target_class)),
            alpha=0.7,
        )
        ax[idx].set_xlim(lims)
        ax[idx].set_ylim(lims)
        ax[idx].set_title(f"t={idx/(num_intpl-1)}", fontsize=18)
    ax[-1].scatter(
        target[:, 0],
        target[:, 1],
        color=cmap(
            (target[:, 2] + num_source_class) / (num_source_class + num_target_class)
        ),
        alpha=0.7,
    )
    ax[-1].set_xlim(lims)
    ax[-1].set_ylim(lims)
    ax[-1].set_title("target", fontsize=18)
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    wandb_img(wandb_banner, fig_path, fig_path)


def grid_nn_2_generator(num_grid, left_place, right_place):
    x, y = xyIndex_generator(num_grid, left_place, right_place)
    x_plot = x.reshape(-1, 1)[:, 0]
    y_plot = y.reshape(-1, 1)[:, 0]
    pos_plot = np.stack((x_plot, y_plot)).T
    return pos_plot


def xyIndex_generator(num_grid, left_place, right_place):
    grid_size = (right_place - left_place) / num_grid
    x, y = np.mgrid[left_place:right_place:grid_size, left_place:right_place:grid_size]
    return x, y
