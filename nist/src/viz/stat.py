import matplotlib
import matplotlib.pyplot as plt
import ternary

matplotlib.rcParams["figure.dpi"] = 80
matplotlib.rcParams["figure.figsize"] = (5.5, 5.5)


def draw_ternary_heatmap(accuracies, num_segment, fig_path, train_ds, title=None):
    fontsize = 10

    _, tax = ternary.figure(scale=num_segment)
    # print(accuracies)
    cb_kwargs = {"shrink": 0.8, "pad": 0.05, "aspect": 30, "orientation": "horizontal"}
    tax.heatmap(accuracies, style="t", cb_kwargs=cb_kwargs)
    tax.boundary()

    tax.right_corner_label(train_ds[0], fontsize=fontsize)
    tax.top_corner_label(train_ds[1], fontsize=fontsize)
    tax.left_corner_label(train_ds[2], fontsize=fontsize)
    if title is not None:
        tax.set_title(
            title,
            y=1.2,
            pad=-14,
        )
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")

    tax.savefig(
        fig_path,
        facecolor="w",
        bbox_inches="tight",
    )


def draw_mccan_band_curve(accuracies_list, labels, x_axis, title, save_path, x_label):
    colors = [
        "blue",
        "darkturquoise",
        "magenta",
        "gold",
        "olive",
        "brown",
        "darkviolet",
    ]
    _, ax = plt.subplots(figsize=(6.5, 5), dpi=80, facecolor="w")

    for color, label, accuracies in zip(colors, labels, accuracies_list):
        average_accuracy = accuracies.mean(axis=0)
        accuracy_std = accuracies.std(axis=0)
        ax.plot(x_axis, average_accuracy, color=color, label=label)
        ax.fill_between(
            x_axis,
            (average_accuracy - accuracy_std),
            (average_accuracy + accuracy_std),
            color=color,
            alpha=0.1,
        )

    ax.legend(prop={"size": 14})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel("Test accuracy", fontsize=16)
    plt.savefig(save_path)
