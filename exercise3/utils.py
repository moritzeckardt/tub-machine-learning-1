import numpy
import matplotlib.pyplot as plt


def render(x, w, h, vmax=None, shape=(62, 47), width=900):
    """Show the rows of x (each an image of the given shape, flattened) as a
    grid of h rows and w columns; gray is 0, the range is [-vmax, vmax]
    (by default the largest absolute value). The grid is at most `width`
    screen pixels wide and at most 4 screen pixels per image pixel."""
    x = numpy.asarray(x, dtype=float).reshape(h * w, *shape)
    vmax = numpy.abs(x).max() if vmax is None else vmax
    hh, ww = shape[0] + 2, shape[1] + 2
    z = numpy.full((h, w, hh, ww), float(vmax))
    z[:, :, 1:-1, 1:-1] = x.reshape(h, w, *shape)
    z = z.transpose(0, 2, 1, 3).reshape(h * hh, w * ww)
    scale = min(4, width / z.shape[1])
    fig = plt.figure(
        figsize=(scale * z.shape[1] / 100, scale * z.shape[0] / 100), dpi=100
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(z, cmap="gray", vmin=-vmax, vmax=vmax, interpolation="lanczos")
    ax.axis("off")
    plt.show()


def scatterplot(x, y, xlabel="", ylabel="", equal=True):
    plt.figure(figsize=(3.5, 3.5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, ".", ms=2)
    if equal:
        plt.gca().set_aspect("equal")
    plt.show()
