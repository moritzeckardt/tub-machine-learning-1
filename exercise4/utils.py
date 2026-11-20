import numpy
import matplotlib.pyplot as plt

# the colors of the lecture: class 1 blue, class 2 red
CLASS1 = (.12, .32, .75)
CLASS2 = (.80, .15, .15)
# line styles of the decision boundaries, the last classifier in black
LINES = [dict(color='k', ls='-'), dict(color='0.5', ls='--'),
         dict(color=(.90, .42, 0), ls='-.'), dict(color=(.30, .62, .05), ls=':')]


def scatter(X):
    # scatter matrix of the rows of X: a sum over the points
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc


def plot_classes(X1, X2, classifiers=(), title=None, ax=None, legend=True,
                 mark_errors=False, ms=3):
    """Scatter plot of two classes in two dimensions with the decision
    boundaries w^T x + b = 0 of the classifiers, a list of (w, b, label).
    With mark_errors=True, the points on the wrong side of the last boundary
    get a black ring."""
    show = ax is None
    if show:
        plt.figure(figsize=(4.5, 4.5))
        ax = plt.gca()
    ax.plot(*X1.T, 'o', ms=ms, color=CLASS1, label=r'class $\omega_1$')
    ax.plot(*X2.T, 'o', ms=ms, color=CLASS2, label=r'class $\omega_2$')
    ax.set_aspect('equal')
    ax.margins(0.1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    styles = LINES[:len(classifiers)][::-1]
    for (w, b, label), style in zip(classifiers, styles):
        w = numpy.asarray(w, dtype=float)
        point = -b * w / (w @ w)                    # the point of the line closest to 0
        along = numpy.array([-w[1], w[0]]) / numpy.linalg.norm(w)
        s = numpy.array([-1, 1]) * 100
        ax.plot(*(point[:, None] + along[:, None] * s), lw=1.5, label=label, **style)
    if mark_errors and classifiers:
        w, b, _ = classifiers[-1]
        wrong = numpy.concatenate([X1[X1 @ w + b > 0], X2[X2 @ w + b <= 0]])
        ax.plot(*wrong.T, 'o', ms=3 * ms, mfc='none', mec='k', label='wrong side')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    if show:
        plt.show()


def plot_projections(panels, ncols=1, bins=25):
    """Histograms of f(x) = w^T x + b for both classes, one panel per
    (X1, X2, w, b, title), in rows of ncols panels. Each class has the same
    area, whatever its number of points. The axes are symmetric around the
    threshold f = 0, with the same range within a row."""
    nrows = (len(panels) + ncols - 1) // ncols
    width, height = (7, 1.5) if ncols == 1 else (5.5 * ncols, 2.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * nrows), squeeze=False)
    for row in range(nrows):
        values = [(X1 @ w + b, X2 @ w + b, title)
                  for X1, X2, w, b, title in panels[row * ncols:(row + 1) * ncols]]
        limit = 1.05 * max(numpy.abs(numpy.concatenate([z1, z2])).max()
                           for z1, z2, _ in values)
        edges = numpy.linspace(-limit, limit, 2 * bins + 1)
        for ax, (z1, z2, title) in zip(axes[row], values):
            ax.hist(z1, edges, color=CLASS1, alpha=.6, density=True)
            ax.hist(z2, edges, color=CLASS2, alpha=.6, density=True)
            ax.axvline(0, color='k', lw=1)
            ax.set_xlim(-limit, limit)
            ax.set_title(f'{title}: {numpy.sum(z1 > 0)} of {len(z1)} and '
                         f'{numpy.sum(z2 <= 0)} of {len(z2)} wrong', fontsize=10)
            ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel(r'$f(\boldsymbol{x}) = \boldsymbol{w}^\top \boldsymbol{x} + b$')
    fig.tight_layout()
    plt.show()


def plot_contributions(fits, titles, examples, names, shape, m, border=2):
    """Top row: the directions w of the fits (w, b) as images. One row per
    example image x below: the image and, for each w, the terms
    w_i (x_i - m_i) of f(x) = w^T (x - m), m the midpoint of the class means
    (red: towards class 2, blue: towards class 1). The dashed line separates
    the outer `border` pixels of the image; the titles split f(x) into the
    terms outside and inside it."""
    rows, cols = 1 + len(examples), 1 + len(fits)
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, (1.8 * shape[0] / shape[1] + 0.6) * rows))
    outer = numpy.zeros(shape, dtype=bool)
    outer[:border] = outer[-border:] = True
    outer[:, :border] = outer[:, -border:] = True
    outer = outer.ravel()

    def signed(ax, image, title):
        vmax = numpy.abs(image).max()
        ax.imshow(image.reshape(shape), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9)

    axes[0, 0].text(0.5, 0.5, r'$\boldsymbol{w}$', fontsize=14, ha='center', va='center')
    for ax, (w, b), title in zip(axes[0, 1:], fits, titles):
        signed(ax, w, title)
    for row, (x, name) in enumerate(zip(examples, names), start=1):
        axes[row, 0].imshow(x.reshape(shape), cmap='gray')
        axes[row, 0].set_title(name, fontsize=9)
        for ax, (w, b) in zip(axes[row, 1:], fits):
            terms = w * (x - m)
            signed(ax, terms, f'$f(\\boldsymbol{{x}})$ = {terms.sum():.2f}\n'
                              f'border {terms[outer].sum():+.2f}, inside {terms[~outer].sum():+.2f}')
            ax.add_patch(plt.Rectangle((border - .5, border - .5), shape[1] - 2 * border,
                                       shape[0] - 2 * border, fill=False, ls='--', lw=1, color='0.3'))
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def render_faces(x, shape, titles=None, signed=False, size=1.5):
    """Show the rows of x as images of the given shape side by side: gray
    values, or with signed=True weights (red positive, blue negative, white 0)."""
    x = numpy.atleast_2d(x)
    fig, axes = plt.subplots(1, len(x), figsize=(size * len(x),
                                                 size * shape[0] / shape[1] + (.3 if titles else 0)),
                             squeeze=False)
    for i, ax in enumerate(axes[0]):
        image = x[i].reshape(shape)
        if signed:
            vmax = numpy.abs(image).max()
            ax.imshow(image, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(image, cmap='gray')
        ax.axis('off')
        if titles:
            ax.set_title(titles[i], fontsize=10)
    plt.show()
