# From https://gist.github.com/thesamovar/52dbbb3a58a73c590d54c34f5f719bac

def panel_specs(layout, fig=None):
    """
    Args:
        layout:
        fig:
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    # default arguments
    if fig is None:
        fig = plt.gcf()
    # format and sanity check grid
    lines = layout.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    linewidths = set(len(line) for line in lines)
    if len(linewidths) > 1:
        raise ValueError('Invalid layout (all lines must have same width)')
    width = linewidths.pop()
    height = len(lines)
    panel_letters = set(c for line in lines for c in line) - set('.')
    # find bounding boxes for each panel
    panel_grid = {}
    for letter in panel_letters:
        left = min(x for x in range(width) for y in range(height) if lines[y][x] == letter)
        right = 1 + max(x for x in range(width) for y in range(height) if lines[y][x] == letter)
        top = min(y for x in range(width) for y in range(height) if lines[y][x] == letter)
        bottom = 1 + max(y for x in range(width) for y in range(height) if lines[y][x] == letter)
        panel_grid[letter] = (left, right, top, bottom)
        # check that this layout is consistent, i.e. all squares are filled
        valid = all(lines[y][x] == letter for x in range(left, right) for y in range(top, bottom))
        if not valid:
            raise ValueError('Invalid layout (not all square)')
    # build axis specs
    gs = gridspec.GridSpec(ncols=width, nrows=height, figure=fig)
    specs = {}
    for letter, (left, right, top, bottom) in panel_grid.items():
        specs[letter] = gs[top:bottom, left:right]
    return specs, gs


def panels(layout, fig=None):
    """
    Args:
        layout:
        fig:
    """
    import matplotlib.pyplot as plt
    # default arguments
    if fig is None:
        fig = plt.gcf()
    specs, gs = panel_specs(layout, fig=fig)
    axes = {}
    for letter, spec in specs.items():
        axes[letter] = fig.add_subplot(spec)
    return axes, gs


def label_panel(ax, letter, *,
                offset_left=0.8, offset_up=0.2, prefix='', postfix='.', **font_kwds):
    """
    Args:
        ax:
        letter:
        offset_left:
        offset_up:
        prefix:
        postfix:
        **font_kwds:
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    kwds = dict(fontsize=18)
    kwds.update(font_kwds)
    # this mad looking bit of code says that we should put the code offset a certain distance in
    # inches (using the fig.dpi_scale_trans transformation) from the top left of the frame
    # (which is (0, 1) in ax.transAxes transformation space)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(-offset_left, offset_up, fig.dpi_scale_trans)
    ax.text(0, 1, prefix + letter + postfix, transform=trans, **kwds)


def label_panels(axes, letters=None, **kwds):
    """
    Args:
        axes:
        letters:
        **kwds:
    """
    import matplotlib.pyplot as plt
    if letters is None:
        letters = axes.keys()
    for letter in letters:
        ax = axes[letter]
        label_panel(ax, letter, **kwds)


#  Example
# layout = '''
#  AAB
#  AA.
#  .CC
#  '''
# fig = plt.figure(figsize=(10, 7))
# axes, spec = panels(layout, fig=fig)
# spec.set_width_ratios([1, 3, 1])
# label_panels(axes, letters='ABC')
# plt.tight_layout()

#  Example
# layout = '''
#  AAAB
#  CDEB
#  '''
# fig = plt.figure(figsize=(10, 5))
# axes, spec = panels(layout, fig=fig)
# label_panels(axes)
# plt.tight_layout()

def tight_xticklabels(ax=None):
    """
    Args:
        ax:
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_ha('left')
    ticklabels[0].set_text(' ' + ticklabels[0].get_text())
    ticklabels[-1].set_ha('right')
    ticklabels[-1].set_text(ticklabels[-1].get_text() + ' ')
    ax.set_xticklabels(ticklabels)


def tight_yticklabels(ax=None):
    """
    Args:
        ax:
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    ticklabels = ax.get_yticklabels()
    ticklabels[0].set_va('bottom')
    ticklabels[-1].set_va('top')

# Example:
# layout = '''
#     AAAB
#     AAAC
#     AAAD
#     '''
# N = 5
# fig = plt.figure(figsize=(8, 6))
# specs, gs = panel_specs(layout, fig=fig)
# axes = {}
# for letter in 'BCD':
#     axes[letter] = ax = fig.add_subplot(specs[letter])
#     label_panel(ax, letter)
# subgs = specs['A'].subgridspec(N, N, wspace=0, hspace=0)
# triaxes = {}
# tighten = []
# for i in range(N):
#     for j in range(i+1):
#         triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
#         if i==N-1:
#             ax.set_xlabel(chr(ord('α')+j))
#             tighten.append((tight_xticklabels, ax))
#         else:
#             ax.set_xticks([])
#         if j==0:
#             ax.set_ylabel(chr(ord('α')+i))
#             tighten.append((tight_yticklabels, ax))
#         else:
#             ax.set_yticks([])
# label_panel(triaxes[0, 0], 'A')
# plt.tight_layout()
# for f, ax in tighten:
#     f(ax)
# plt.tight_layout()