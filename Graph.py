import numpy as np
import matplotlib.pyplot as plt

def plot_ConvLSTM_curves():
    # Values copied from the ConvLSTM log
    train_mse = [4.479, 2.424, 2.022, 1.746, 1.650, 1.620, 1.549, 1.506, 1.483, 1.482,
                 1.459, 1.451, 1.436, 1.423, 1.420, 1.408, 1.396, 1.378, 1.382, 1.364]
    val_rmse = [1.561, 1.379, 1.253, 1.197, 1.215, 1.153, 1.136, 1.120, 1.121, 1.099,
                1.102, 1.099, 1.086, 1.093, 1.080, 1.080, 1.069, 1.102, 1.069, 1.093]
    Test_Rmse = 1.136
    epochs = np.arange(1, len(train_mse) + 1)
    train_rmse = np.sqrt(train_mse)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_rmse, marker='o', label='Train RMSE')
    plt.plot(epochs, val_rmse, marker='s', label='Val RMSE')

    # Plot a horizontal line for the test set RMSE
    plt.axhline(Test_Rmse, linestyle='--', alpha=0.6, label=f'Test RMSE {Test_Rmse:.3f}')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training / Validation RMSE (ConvLSTM)')
    plt.legend()
    plt.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.show()



def plot_informer_curves():

    # Values copied from the Informer log
    train_mse = np.array([
        6.955, 3.143, 2.842, 2.505, 2.272, 2.094, 1.950, 1.827, 1.731, 1.658,
        1.605, 1.562, 1.522, 1.492, 1.469, 1.452, 1.440, 1.431, 1.426, 1.425
    ])
    val_rmse = np.array([
        1.791, 1.714, 1.625, 1.543, 1.497, 1.468, 1.405, 1.408, 1.389, 1.377,
        1.367, 1.348, 1.337, 1.330, 1.332, 1.331, 1.333, 1.328, 1.331, 1.331
    ])
    test_rmse = 1.355

    epochs = np.arange(1, len(train_mse) + 1)
    train_rmse = np.sqrt(train_mse)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_rmse, marker='o', label='Train RMSE')
    plt.plot(epochs, val_rmse,  marker='s', label='Val RMSE')

    # Plot a horizontal line for the test set RMSE
    plt.axhline(test_rmse, linestyle='--', alpha=0.6, label=f'Test RMSE {test_rmse:.3f}')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training / Validation RMSE (Informer)')
    plt.legend()
    plt.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.show()



def plot_PatchTST_grid_series_curves():

    # Values copied from the PatchTST log
    train_mse_grid = np.array([
        5.069, 2.890, 2.735, 2.648, 2.570, 2.490, 2.415, 2.330, 2.256, 2.190,
        2.126, 2.059, 2.011, 1.965, 1.929, 1.894, 1.874, 1.857, 1.848, 1.844
    ])
    val_rmse_grid = np.array([
        1.726, 1.673, 1.650, 1.652, 1.658, 1.657, 1.667, 1.685, 1.676, 1.692,
        1.689, 1.698, 1.700, 1.711, 1.713, 1.712, 1.721, 1.722, 1.716, 1.720
    ])
    val_rmse_series = np.array([
        0.263, 0.224, 0.207, 0.214, 0.227, 0.206, 0.219, 0.237, 0.180, 0.185,
        0.161, 0.196, 0.153, 0.152, 0.156, 0.153, 0.155, 0.154, 0.153, 0.150
    ])
    test_rmse_grid = 1.740  # Test RMSE（grid）

    epochs = np.arange(1, len(train_mse_grid) + 1)

    # Turn Train_MSE(grid) to Train_RMSE(grid)
    train_rmse_grid = np.sqrt(train_mse_grid)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_rmse_grid, marker='o', label='Train RMSE (grid)')
    plt.plot(epochs, val_rmse_grid,  marker='s', label='Val RMSE (grid)')
    plt.plot(epochs, val_rmse_series, marker='^', label='Val RMSE (series)')

    # Plot a horizontal line for the test set RMSE
    plt.axhline(test_rmse_grid, linestyle='--', alpha=0.7,
                label=f'Test RMSE (grid) {test_rmse_grid:.3f}')


    plt.xlabel('Epoch'); plt.ylabel('RMSE')
    plt.title('Training / Validation RMSE (PatchTST)')
    plt.legend(); plt.grid(alpha=0.3, ls='--')
    plt.tight_layout(); plt.show()


def plot_Patch_Informer_grid_series_curves():

    # Values copied from the Patch-Informer log
    train_mse_grid = np.array([
        5.928, 3.351, 3.062, 2.834, 2.602, 2.398, 2.232, 2.106, 1.998, 1.902,
        1.817, 1.746, 1.689, 1.637, 1.594, 1.558, 1.525, 1.494, 1.469, 1.443
    ])

    val_rmse_grid = np.array([
        1.865, 1.757, 1.703, 1.636, 1.582, 1.533, 1.512, 1.469, 1.437, 1.433,
        1.392, 1.392, 1.368, 1.351, 1.346, 1.355, 1.343, 1.333, 1.326, 1.341
    ])

    val_rmse_series = np.array([
        0.411, 0.312, 0.293, 0.313, 0.328, 0.304, 0.351, 0.337, 0.326, 0.384,
        0.322, 0.394, 0.325, 0.324, 0.327, 0.386, 0.371, 0.340, 0.329, 0.375
    ])

    test_rmse_grid = 1.352  # Test RMSE（grid）
    test_rmse_series = 0.313  # Test RMSE（series）

    epochs = np.arange(1, len(train_mse_grid) + 1)

    # Turn Train_MSE(grid) to Train_RMSE(grid)
    train_rmse_grid = np.sqrt(train_mse_grid)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_rmse_grid, marker='o', label='Train RMSE (grid)')
    plt.plot(epochs, val_rmse_grid, marker='s', label='Val RMSE (grid)')
    plt.plot(epochs, val_rmse_series, marker='^', label='Val RMSE (series)')

    # Plot a horizontal line for the test set RMSE
    plt.axhline(test_rmse_grid, linestyle='--', alpha=0.7,
                label=f'Test RMSE (grid) {test_rmse_grid:.3f}')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training / Validation RMSE (Patch_Informer)')
    plt.legend()
    plt.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.show()


#Boxplot for Test_RMSE
def boxplot1():
    import numpy as np
    import matplotlib.pyplot as plt

    #---- Val-RMSE (grid) arrays ----
    convlstm_val = np.array([
        1.561, 1.379, 1.253, 1.197, 1.215, 1.153, 1.136, 1.120, 1.121, 1.099,
        1.102, 1.099, 1.086, 1.093, 1.080, 1.080, 1.069, 1.102, 1.069, 1.093
    ])
    informer_val = np.array([
        1.791, 1.714, 1.625, 1.543, 1.497, 1.468, 1.405, 1.408, 1.389, 1.377,
        1.367, 1.348, 1.337, 1.330, 1.332, 1.331, 1.333, 1.328, 1.331, 1.331
    ])
    patchtst_val = np.array([
        1.726, 1.673, 1.650, 1.652, 1.658, 1.657, 1.667, 1.685, 1.676, 1.692,
        1.689, 1.698, 1.700, 1.711, 1.713, 1.712, 1.721, 1.722, 1.716, 1.720
    ])
    patchinf_val = np.array([
        1.865, 1.757, 1.703, 1.636, 1.582, 1.533, 1.512, 1.469, 1.437, 1.433,
        1.392, 1.392, 1.368, 1.351, 1.346, 1.355, 1.343, 1.333, 1.326, 1.341
    ])

    #---- Test RMSE  ----
    convlstm_test = 1.136
    informer_test = 1.355
    patchtst_test = 1.740
    patchinf_test = 1.352

    labels = ["ConvLSTM", "Informer", "PatchTST", "Patch-Informer"]
    val_sets = [convlstm_val, informer_val, patchtst_val, patchinf_val]
    test_vals = [convlstm_test, informer_test, patchtst_test, patchinf_test]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))


    ax.boxplot(val_sets, positions=np.arange(1, 5), widths=0.55, patch_artist=False)

    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Val RMSE (grid)")
    ax.set_title("Validation RMSE (grid) across 4 models", fontsize=11)


    x_pts = np.arange(1, 5) + 0.18
    ax.plot(x_pts, test_vals, linestyle='None', marker='o', markersize=4, label="Test RMSE")


    ymax_per_box = [v.max() for v in val_sets]
    for i, (x, t) in enumerate(zip(x_pts, test_vals), start=1):
        dx = -28 if i == 4 else 8
        dy = 8
        ax.annotate(
            f"Test = {t:.3f}",
            xy=(x, t),
            xytext=(dx, dy),
            textcoords='offset points',
            ha='left' if dx > 0 else 'right',
            va='bottom',
            fontsize=8,
            arrowprops=dict(arrowstyle='-', lw=0.6)
        )

    ax.grid(alpha=0.3, ls='--', axis='y')
    fig.tight_layout()
    plt.show()


def boxplot_test_rmse(
    labels=None,
    val_sets=None,
    test_vals=None,
    title="Validation RMSE (grid) across 4 models",
    ylabel="Val RMSE (grid)",
    figsize=(8.5, 4.8),
    savepath=None,
    show=True,
):

    # Results from training
    if labels is None or val_sets is None or test_vals is None:
        labels = ["ConvLSTM", "Informer", "PatchTST", "Patch-Informer"]

        convlstm_val = np.array([
            1.561, 1.379, 1.253, 1.197, 1.215, 1.153, 1.136, 1.120, 1.121, 1.099,
            1.102, 1.099, 1.086, 1.093, 1.080, 1.080, 1.069, 1.102, 1.069, 1.093
        ])
        informer_val = np.array([
            1.791, 1.714, 1.625, 1.543, 1.497, 1.468, 1.405, 1.408, 1.389, 1.377,
            1.367, 1.348, 1.337, 1.330, 1.332, 1.331, 1.333, 1.328, 1.331, 1.331
        ])
        patchtst_val = np.array([
            1.726, 1.673, 1.650, 1.652, 1.658, 1.657, 1.667, 1.685, 1.676, 1.692,
            1.689, 1.698, 1.700, 1.711, 1.713, 1.712, 1.721, 1.722, 1.716, 1.720
        ])
        patchinf_val = np.array([
            1.865, 1.757, 1.703, 1.636, 1.582, 1.533, 1.512, 1.469, 1.437, 1.433,
            1.392, 1.392, 1.368, 1.351, 1.346, 1.355, 1.343, 1.333, 1.326, 1.341
        ])

        convlstm_test = 1.136
        informer_test = 1.355
        patchtst_test = 1.740
        patchinf_test = 1.352

        val_sets = [convlstm_val, informer_val, patchtst_val, patchinf_val]
        test_vals = [convlstm_test, informer_test, patchtst_test, patchinf_test]


    fig, ax = plt.subplots(figsize=figsize)

    # Boxplot
    positions = np.arange(1, len(labels) + 1)
    ax.boxplot(val_sets, positions=positions, widths=0.55, patch_artist=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

    # Test-RMSE Point
    x_pts = positions + 0.18
    ax.plot(x_pts, test_vals, linestyle='None', marker='o', markersize=4, label="Test RMSE")

    # Nearest Short Arrow
    for idx, (x, t) in enumerate(zip(x_pts, test_vals), start=1):
        dx = 8 if idx < len(labels) else -28
        dy = 8
        ha = 'left' if dx > 0 else 'right'
        ax.annotate(
            f"Test = {t:.3f}",
            xy=(x, t),
            xytext=(dx, dy),
            textcoords='offset points',
            ha=ha, va='bottom', fontsize=8,
            arrowprops=dict(arrowstyle='-', lw=0.6)
        )

    ax.grid(alpha=0.3, ls='--', axis='y')
    fig.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()



def boxplot_paired_rmse():

    # Results from model training
    conv_val = np.array([1.561, 1.379, 1.253, 1.197, 1.215, 1.153, 1.136, 1.120, 1.121, 1.099,
                         1.102, 1.099, 1.086, 1.093, 1.080, 1.080, 1.069, 1.102, 1.069, 1.093])
    inf_val  = np.array([1.791, 1.714, 1.625, 1.543, 1.497, 1.468, 1.405, 1.408, 1.389, 1.377,
                         1.367, 1.348, 1.337, 1.330, 1.332, 1.331, 1.333, 1.328, 1.331, 1.331])
    patch_val_grid = np.array([1.726, 1.673, 1.650, 1.652, 1.658, 1.657, 1.667, 1.685, 1.676, 1.692,
                               1.689, 1.698, 1.700, 1.711, 1.713, 1.712, 1.721, 1.722, 1.716, 1.720])
    pinf_val_grid  = np.array([1.865, 1.757, 1.703, 1.636, 1.582, 1.533, 1.512, 1.469, 1.437, 1.433,
                               1.392, 1.392, 1.368, 1.351, 1.346, 1.355, 1.343, 1.333, 1.326, 1.341])

    models = {
        "ConvLSTM": conv_val,
        "Informer": inf_val,
        "PatchTST (grid)": patch_val_grid,
        "Patch-Informer (grid)": pinf_val_grid,
    }
    baseline = "ConvLSTM"
    names = list(models.keys())
    n = len(conv_val)

    X = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n):
        ys = [models[k][i] for k in names]
        ax.plot(X, ys, alpha=0.15, lw=1, color='C7')

    # Mean and 95%CI
    means = np.array([np.mean(models[k]) for k in names], dtype=float)
    sem   = np.array([np.std(models[k], ddof=1)/np.sqrt(n) for k in names], dtype=float)
    ax.errorbar(X, means, yerr=1.96*sem, fmt='o', color='black', capsize=3,
                label='Mean ±95% CI', zorder=5)

    # Compare to Baseline  ΔRMSE
    base = models[baseline]
    deltas = {k: float(np.mean(models[k] - base)) for k in names if k != baseline}
    delta_text = "\n".join([f"ΔRMSE({k} − {baseline}) = {d:+.3f}" for k, d in deltas.items()])
    ax.text(0.99, 0.02, delta_text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))


    ax.set_xticks(X)
    ax.set_xticklabels(names)
    ax.set_ylabel('Val RMSE (grid)')
    ax.set_title('Per-epoch paired comparison of Val-RMSE (grid) across 4 models', fontsize=12)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()

    plt.show()



if __name__ == "__main__":
    boxplot_test_rmse()
    boxplot1()
    plot_ConvLSTM_curves()   # ConvLSTM
    plot_informer_curves()   # Informer
    plot_PatchTST_grid_series_curves()# PatchTST
    plot_Patch_Informer_grid_series_curves()#Patch-Informer

