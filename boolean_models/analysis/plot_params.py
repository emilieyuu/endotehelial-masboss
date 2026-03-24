import matplotlib.pyplot as plt
import seaborn as sns

def create_subplot_layout(n_panels, width=6, height=5, title=None, adjust=True):
    """ 
    Create a horixontal subplot with share y_axis.

    param n_panels: Number of plots to include.
    return figure and list of axes.
    """
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(width*n_panels, height), sharey=True
    )

    if adjust: 
        fig.subplots_adjust(bottom=0.18, top=0.85, wspace=0.1)

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.95) # Add title if provided
    
    if n_panels == 1: 
        axes = [axes] # Ensures axes is always iterable

    return fig, axes

def add_annotation(fig, x_start=0.1, y_start=0.05, y_space=0.025):
    # Phenotypes (Hyper / Failed)
    fig.text(x_start, y_start, "Hyper = RhoC Dominant", 
             ha='left', va='bottom', fontsize=8.5, color='blue')
    
    fig.text(x_start, y_start - y_space, "Failed = RhoA Dominant", 
             ha='left', va='bottom', fontsize=8.5, color='red')

    # KO meanings
    fig.text(x_start, y_start - y_space*2,
             "DSP KO = reduced RhoA  |  TJP1 KO = reduced RhoC",
             ha='left', va='bottom', fontsize=8.5, color='black', style='italic')

#=======================
# 1D Sweep Plot Functions
#=======================
def plot_param_panel(ax, data, param_name, eps=0.25, metric="delta"):
    """ 
    Plot 1D sensitivity curve for single parameter on provided axis.
    param metric: String; delta or Rho activation probabilities
    """
    # Line plot grouped by perturbation.
    sns.lineplot(
        data=data, x='p1_value', y=metric,
        hue='perturbation', style='perturbation',
        markers=True, dashes=False,
        ax=ax, legend='brief'
    )

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("")  # remove title
        for text in leg.get_texts():
            text.set_fontsize(7)
        leg._legend_box.align = "left"
        # Optional: move inside upper right corner
        leg.set_bbox_to_anchor((1.02, 1))
        leg.set_frame_on(False)

    if metric == 'delta':
        # Add phenotype shading as horizontal bands
        ax.axhspan(eps, 1.0, color='blue', alpha=0.1)
        ax.axhspan(-eps, eps, color='gray', alpha=0.1)
        ax.axhspan(-1.0, -eps, color='red', alpha=0.1)

        ax.text(0.95, 0.85, "Hyper", transform=ax.transAxes, ha="center", color='blue', fontsize=8)
        ax.text(0.95, 0.47, "Normal", transform=ax.transAxes, ha="center", color='dimgray', fontsize=8)
        ax.text(0.95, 0.10, "Failed", transform=ax.transAxes, ha="center", color='red', fontsize=8)

    # Formatting
    ax.set_xlabel(param_name)
    if metric == "delta": 
        ax.set_ylabel("Balance ($\\Delta = RhoC - RhoA$)")
        ax.set_ylim(-0.8, 0.8)
    else:
        ax.set_ylabel(f"Activation Probability ({metric})")
        ax.set_ylim(0, 1)

    ax.set_xlim(left=0)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

def plot_1d(df_1d, metric='delta', title=None, eps=0.25, group=None, outdir=None):

    fig, axes = create_subplot_layout(len(group), title=title)

    for ax, p in zip(axes, group):
        data = df_1d[df_1d['p1_name'] == p]
        plot_param_panel(ax, data, p, eps, metric)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Add bottom annotation strip
    add_annotation(fig, x_start=0)

    # Optional saving
    if outdir is not None:
        group_name = "_".join([p.replace('$','') for p in group])
        outpath = outdir / f"sensitivity_{group_name}.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

#=======================
# 2D Sweep Plot Functions
#=======================

def add_heatmap_colorbar(fig, heatmap):
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4]) # [left, bottom, width, height]
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax)
    cbar.set_label("Delta (RhoC - RhoA)", labelpad=10)
    
    # Colorbar labels
    cbar_ax.text(0.5, 1.06, 'Hyper', color='blue', ha='center', va='top', 
                transform=cbar_ax.transAxes, fontsize=8.5)
    cbar_ax.text(0.5, -0.06, 'Failed', color='red', ha='center', va='bottom', 
                 transform=cbar_ax.transAxes, fontsize=8.5)
 

    
def plot_experiment_heatmaps(exp_df, outdir=None):
    exp_name = exp_df['exp_name'].iloc[0]
    perturbations = exp_df["perturbation"].unique()
    n_perbs = len(perturbations)

    fig, axes = create_subplot_layout(n_panels=n_perbs, width=6, height=6, title=exp_name)
    
    vmin, vmax = exp_df["delta"].min(), exp_df["delta"].max()
    heatmap = None 

    for ax, perb in zip(axes, perturbations):
        perb_df = exp_df[exp_df["perturbation"] == perb]
        pivot_df = perb_df.pivot(index='p1_value', columns='p2_value', values='delta')

        hm = sns.heatmap(
            pivot_df, cmap="RdBu",
            center=0, vmin=vmin, vmax=vmax, cbar=False, ax=ax
        )
        ax.set_title(perb, pad=10)
        ax.set_xlabel(perb_df['p2_name'].iloc[0])
        ax.set_ylabel(perb_df['p1_name'].iloc[0])
        
        if heatmap is None:
            heatmap = hm

    add_heatmap_colorbar(fig, heatmap)
    add_annotation(fig)

    if outdir is not None: 
        plt.savefig(outdir / f"{exp_name}_heatmaps.png", dpi=300, bbox_inches='tight')

    plt.show()