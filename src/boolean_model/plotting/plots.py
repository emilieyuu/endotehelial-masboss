#plots.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def plot_rhos(name, df, save=True, outdir=None, show=False):
    """
    Plot RhoA/RhoC activation probabilities/balance over time. 
    
    :param name: Description
    :param df: Description
    :param save: Description
    :param outdir: Description
    :param show: Description
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(df["RhoA"], label="RhoA")
    ax.plot(df["RhoC"], label="RhoC")

    ax.set_title(f"{name}: RhoA / RhoC activation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.legend()

    fig.tight_layout()

    if save and outdir is not None:
        outpath = outdir / f"{name}_rho_activation.png"
        fig.savefig(outpath, dpi=300)

    if show: 
        plt.show()

    plt.close(fig)
    return fig


def plot_delta(name, df, eps=0.20, save=True, outdir=None, show=False):
    """
    Plot Delta compared to eps threshold over time. 
    
    :param name: Description
    :param df: Description
    :param eps: Description
    :param save: Description
    :param outdir: Description
    :param show: Description
    """
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(df.index, df["delta"], label="Δ(t)")

    ax.axhline(eps, linestyle="--", label="+ε", color='red')
    ax.axhline(-eps, linestyle="--", label="-ε", color='red')
    ax.axhline(0, linestyle=":", color='grey')

    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"{name} – Rho balance")
    ax.set_xlabel("Time")
    ax.set_ylabel("Δ = P(RhoC) − P(RhoA)")
    ax.legend()

    plt.tight_layout()

    if save and outdir is not None:
        outpath = outdir / f"{name}_delta_time.png"
        fig.savefig(outpath, dpi=300)

    if show: 
        plt.show()
        
    plt.close(fig)
    return fig

def plot_full_ss(df, eps=0.25, save=True, outdir=None, show=False):
    fig, ax = plt.subplots(figsize=(6,4))

    df["delta"].plot(kind="bar", ax=ax)

    ax.axhline(eps, linestyle="--", label="+ε", color='red')
    ax.axhline(-eps, linestyle="--", label="-ε", color='red')
    ax.axhline(0, linestyle=":", color='grey')

    ax.set_ylabel("Steady-state Δ")
    ax.set_title("Steady-state Rho balance across perturbations")
    ax.legend()

    plt.tight_layout()
    plt.show()

    if save and outdir is not None:
        outpath = outdir / "ss_balance.png"
        fig.savefig(outpath, dpi=300)

    if show: 
        plt.show()

    plt.close(fig)