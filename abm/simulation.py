
import numpy as np
import matplotlib.pyplot as plt

from abm.flow_field import FlowField
from abm.endothelial_cell import EndothelialCell
import copy

def run_sim(cfg, lut, label='cell', n_steps=500, print_every=100,
            plot=False, print_spring=False, conditions_override=None):
    """
    Run a single-cell simulation and return full state history.
    
    cfg:                 config dict
    lut:                 RhoLookupTable (or PassiveLUT)
    label:               name for print output
    n_steps:             number of steps to run
    print_every:         print summary every N steps
    plot:                if True, plot cell shape at end
    conditions_override: dict of hill_params overrides for knockouts
                         e.g. {'DSP': {'knocked_out': True}}
    
    Returns: dict with final state + full history
    """

    # Apply knockout overrides
    cfg = copy.deepcopy(cfg)
    if conditions_override:
        for protein, overrides in conditions_override.items():
            cfg['hill_params'][protein].update(overrides)

    # Build cell and flow
    flow = FlowField(magnitude=cfg['mechanics']['f_magnitude'])
    cell = EndothelialCell(
        cell_id=0,
        centroid=np.array([0.0, 0.0]),
        lut=lut, cfg=cfg,
        n_nodes=cfg['sim']['n_nodes'],
        radius=cfg['sim']['radius'],
        flow_direction = flow.direction
    )

    # History storage
    history = {
        'step': [], 'ar': [], 'orientation': [],
        'rho_balance': [], 'k_active': [], 'lsf': [],
        't_cortex': [], 't_sf': [], 'area_err': [],
    }

    # Header
    print(f"\n{'='*70}")
    print(f"  {label}  ({n_steps} steps, f={cfg['mechanics']['f_magnitude']})")
    print(f"{'='*70}")
    print(f"{'step':>6} {'AR':>6} {'orient':>8} {'bal':>7} "
          f"{'k':>6} {'lsf':>6} {'t_sf':>7} {'area':>6}")

    for step in range(n_steps):
        cell.step(flow, dt=cfg['sim']['dt'])

        if step % print_every == 0 or step == n_steps - 1:
            s   = cell.get_state()
            lat, _ = cell._spring_populations()
            history['step'].append(step)
            history['ar'].append(s['metrics']['ar'])
            history['orientation'].append(s['metrics']['orientation'])
            history['rho_balance'].append(s['signalling']['rho_balance'])
            history['k_active'].append(s['remodelling']['mean_k_active'])
            history['lsf'].append(s['remodelling']['mean_lsf_ratio'])
            history['t_cortex'].append(s['mechanics']['t_cortex'])
            history['t_sf'].append(s['mechanics']['t_sf'])
            history['area_err'].append(s['metrics']['area_err'])

            print(f"{step:>6} {s['metrics']['ar']:>6.3f} "
                  f"{s['metrics']['orientation']:>8.1f}° "
                  f"{s['signalling']['rho_balance']:>+7.3f} "
                  f"{s['remodelling']['mean_k_active']:>6.3f} "
                  f"{s['remodelling']['mean_lsf_ratio']:>6.3f} "
                  f"{s['mechanics']['t_sf']:>7.4f} "
                  f"{s['metrics']['area_err']:>6.3f}")

    if print_spring:
        # Final spring table
        print(f"\n--- Spring state at step {n_steps} ---")
        print(f"{'id':>3} {'pop':>8} {'align':>7} {'T_tot':>7} "
            f"{'T_cort':>7} {'T_sf':>6} {'k':>6} "
            f"{'lsf':>6} {'RhoA':>6} {'RhoC':>6}")

        lateral, polar = cell._spring_populations()
        for s in cell.springs:
            pop = 'lateral' if s in lateral else 'polar'
            print(f"{s.id:>3} {pop:>8} {s._init_alignment:>7.3f} "
                f"{s.tension_total:>7.3f} {s.tension_cortex:>7.3f} "
                f"{s.tension_sf:>6.3f} {s.k_active:>6.3f} "
                f"{s.L_sf/s.L_cortex:>6.3f} "
                f"{s.P_RhoA:>6.3f} {s.P_RhoC:>6.3f}")

    if plot:
        plot_cell(cell, title=label)

    return {
        'cell':    cell,
        'history': history,
        'final':   cell.get_state(),
        'label':   label,
    }


def plot_cell(cell, title=''):
    """
    Minimal cell plot — nodes and springs on axes only.
    Call standalone or via run_sim(plot=True).
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Springs
    for s in cell.springs:
        p1, p2 = s.node_1.pos, s.node_2.pos
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                'b-', linewidth=1.2, alpha=0.6)

    # Nodes coloured by role
    colors = {'upstream': 'red', 'downstream': 'orange', 'lateral': 'steelblue'}
    for n in cell.nodes:
        ax.scatter(*n.pos, c=colors[n.role], s=50, zorder=4)

    ax.scatter(*cell.centroid, marker='x', c='gray', s=60, zorder=5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.4, linestyle='--')
    ax.set_title(title or 'cell shape')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()