import numpy as np

# ------------------------------------------------------------------
# Spring Class Helpers
# ------------------------------------------------------------------
def hill(tau, K, n):
    """
    Hill activation function. Maps mechanical stimulus (tau) to 
    protein recruitment probability. 

    tau: Mechnical stimulus – a tension magnitude
    K: Half Activation Threshold – the tau at which recruitment = 0.5.
    n: Hill coefficient – controls spring sharpness.
    """
    if tau <= 0: 
        return 0.0 # No recruitment under compression. 
    
    return tau**n / (K**n + tau**n)

def get_protein_recruitment(cfg, tau, protein, perturbation='WT'):
    """
    Look up Hill parameters for protein and return recruitment probability.
    """
    params = cfg['hill_params'][protein]

    if params.get('knocked_out', False):
        print(f">>> DEBUG: {protein} is knocked out, recruitment is 0")
        return 0.0 # No recruitment if protein is knocked out regardless of tau.

    p_raw = hill(tau, params['K'], params['n'])

    # Scale to physiological range from MaBoSS
    p_max = params.get('p_max', 1.0) 
    return p_raw * p_max

def bilinear_tension(l_current, l_rest, k_tensile, kc_ratio):
    """
    Bilinear elastic law for single spring component.

    l_current: Current physical length of the spring
    l_rest: Rest length – length when spring is stress free
    k_tensile: Stiffness in the stretched regime
    kc_ratio: Compressive stiffness as a fraction of tensile stiffness. 
              0.1 means 10x softer in compression
    """

    extension = l_current - l_rest # current_length - rest_length

    if extension > 0:
        # Stretching Regime: tension > 0, pulls nodes together
        return k_tensile * extension 
    else:
        # Compression Regime: tension < 0, pushes nodes aåart 
        k_comp = k_tensile * kc_ratio
        return k_comp * extension 

# ------------------------------------------------------------------
# Cell Analysis
# ------------------------------------------------------------------

def measure_shape(cell) -> dict:
    """
    PCA-based shape descriptors matching ImageJ conventions.

    Aspect ratio = 2√λ_major / 2√λ_minor
        λ from covariance matrix of node positions.
        1.0 = circle, >1.0 = elongated.

    Orientation = angle of major eigenvector to x-axis (flow direction).
        0° = elongated along flow (correct response).
        90° = elongated perpendicular (incorrect).
        Note: PCA eigenvectors have no preferred sign — use
        elongation_index for a consistent signed measure.

    Circularity = 4π × area / perimeter²
        1.0 = perfect circle, →0 = elongated/irregular.
        Matches ImageJ circularity metric exactly — allows direct
        comparison with experimental microscopy data.

    Elongation index = (AR - 1) × |cos(orientation)|
        Combines AR and alignment into one number.
        0 = circular or perpendicular, positive = elongated along flow.
        Use this as your primary phenotype classifier in time series.
    """
    pos = cell.positions
    centered = pos - pos.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    eigvals = np.maximum(eigvals, 0.0)

    major_vec = eigvecs[:, 1]
    major = 2.0 * np.sqrt(eigvals[1])
    minor = 2.0 * np.sqrt(eigvals[0])

    perim = np.sum(np.linalg.norm(
        np.diff(np.vstack([pos, pos[0]]), axis=0), axis=1
    ))

    ar = major / (minor + 1e-10)
    orientation = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    elong_idx   = (ar - 1.0) * abs(np.cos(np.radians(orientation)))

    return {
        'aspect_ratio':     round(ar, 3),
        'orientation':      round(orientation, 2),
        'circularity':      round(4.0 * np.pi * cell.current_area /
                                  (perim ** 2 + 1e-10), 3),
        'elongation_index': round(elong_idx, 4),
        'perimeter':        round(perim, 4),
    }


def classify_phenotype(summary: dict) -> str:
    """
    Classify cell into one of three phenotypes from a get_cell_summary()
    output. Thresholds are starting points — calibrate against your
    MaBoSS knockout experiments.

    Returns: 'failed' | 'normal' | 'hyper'
    """
    ar = summary['metrics']['ar']
    balance = summary['signalling']['rho_balance']

    if ar < 1.4 or balance < -0.1:
        return 'failed'
    elif ar > 3.5 or balance > 0.25:
        return 'hyper'
    else:
        return 'normal'