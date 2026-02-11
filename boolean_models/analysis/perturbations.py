# boolean_models/analysis/perturbations.py

from itertools import combinations

def generate_ko_models(base_model, nodes):
    """
    Generate models for different perturbations. 

    :param base_model: A MaBoss model with WT configurations.
    :param nodes: A list of nodes to combine into perturbation scenarios. 

    :return pertubations: A dictionary of each pertubation name and its related MaBoSS object. 
    """

    # Initiate perturbation dict. with WT (no knockouts).
    pertubations = {"WT": base_model}

    # Add all possible single and double knockout combinations. 
    for r in [1, 2]:
        for combo in combinations(nodes, r):
            name = "_".join(combo) 
            
            # Simulate knockout
            m = base_model.copy()
            for node in combo:
                m.mutate(node, "OFF") 
                m.update_parameters(**{f"${node}_base": 0.0})

            pertubations[name] = m

    return pertubations

def generate_ko_names(nodes):
    """
    Return KO scenario names without creating models.
    """
    names = ["WT"]

    for r in range(1, len(nodes) + 1):
        for combo in combinations(nodes, r):
            name = "_".join(combo) 
            names.append(name)

    return names
