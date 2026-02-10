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

    # Add all possible knockout combinations for procided nodes to perturbation list. 
    for r in range(1, len(nodes) + 1):
        for combo in combinations(nodes, r):
            name = "_".join(combo) + "_ko"

            #print(f"name: {name} for combo {combo}")
            m = base_model.copy()

            for node in combo:
                # Simulate knockout
                m.mutate(node, "OFF") 
                m.update_parameters(**{f"${node}_base": 0.0})

            pertubations[name] = m

    return pertubations
