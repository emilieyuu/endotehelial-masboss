# abm/api.py
import numpy as np

def apply_flow_to_cell(field, cell):
    """
    Compute and write mechanical state onto every membrane node. 
    Called once per timestep before signalling updates. 

    Writes to each node: shear, normal_force, face_type, outward_normal, tangent, flow_force
    """

    positions = cell.positons
    centre = positions.mean(axis=0)

    outward_normals = field.compute_normals(positions, centre)
    face_types = field.classify_faces(outward_normals)

    for i, node in enumerate(cell.nodes):
        shear, normal_force, tangent, flow_force = field.compute_node_forces(outward_normals[i])

        node.update_from_flow(shear, normal_force, outward_normals[i], 
                              face_types[i], tangent, flow_force)
        print(f">>> DEBUG: [FlowField] Applied to cell {cell.id} | "
              f"centre={centre.round(2)} | "
              f"upstream={np.sum(face_types=='upstream')} "
              f"lateral={np.sum(face_types=='lateral')} "
              f"downstream={np.sum(face_types=='downstream')}")