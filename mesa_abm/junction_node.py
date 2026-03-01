from mesa import Agent
import numpy as np

class JunctionNode(Agent):

    def __init__(self, unique_id, model, pos, cell):
        super().__inirt__(unique_id, model)
        self.pos = np.array(pos)
        self.cell = cell
        self.rho_delta = 0.0

    def step(): 
        pass