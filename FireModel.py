from mesa import Model
from mesa.space import SingleGrid
from ForestCell import ForestCell, FuelType, CellState
from enum import Enum

class FireModel(Model):
    def __init__(self, width, height):
        super().__init__()
        self.grid = SingleGrid(width, height, torus=False)
        #tymczasowe przygotowanie typów paliwa
        self.fuel_grass = FuelType(name="grass", burn_time=10, color="green")

        for (content, (x,y)) in self.grid.coord_iter():
            fuel = self.fuel_grass
            state = CellState.Fuel
            cell = ForestCell((x, y), self, fuel, state)
            self.grid.place_agent(cell, (x, y))
            self.agents.add(cell)

    def step(self):
        for agent in self.agents.shuffle():
            agent.step()
        for agent in self.agents.shuffle():
            agent.advance()
