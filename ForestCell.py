from enum import Enum, auto
from typing import Tuple

from mesa import Agent
class CellState(Enum):
    Empty = 0
    Fuel = 1
    Burning = 2
    Burned = 3

class FuelType:
    def __init__(self, name, burn_time, color):
        self.name = name
        self.burn_time = burn_time
        self.color = color

    def __str__(self):
        return f"Fuel type: {self.name}, burn time: {self.burn_time}"

class ForestCell(Agent):
    def __init__(self, pos:Tuple[int, int], model, fuel:FuelType, state:CellState):
        self.pos = None #potrzebne z powodu wymagań biblioteki
        self.unique_id = pos #unikalne id też dla biblioteki
        self.model = model
        self.fuel = fuel
        self.state = state
        self.burn_timer = int(fuel.burn_time) if state == CellState.Burning else 0
        self.next_state = state

    def is_burnable(self) -> bool:
        if self.fuel.name == "water" or self.state != CellState.Fuel:
            return False
        else:
            return True

    def burning_chance(self):
        burning_chance = 0
        for neighbour in self.model.grid.get_neighbors(self.pos, moore=True, include_center=False): #moore czy w 8 kierunkach patrzymy, include center czy uwzgledniamy sami siebie
            if isinstance(neighbour, ForestCell) and neighbour.state == CellState.Burning:
                burning_chance += 0.1
        return burning_chance

    def step(self):
        self.next_state = self.state
        if self.state == CellState.Burning:
            self.burn_timer -= 1
            if self.burn_timer <= 0:
                self.next_state = CellState.Burned
            return
        if self.is_burnable() == True:
            p = self.burning_chance()
            if self.model.random.random() < p:
                self.next_state = CellState.Burning

    def advance(self):
        prev = self.state
        self.state = self.next_state

        if prev != CellState.Burning and self.state == CellState.Burning:
            self.burn_timer = int(self.fuel.burn_time)


