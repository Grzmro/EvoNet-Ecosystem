from src.core.settings import (
    FPS,
    COLOR_WORLD,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from src.entities.food import Food


class Simulation:
    def __init__(self, agents, params):
        self.agents = agents
        self.params = params
        self.foods = [Food(WORLD_WIDTH, WORLD_HEIGHT) for _ in range(self.params.food_count)]
        self.elapsed_time = 0.0
        self.finished = False

    def sync_food_count(self):
        target = int(self.params.food_count)
        current = len(self.foods)

        if current < target:
            for _ in range(target - current):
                self.foods.append(Food(WORLD_WIDTH, WORLD_HEIGHT))
        elif current > target:
            del self.foods[target:]

    def update(self, dt):
        if self.finished:
            return

        self.sync_food_count()

        dt_60 = dt * FPS
        self.elapsed_time += dt

        alive_count = 0
        for agent in self.agents:
            if not agent.alive:
                continue

            alive_count += 1
            agent.update(self.foods, dt_60)
            agent.genome.fitness += self.params.survival_fitness_per_frame
            agent.try_eat(self.foods)
            agent.check_alive()

        if self.elapsed_time >= self.params.max_generation_time or alive_count == 0:
            self.finished = True

    def alive_count(self):
        return sum(1 for agent in self.agents if agent.alive)

    def best_fitness(self):
        return max((agent.genome.fitness for agent in self.agents), default=0.0)

    def draw_world(self, world_surface):
        world_surface.fill(COLOR_WORLD)

        for food in self.foods:
            food.draw(world_surface)
        for agent in self.agents:
            agent.draw(world_surface)
