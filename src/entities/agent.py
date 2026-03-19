import math
import pygame

from src.core.settings import (
    AGENT_SIZE,
    COLOR_BLUE,
)
from src.core.utils import clamp, normalize_angle


class Agent:
    def __init__(self, x, y, angle, genome, network, params, world_width, world_height):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0
        self.alive = True
        self.genome = genome
        self.network = network
        self.params = params
        self.world_width = world_width
        self.world_height = world_height
        self.energy = self.params.initial_energy

    def get_nearest_food(self, foods):
        nearest = None
        nearest_dist_sq = float("inf")
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest = food
        return nearest, math.sqrt(nearest_dist_sq)

    def build_inputs(self, foods):
        nearest, distance = self.get_nearest_food(foods)
        dx = nearest.x - self.x
        dy = nearest.y - self.y

        max_distance = math.hypot(self.world_width, self.world_height)
        distance_norm = clamp(distance / max_distance, 0.0, 1.0)

        food_angle = math.atan2(dy, dx)
        relative_angle = normalize_angle(food_angle - self.angle)
        relative_angle_norm = relative_angle / math.pi

        energy_norm = clamp(self.energy / max(self.params.max_energy, 1e-6), 0.0, 1.0)

        return distance_norm, relative_angle_norm, energy_norm

    def update(self, foods, dt_60):
        if not self.alive:
            return []

        inputs = self.build_inputs(foods)
        output_turn, output_move = self.network.activate(inputs)

        turn_signal = clamp(output_turn, -1.0, 1.0)
        move_signal = clamp((output_move + 1.0) / 2.0, 0.0, 1.0)

        self.angle += turn_signal * self.params.turn_rate * dt_60

        self.speed += move_signal * self.params.acceleration * dt_60
        self.speed *= self.params.friction ** dt_60
        self.speed = clamp(self.speed, 0.0, self.params.max_speed)

        self.x += math.cos(self.angle) * self.speed * dt_60
        self.y += math.sin(self.angle) * self.speed * dt_60

        self.x %= self.world_width
        self.y %= self.world_height

        energy_cost = self.params.base_energy_drain + self.speed * self.params.move_energy_drain_factor
        self.energy -= energy_cost * dt_60

        return inputs

    def try_eat(self, foods):
        if not self.alive:
            return False

        for food in foods:
            if math.hypot(food.x - self.x, food.y - self.y) <= self.params.eat_distance:
                food.respawn()
                self.energy = clamp(self.energy + self.params.food_energy_gain, 0.0, self.params.max_energy)
                self.genome.fitness += self.params.food_fitness_reward
                return True
        return False

    def check_alive(self):
        if self.energy <= 0.0 and self.alive:
            self.alive = False
            self.speed = 0.0
            self.genome.fitness -= self.params.death_fitness_penalty

    def draw(self, surface):
        if not self.alive:
            return

        tip = (
            self.x + math.cos(self.angle) * AGENT_SIZE,
            self.y + math.sin(self.angle) * AGENT_SIZE,
        )
        left = (
            self.x + math.cos(self.angle + 2.45) * AGENT_SIZE * 0.75,
            self.y + math.sin(self.angle + 2.45) * AGENT_SIZE * 0.75,
        )
        right = (
            self.x + math.cos(self.angle - 2.45) * AGENT_SIZE * 0.75,
            self.y + math.sin(self.angle - 2.45) * AGENT_SIZE * 0.75,
        )

        pygame.draw.polygon(surface, COLOR_BLUE, [tip, left, right])
