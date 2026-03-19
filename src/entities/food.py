import random

import pygame

from src.core.settings import COLOR_GREEN, FOOD_RADIUS


class Food:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = 0.0
        self.y = 0.0
        self.respawn()

    def respawn(self):
        self.x = random.uniform(FOOD_RADIUS, self.width - FOOD_RADIUS)
        self.y = random.uniform(FOOD_RADIUS, self.height - FOOD_RADIUS)

    def draw(self, surface):
        pygame.draw.circle(surface, COLOR_GREEN, (int(self.x), int(self.y)), FOOD_RADIUS)
