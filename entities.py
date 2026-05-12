import math
import random
from dataclasses import dataclass
import pygame
from pygame.math import Vector2
import settings

@dataclass
class InformationWave:
    pos: Vector2
    radius: float
    timestamp: int # The frame when it was emitted

class Guru:
    def __init__(self, x: float, y: float):
        self.pos = Vector2(x, y)
        self.waves = []
        self.cooldown = 0
    
    def update(self, frame_count: int):
        # Update wave radii
        for wave in self.waves:
            wave.radius += settings.WAVE_SPEED
            
        # Remove waves that are off-screen (approximate)
        max_dist = math.hypot(settings.WORLD_WIDTH, settings.WINDOW_HEIGHT)
        self.waves = [w for w in self.waves if w.radius < max_dist]
        
        # Emit new waves
        if self.cooldown <= 0:
            self.waves.append(InformationWave(Vector2(self.pos), 0.0, frame_count))
            self.cooldown = settings.WAVE_COOLDOWN
        else:
            self.cooldown -= 1
            
    def draw(self, screen, wave_surface):
        pygame.draw.circle(screen, settings.COLOR_GURU, (int(self.pos.x), int(self.pos.y)), 10)
        
        for wave in self.waves:
            pygame.draw.circle(wave_surface, settings.COLOR_WAVE, (int(wave.pos.x), int(wave.pos.y)), int(wave.radius), 2)


class InvestmentZone:
    def __init__(self, x: float, y: float):
        self.pos = Vector2(x, y)
        self.value = 0.0
        self.age = 0
        self.crashed = False
        self.radius = settings.ZONE_RADIUS
        
    def update(self, num_investing_agents: int):
        self.age += 1
        
        if self.crashed:
            # Reset after a short delay (1 second = 60 frames)
            if self.age > 60: 
                self.reset()
            return False # Return whether a crash just happened this frame
            
        # Grow value
        if num_investing_agents > 0:
            self.value += num_investing_agents * settings.GROWTH_FACTOR
            
        # Check crash condition
        crash_prob = settings.CRASH_PROBABILITY_BASE * (1.0 + self.age / 100.0)
        
        if self.value >= settings.MAX_CAPACITY or random.random() < crash_prob:
            self.crashed = True
            self.age = 0 # Use age to time the crashed state
            return True # Crash just happened
            
        return False
            
    def reset(self):
        self.value = 0.0
        self.age = 0
        self.crashed = False
        # Move to a new random location to prevent camping
        self.pos = Vector2(random.randint(100, settings.WORLD_WIDTH - 100), 
                           random.randint(100, settings.WINDOW_HEIGHT - 100))
                           
    def get_color(self):
        if self.crashed:
            return (255, 0, 0) # Czerwony po krachu
        # Gradient od zielonego do żółtego/czerwonego
        risk = min(self.value / settings.MAX_CAPACITY, 1.0)
        r = int(255 * risk)
        g = int(255 * (1 - risk))
        return (r, g, 0)
        
    def draw(self, screen):
        color = self.get_color()
        # Draw area
        surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*color, 50), (self.radius, self.radius), self.radius)
        screen.blit(surface, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))
        
        # Draw border
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), self.radius, 2)
        
        # Draw value
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"{int(self.value)}", True, settings.COLOR_TEXT)
        text_rect = text.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        screen.blit(text, text_rect)
