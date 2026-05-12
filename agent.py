import math
import random
import pygame
from pygame.math import Vector2
import settings
from entities import Guru, InvestmentZone

class Agent:
    def __init__(self, genome_id, net, x, y):
        self.genome_id = genome_id
        self.net = net
        self.pos = Vector2(x, y)
        self.vel = Vector2(0, 0)
        self.angle = random.uniform(0, 360)
        
        self.capital = settings.STARTING_CAPITAL
        self.is_bankrupt = False
        self.survival_time = 0
        self.fitness = 0.0
        
        # Stany
        self.investing = False
        self.signal_freshness = 0.0 # 0.0 do 1.0
        self.last_wave_timestamp = -1
        
    def update_sensors(self, frame_count: int, guru: Guru, zones: list[InvestmentZone], other_agents: list['Agent']):
        if self.is_bankrupt:
            return
            
        # 1. Świeżość sygnału (z Guru)
        # Check if hit by any wave
        hit_by_wave = False
        for wave in guru.waves:
            # Check distance to guru's center
            dist_to_guru = self.pos.distance_to(guru.pos)
            # If distance is close to wave radius (tolerance of wave speed)
            if abs(dist_to_guru - wave.radius) <= settings.WAVE_SPEED * 2:
                self.last_wave_timestamp = wave.timestamp
                self.signal_freshness = 1.0
                hit_by_wave = True
                
        # Wygasanie sygnału
        if not hit_by_wave and self.signal_freshness > 0:
            self.signal_freshness -= 0.005 # Powoli spada
            if self.signal_freshness < 0:
                self.signal_freshness = 0.0

        # 2-4. Najbliższa strefa (dystans i wektor)
        closest_zone = None
        min_dist = float('inf')
        for zone in zones:
            dist = self.pos.distance_to(zone.pos)
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone
                
        if closest_zone:
            # Znormalizowany dystans (0 to środek ekranu w przybliżeniu)
            max_screen_dist = math.hypot(settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT)
            norm_dist = min_dist / max_screen_dist
            
            direction = closest_zone.pos - self.pos
            if direction.length() > 0:
                direction = direction.normalize()
            zone_vec_x = direction.x
            zone_vec_y = direction.y
        else:
            norm_dist = 1.0
            zone_vec_x = 0.0
            zone_vec_y = 0.0
            
        # 5. Lokalna gęstość tłumu
        crowd_count = 0
        for agent in other_agents:
            if agent is not self and not agent.is_bankrupt:
                if self.pos.distance_to(agent.pos) <= settings.SENSOR_RADIUS:
                    crowd_count += 1
        # Normalizacja gęstości (zakładamy że 20 agentów w okolicy to już tłum)
        local_density = min(crowd_count / 20.0, 1.0)
        
        # 6. Aktualny kapitał (znormalizowany)
        norm_capital = min(self.capital / (settings.STARTING_CAPITAL * 10), 1.0) # zakładamy max 10x początkowy
        
        inputs = (
            self.signal_freshness,
            norm_dist,
            zone_vec_x,
            zone_vec_y,
            local_density,
            norm_capital
        )
        
        # Pobieranie akcji z sieci
        outputs = self.net.activate(inputs)
        self.apply_action(outputs)
        
    def apply_action(self, outputs: list[float]):
        if self.is_bankrupt:
            return
            
        # Wyjścia: [przyspieszenie, moment_obrotowy, decyzja_binarna_wejście_wyjście]
        acceleration = outputs[0]
        torque = outputs[1]
        decision = outputs[2]
        
        # Obrót
        self.angle += torque * settings.TURN_RATE * 360 # w stopniach
        self.angle %= 360
        
        # Przyspieszenie
        # Map acceleration from [-1, 1] to moving forward/backward
        acc_vec = Vector2(acceleration * settings.ACCELERATION_FACTOR, 0).rotate(self.angle)
        self.vel += acc_vec
        
        # Limit speed
        if self.vel.length() > settings.MAX_SPEED:
            self.vel.scale_to_length(settings.MAX_SPEED)
            
        # Decyzja o inwestycji (> 0.0 = wejdź, <= 0.0 = wyjdź)
        self.investing = (decision > 0.0)
        
    def update_physics(self, zones: list[InvestmentZone]):
        if self.is_bankrupt:
            return
            
        # Ruch
        self.pos += self.vel
        
        # Tarcie
        self.vel *= 0.95
        
        # Odbicia od ścian
        if self.pos.x <= 0:
            self.pos.x = 0
            self.vel.x *= -1
        elif self.pos.x >= settings.WINDOW_WIDTH:
            self.pos.x = settings.WINDOW_WIDTH
            self.vel.x *= -1
            
        if self.pos.y <= 0:
            self.pos.y = 0
            self.vel.y *= -1
        elif self.pos.y >= settings.WINDOW_HEIGHT:
            self.pos.y = settings.WINDOW_HEIGHT
            self.vel.y *= -1
            
        # Koszt życia
        self.capital -= settings.ENERGY_DRAIN_PER_FRAME
        self.survival_time += 1
        
        if self.capital <= 0:
            self.is_bankrupt = True
            
    def handle_market(self, zones: list[InvestmentZone], crashed_zones: list[InvestmentZone]):
        if self.is_bankrupt:
            return
            
        # Sprawdzanie czy agent znajduje się w strefie i inwestuje
        in_any_zone = False
        for zone in zones:
            if self.pos.distance_to(zone.pos) <= zone.radius:
                in_any_zone = True
                if self.investing:
                    # Agent is actively investing inside the zone
                    # (To strefa decyduje o wzroście wartości na podstawie liczby agentów)
                    pass
                    
        # Sprawdzanie czy strefa w której jest agent właśnie zbankrutowała
        for zone in crashed_zones:
            if self.pos.distance_to(zone.pos) <= zone.radius:
                if self.investing:
                    # Krach! Agent traci kapitał
                    self.capital = 0
                    self.is_bankrupt = True
                    break
        
    def calculate_fitness(self):
        if self.is_bankrupt:
            self.fitness = self.capital + (self.survival_time * settings.BONUS_SURVIVAL_PER_FRAME) - settings.PENALTY_BANKRUPTCY
        else:
            self.fitness = self.capital + (self.survival_time * settings.BONUS_SURVIVAL_PER_FRAME)
        return self.fitness

    def draw(self, screen):
        color = settings.COLOR_AGENT_BANKRUPT if self.is_bankrupt else settings.COLOR_AGENT
        if self.investing and not self.is_bankrupt:
            # Highlight investing agents (e.g. outline or different color tone)
            color = (50, 255, 100)
            
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), settings.AGENT_RADIUS)
        
        # Kierunek
        if not self.is_bankrupt:
            dir_vec = Vector2(settings.AGENT_RADIUS * 1.5, 0).rotate(self.angle)
            end_pos = self.pos + dir_vec
            pygame.draw.line(screen, (255, 255, 255), (int(self.pos.x), int(self.pos.y)), (int(end_pos.x), int(end_pos.y)), 1)
