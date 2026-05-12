import pygame
import random
import neat
import settings
from entities import Guru, InvestmentZone
from agent import Agent

class Ecosystem:
    def __init__(self, genomes, config):
        self.frame_count = 0
        self.guru = Guru(settings.WINDOW_WIDTH / 2, settings.WINDOW_HEIGHT / 2)
        
        # Inicjalizacja stref (np. 3 strefy na początku)
        self.zones = []
        for _ in range(3):
            self.zones.append(InvestmentZone(
                random.randint(100, settings.WINDOW_WIDTH - 100),
                random.randint(100, settings.WINDOW_HEIGHT - 100)
            ))
            
        # Inicjalizacja agentów z genomów NEAT
        self.agents = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Rozmieszczenie losowe na krawędziach lub w całym oknie
            x = random.randint(50, settings.WINDOW_WIDTH - 50)
            y = random.randint(50, settings.WINDOW_HEIGHT - 50)
            agent = Agent(genome_id, net, x, y)
            genome.fitness = 0.0 # reset fitness
            self.agents.append((genome, agent))
            
    def run_frame(self, screen=None):
        self.frame_count += 1
        
        # Aktualizacja Guru
        self.guru.update(self.frame_count)
        
        # Obliczanie agentów w strefach
        investing_counts = {zone: 0 for zone in self.zones}
        for _, agent in self.agents:
            if agent.is_bankrupt or not agent.investing:
                continue
            for zone in self.zones:
                if agent.pos.distance_to(zone.pos) <= zone.radius:
                    investing_counts[zone] += 1
                    
        # Aktualizacja stref
        crashed_zones = []
        for zone in self.zones:
            crashed = zone.update(investing_counts[zone])
            if crashed:
                crashed_zones.append(zone)
                
        # Aktualizacja agentów
        all_agent_instances = [agent for _, agent in self.agents]
        
        for genome, agent in self.agents:
            if agent.is_bankrupt:
                continue
                
            # Sensory i sieci neuronowe
            agent.update_sensors(self.frame_count, self.guru, self.zones, all_agent_instances)
            
            # Fizyka (ruch, granice)
            agent.update_physics(self.zones)
            
            # Logika rynkowa (zyski/straty/krachy)
            agent.handle_market(self.zones, crashed_zones)
            
            # Aktualizacja fitnessu w genomie
            genome.fitness = agent.calculate_fitness()
            
        # Rysowanie (jeśli renderowanie włączone)
        if screen:
            screen.fill(settings.COLOR_BG)
            
            # Oddzielna powierzchnia dla przezroczystych fal
            wave_surface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
            
            self.guru.draw(screen, wave_surface)
            screen.blit(wave_surface, (0, 0))
            
            for zone in self.zones:
                zone.draw(screen)
                
            for _, agent in self.agents:
                agent.draw(screen)
                
            # Statystyki na ekranie
            font = pygame.font.SysFont(None, 24)
            active_agents = sum(1 for _, a in self.agents if not a.is_bankrupt)
            info = font.render(f"Aktywni agenci: {active_agents} / {len(self.agents)} | Klatka: {self.frame_count}/{settings.TIME_LIMIT_FRAMES}", True, settings.COLOR_TEXT)
            screen.blit(info, (10, 10))
            
            pygame.display.flip()

    def all_dead(self):
        return all(agent.is_bankrupt for _, agent in self.agents)
