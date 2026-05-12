import pygame
import random
import neat
import settings
from entities import Guru, InvestmentZone
from agent import Agent

class Ecosystem:
    def __init__(self, genomes, config, generation=1, best_fitness=0.0):
        self.frame_count = 0
        self.generation = generation
        self.best_fitness = best_fitness
        self.guru = Guru(settings.WORLD_WIDTH / 2, settings.WINDOW_HEIGHT / 2)
        
        # Inicjalizacja stref (np. 3 strefy na początku)
        self.zones = []
        for _ in range(3):
            self.zones.append(InvestmentZone(
                random.randint(100, settings.WORLD_WIDTH - 100),
                random.randint(100, settings.WINDOW_HEIGHT - 100)
            ))
            
        # Inicjalizacja agentów z genomów NEAT
        self.agents = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Rozmieszczenie losowe wewnątrz WORLD_WIDTH
            x = random.randint(50, settings.WORLD_WIDTH - 50)
            y = random.randint(50, settings.WINDOW_HEIGHT - 50)
            agent = Agent(genome_id, net, x, y)
            genome.fitness = 0.0 # reset fitness
            self.agents.append((genome, agent))
            
    def run_frame(self, screen=None, draw_vision=True):
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
        
        current_best = -9999
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
            if genome.fitness > current_best:
                current_best = genome.fitness
                
        # Rysowanie (jeśli renderowanie włączone)
        if screen:
            # Wypełniamy tło tylko dla obszaru świata (żeby nie nadpisać panelu)
            pygame.draw.rect(screen, settings.COLOR_BG, (0, 0, settings.WORLD_WIDTH, settings.WINDOW_HEIGHT))
            
            # Oddzielna powierzchnia dla przezroczystych fal
            wave_surface = pygame.Surface((settings.WORLD_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
            
            self.guru.draw(screen, wave_surface)
            screen.blit(wave_surface, (0, 0))
            
            for zone in self.zones:
                zone.draw(screen)
                
            for _, agent in self.agents:
                agent.draw(screen)
                if draw_vision and not agent.is_bankrupt:
                    # Rysowanie wizji opcjonalnie
                    pass
                
            # Rysowanie panelu UI
            self.draw_ui_panel(screen, current_best)
            
            pygame.display.flip()

    def draw_ui_panel(self, screen, current_best):
        panel_rect = pygame.Rect(settings.WORLD_WIDTH, 0, settings.PANEL_WIDTH, settings.WINDOW_HEIGHT)
        pygame.draw.rect(screen, settings.COLOR_PANEL, panel_rect)
        
        # Linie oddzielające wewnątrz panelu
        pygame.draw.line(screen, settings.COLOR_BG, (settings.WORLD_WIDTH, 0), (settings.WORLD_WIDTH, settings.WINDOW_HEIGHT), 2)
        
        font_title = pygame.font.SysFont(None, 32)
        font_text = pygame.font.SysFont(None, 24)
        
        x_offset = settings.WORLD_WIDTH + 20
        y_offset = 20
        line_height = 30
        
        # Statystyki ogólne
        title = font_title.render("STATYSTYKI", True, settings.COLOR_PANEL_TEXT)
        screen.blit(title, (x_offset, y_offset))
        y_offset += 40
        
        active_agents = sum(1 for _, a in self.agents if not a.is_bankrupt)
        total_agents = len(self.agents)
        
        stats = [
            f"Generacja: {self.generation}",
            f"Klatka: {self.frame_count} / {settings.TIME_LIMIT_FRAMES}",
            f"Aktywni agenci: {active_agents} / {total_agents}",
            f"Akt. Najlepszy Fitness: {current_best:.1f}",
            f"Poprz. Najlepszy Fitness: {self.best_fitness:.1f}"
        ]
        
        for text in stats:
            rendered = font_text.render(text, True, settings.COLOR_PANEL_TEXT)
            screen.blit(rendered, (x_offset, y_offset))
            y_offset += line_height
            
        y_offset += 20
        title2 = font_title.render("STREFY INWESTYCYJNE", True, settings.COLOR_PANEL_TEXT)
        screen.blit(title2, (x_offset, y_offset))
        y_offset += 40
        
        for i, zone in enumerate(self.zones):
            risk = min(zone.value / settings.MAX_CAPACITY, 1.0)
            status = "KRACH!" if zone.crashed else f"Wartość: {int(zone.value)}"
            zone_text = f"Strefa {i+1}: {status} (Ryzyko: {risk*100:.0f}%)"
            color = (255, 100, 100) if zone.crashed else settings.COLOR_PANEL_TEXT
            rendered = font_text.render(zone_text, True, color)
            screen.blit(rendered, (x_offset, y_offset))
            y_offset += line_height
            
        y_offset += 40
        title3 = font_title.render("STEROWANIE", True, settings.COLOR_PANEL_TEXT)
        screen.blit(title3, (x_offset, y_offset))
        y_offset += 40
        
        controls = [
            "SPACJA - Pauza",
            "F - Fast Forward (Ukryj obraz)",
            "S - Zabij wszystkich (Skip gen.)",
            "V - Pokaż wektory"
        ]
        
        for text in controls:
            rendered = font_text.render(text, True, settings.COLOR_PANEL_TEXT)
            screen.blit(rendered, (x_offset, y_offset))
            y_offset += line_height

    def all_dead(self):
        return all(agent.is_bankrupt for _, agent in self.agents)
