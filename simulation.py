import pygame
import random
import neat
import settings
from entities import Guru, InvestmentZone, get_font
from agent import Agent

class Ecosystem:
    def __init__(self, genomes, config, generation=1, best_fitness=0.0):
        self.frame_count = 0
        self.generation = generation
        self.best_fitness = best_fitness
        self.config = config
        self.guru = Guru(settings.WORLD_WIDTH / 2, settings.WINDOW_HEIGHT / 2)
        
        # --- Obligacja państwowa (permanentna, 1 szt.) ---
        self.bond_zone = InvestmentZone(
            settings.WORLD_WIDTH // 2,
            settings.WINDOW_HEIGHT - 80,
            zone_type=settings.ZONE_TYPE_BOND
        )
        self.bond_zone.radius = 80
        
        # --- Akcje (3 szt.) ---
        self.stock_zones = []
        for i in range(3):
            self.stock_zones.append(InvestmentZone(
                200 + i * 300,
                300,
                zone_type=settings.ZONE_TYPE_STOCK
            ))
            
        # --- Krypto (1 szt.) ---
        self.crypto_zone = InvestmentZone(
            settings.WORLD_WIDTH // 2,
            100,
            zone_type=settings.ZONE_TYPE_CRYPTO
        )
        
        # Wszystkie strefy razem (do przekazywania agentom)
        self.all_zones = [self.bond_zone] + self.stock_zones + [self.crypto_zone]
            
        # Inicjalizacja agentów z genomów NEAT
        self.agents = []
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            x = random.randint(50, settings.WORLD_WIDTH - 50)
            y = random.randint(50, settings.WINDOW_HEIGHT - 50)
            agent = Agent(genome_id, net, x, y)
            genome.fitness = 0.0
            self.agents.append((genome, agent))
            
    def run_frame(self, screen=None, draw_vision=True):
        self.frame_count += 1
        
        # Aktualizacja Guru
        self.guru.update(self.frame_count)
        
        # Obliczanie inwestorów w strefach (agenci w zasięgu z portfolio)
        investing_counts = {zone.zone_id: 0 for zone in self.all_zones}
        for _, agent in self.agents:
            if agent.is_bankrupt:
                continue
            for zone in self.all_zones:
                if agent.pos.distance_to(zone.pos) <= zone.radius:
                    if zone.zone_id in agent.portfolio:
                        investing_counts[zone.zone_id] += 1
                    
        # Aktualizacja stref (Poisson + tłum + cykl rynkowy)
        crashed_zones = []
        for zone in self.all_zones:
            did_crash = zone.update(investing_counts.get(zone.zone_id, 0), self.frame_count)
            if did_crash:
                crashed_zones.append(zone)
                
        # Powiadom agentów o korektach rynkowych
        for zone in crashed_zones:
            for _, agent in self.agents:
                agent.handle_crash(zone)
                
        # Aktualizacja agentów
        all_agent_instances = [agent for _, agent in self.agents]
        
        current_best = -9999
        for genome, agent in self.agents:
            if agent.is_bankrupt:
                continue
                
            agent.update_sensors(self.frame_count, self.guru, self.all_zones, all_agent_instances)
            agent.update_physics()
            agent.handle_market_metrics(self.all_zones)
            
            genome.fitness = agent.calculate_fitness(self.all_zones)
            if genome.fitness > current_best:
                current_best = genome.fitness
                
        # Rysowanie
        if screen:
            pygame.draw.rect(screen, settings.COLOR_BG, (0, 0, settings.WORLD_WIDTH, settings.WINDOW_HEIGHT))
            
            wave_surface = pygame.Surface((settings.WORLD_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
            self.guru.draw(screen, wave_surface)
            screen.blit(wave_surface, (0, 0))
            
            for zone in self.all_zones:
                zone.draw(screen)
                
            for _, agent in self.agents:
                agent.draw(screen)
                
            self.draw_ui_panel(screen, current_best)
            pygame.display.flip()

    def draw_ui_panel(self, screen, current_best):
        panel_rect = pygame.Rect(settings.WORLD_WIDTH, 0, settings.PANEL_WIDTH, settings.WINDOW_HEIGHT)
        pygame.draw.rect(screen, settings.COLOR_PANEL, panel_rect)
        pygame.draw.line(screen, settings.COLOR_BG, (settings.WORLD_WIDTH, 0), (settings.WORLD_WIDTH, settings.WINDOW_HEIGHT), 2)
        
        font_title = get_font(28)
        font_text = get_font(20)
        
        x_offset = settings.WORLD_WIDTH + 15
        y_offset = 15
        line_height = 24
        
        # === STATYSTYKI ===
        title = font_title.render("STATYSTYKI", True, settings.COLOR_PANEL_TEXT)
        screen.blit(title, (x_offset, y_offset))
        y_offset += 32
        
        active_agents = sum(1 for _, a in self.agents if not a.is_bankrupt)
        total_agents = len(self.agents)
        agents_with_portfolio = sum(1 for _, a in self.agents if a.portfolio and not a.is_bankrupt)
        
        avg_worth = 0.0
        if active_agents > 0:
            avg_worth = sum(a.get_total_worth(self.all_zones) for _, a in self.agents if not a.is_bankrupt) / active_agents
        
        # Faza rynku
        is_bear = self.frame_count > settings.BEAR_MARKET_START
        market_label = "BEAR 🐻" if is_bear else "BULL 🐂"
        market_color = (255, 100, 100) if is_bear else (100, 255, 100)
        
        stats = [
            f"Generacja: {self.generation}",
            f"Klatka: {self.frame_count} / {settings.TIME_LIMIT_FRAMES}",
            f"Aktywni: {active_agents}/{total_agents}",
            f"Z portfelem: {agents_with_portfolio}",
            f"Śr. majątek: ${avg_worth:.1f}",
            f"Best Fitness: {current_best:.1f}",
            f"Poprz. Best: {self.best_fitness:.1f}",
            f"Inflacja: {settings.INFLATION_RATE*100:.2f}%/kl.",
            f"Prowizja: {settings.TRANSACTION_FEE_RATE*100:.0f}%"
        ]
        
        for text in stats:
            rendered = font_text.render(text, True, settings.COLOR_PANEL_TEXT)
            screen.blit(rendered, (x_offset, y_offset))
            y_offset += line_height
            
        # Faza rynku — kolorowy wskaźnik
        rendered = font_text.render(f"Faza: {market_label}", True, market_color)
        screen.blit(rendered, (x_offset, y_offset))
        y_offset += line_height
            
        # === RYNEK ===
        y_offset += 12
        title2 = font_title.render("RYNEK", True, settings.COLOR_PANEL_TEXT)
        screen.blit(title2, (x_offset, y_offset))
        y_offset += 32
        
        type_labels = {
            settings.ZONE_TYPE_BOND: "BOND",
            settings.ZONE_TYPE_STOCK: "STOCK",
            settings.ZONE_TYPE_CRYPTO: "CRYPTO"
        }
        
        for zone in self.all_zones:
            label = type_labels.get(zone.zone_type, "?")
            trend = zone.get_price_trend()
            trend_arrow = "↑" if trend > 0.1 else ("↓" if trend < -0.1 else "→")
            investors = investing_counts = sum(1 for _, a in self.agents 
                                               if not a.is_bankrupt 
                                               and zone.zone_id in a.portfolio 
                                               and a.pos.distance_to(zone.pos) <= zone.radius)
            
            if zone.crashed:
                zone_text = f"{label}: ${zone.share_price:.1f} KRACH!"
                color = (255, 100, 100)
            else:
                zone_text = f"{label}: ${zone.share_price:.1f} {trend_arrow} [{investors}]"
                color = settings.COLOR_PANEL_TEXT
                
            rendered = font_text.render(zone_text, True, color)
            screen.blit(rendered, (x_offset, y_offset))
            y_offset += line_height
            
        # === STEROWANIE ===
        y_offset += 20
        title3 = font_title.render("STEROWANIE", True, settings.COLOR_PANEL_TEXT)
        screen.blit(title3, (x_offset, y_offset))
        y_offset += 32
        
        controls = [
            "SPACJA - Pauza",
            "F - Fast Forward",
            "S - Skip generacji",
            "V - Wektory"
        ]
        
        for text in controls:
            rendered = font_text.render(text, True, settings.COLOR_PANEL_TEXT)
            screen.blit(rendered, (x_offset, y_offset))
            y_offset += line_height

    def all_dead(self):
        return all(agent.is_bankrupt for _, agent in self.agents)
