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
        
        # Finanse
        self.cash = settings.STARTING_CAPITAL
        self.portfolio = {}  # {zone_id: liczba_akcji}
        self.is_bankrupt = False
        self.survival_time = 0
        self.fitness = 0.0
        
        # Sensory
        self.signal_freshness = 0.0
        self.last_wave_timestamp = -1
        
        # Statystyki strategii (do badań)
        self.metrics = {
            'trades': 0,
            'time_in_bond': 0,
            'time_in_stock': 0,
            'time_in_crypto': 0,
            'buys': 0,
            'sells': 0,
            'max_wealth': settings.STARTING_CAPITAL,
            'fees_paid': 0.0,
            'crash_losses': 0,
            'spent_on_bond': 0.0,
            'spent_on_stock': 0.0,
            'spent_on_crypto': 0.0
        }
        
    def get_total_worth(self, zones: list[InvestmentZone]) -> float:
        """Oblicza całkowity majątek: gotówka + wartość rynkowa portfela."""
        portfolio_value = 0.0
        zone_map = {z.zone_id: z for z in zones}
        for zone_id, shares in self.portfolio.items():
            if zone_id in zone_map:
                portfolio_value += shares * zone_map[zone_id].share_price
        return self.cash + portfolio_value
    
    def get_portfolio_value_in_zone(self, zone: InvestmentZone) -> float:
        """Zwraca wartość akcji agenta w konkretnej strefie."""
        shares = self.portfolio.get(zone.zone_id, 0)
        return shares * zone.share_price
    
    def _get_held_asset_classes(self, zones: list[InvestmentZone]) -> set:
        """Zwraca zbiór klas aktywów w portfelu (do dywersyfikacji)."""
        zone_map = {z.zone_id: z for z in zones}
        classes = set()
        for zone_id, shares in self.portfolio.items():
            if shares > 0 and zone_id in zone_map:
                classes.add(zone_map[zone_id].zone_type)
        return classes
        
    def update_sensors(self, frame_count: int, guru: Guru, zones: list[InvestmentZone], other_agents: list['Agent']):
        if self.is_bankrupt:
            return
            
        # 1. Świeżość sygnału (z Guru)
        hit_by_wave = False
        for wave in guru.waves:
            dist_to_guru = self.pos.distance_to(guru.pos)
            if abs(dist_to_guru - wave.radius) <= settings.WAVE_SPEED * 2:
                self.last_wave_timestamp = wave.timestamp
                self.signal_freshness = 1.0
                hit_by_wave = True
                
        if not hit_by_wave and self.signal_freshness > 0:
            self.signal_freshness -= 0.005
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
            max_screen_dist = math.hypot(settings.WORLD_WIDTH, settings.WINDOW_HEIGHT)
            norm_dist = min_dist / max_screen_dist
            
            direction = closest_zone.pos - self.pos
            if direction.length() > 0:
                direction = direction.normalize()
                
            # Rzutowanie wektora na lokalny układ współrzędnych agenta
            forward_vec = Vector2(1, 0).rotate(self.angle)
            right_vec = Vector2(0, 1).rotate(self.angle)
            
            zone_vec_x = direction.dot(forward_vec)  # 1.0 = strefa idealnie z przodu
            zone_vec_y = direction.dot(right_vec)    # 1.0 = strefa idealnie z prawej
            zone_type = closest_zone.zone_type
            
            # 8. Wartość portfela w najbliższej strefie (znormalizowana)
            portfolio_val = self.get_portfolio_value_in_zone(closest_zone)
            norm_portfolio = min(portfolio_val / (settings.STARTING_CAPITAL * 5), 1.0)
            
            # 9. Trend cenowy strefy
            price_trend = closest_zone.get_price_trend()
        else:
            norm_dist = 1.0
            zone_vec_x = 0.0
            zone_vec_y = 0.0
            zone_type = settings.ZONE_TYPE_STOCK
            norm_portfolio = 0.0
            price_trend = 0.0
            
        # 5. Lokalna gęstość tłumu
        crowd_count = 0
        for agent in other_agents:
            if agent is not self and not agent.is_bankrupt:
                if self.pos.distance_to(agent.pos) <= settings.SENSOR_RADIUS:
                    crowd_count += 1
        local_density = min(crowd_count / 20.0, 1.0)
        
        # 6. Aktualny kapitał gotówkowy (znormalizowany)
        norm_cash = min(self.cash / (settings.STARTING_CAPITAL * 10), 1.0)
        
        # 10. [NOWE] Faza rynku: 0.0 = bull market, 1.0 = bear market
        market_phase = 1.0 if frame_count > settings.BEAR_MARKET_START else 0.0
        
        # 10 wejść do sieci neuronowej
        inputs = (
            self.signal_freshness,    # 1
            norm_dist,                # 2
            zone_vec_x,               # 3
            zone_vec_y,               # 4
            local_density,            # 5
            norm_cash,                # 6
            zone_type,                # 7
            norm_portfolio,           # 8
            price_trend,              # 9
            market_phase              # 10 [NOWE]
        )
        
        outputs = self.net.activate(inputs)
        self.apply_action(outputs, zones)
        
    def apply_action(self, outputs: list[float], zones: list[InvestmentZone]):
        if self.is_bankrupt:
            return
            
        # Wyjścia: [przyspieszenie, moment_obrotowy, decyzja_kup_sprzedaj, intensywność]
        acceleration = outputs[0]
        torque = outputs[1]
        decision = outputs[2]      # < -0.3 = sprzedaj, > 0.3 = kup, else trzymaj
        intensity = (outputs[3] + 1.0) / 2.0  # Mapowanie [-1,1] → [0,1]
        intensity = max(0.0, min(1.0, intensity))
        
        # Obrót
        self.angle += torque * settings.TURN_RATE * 360
        self.angle %= 360
        
        # Przyspieszenie
        acc_vec = Vector2(acceleration * settings.ACCELERATION_FACTOR, 0).rotate(self.angle)
        self.vel += acc_vec
        
        if self.vel.length() > settings.MAX_SPEED:
            self.vel.scale_to_length(settings.MAX_SPEED)
            
        # Logika kupna/sprzedaży — tylko gdy agent jest w zasięgu strefy
        for zone in zones:
            if zone.crashed:
                continue
            if self.pos.distance_to(zone.pos) <= zone.radius:
                if decision > 0.3:
                    self._buy_shares(zone, intensity)
                elif decision < -0.3:
                    self._sell_shares(zone, intensity)
                break  # Obsługuj tylko jedną strefę na raz
                
    def _buy_shares(self, zone: InvestmentZone, intensity: float):
        """Kupuje akcje za gotówkę (z prowizją transakcyjną)."""
        if zone.share_price <= 0 or self.cash <= 0:
            return
            
        # Ile gotówki przeznaczyć na zakup
        budget = self.cash * intensity
        
        # Potrącenie prowizji z budżetu
        fee = budget * settings.TRANSACTION_FEE_RATE
        effective_budget = budget - fee
        
        if effective_budget < zone.share_price:
            return  # Nie stać na ani jedną akcję po prowizji
            
        num_shares = int(effective_budget / zone.share_price)
        if num_shares <= 0:
            return
            
        cost = num_shares * zone.share_price
        total_cost = cost + fee  # Koszt akcji + prowizja
        self.cash -= total_cost
        self.portfolio[zone.zone_id] = self.portfolio.get(zone.zone_id, 0) + num_shares
        
        self.metrics['trades'] += 1
        self.metrics['buys'] += 1
        self.metrics['fees_paid'] += fee
        
        if zone.zone_type == settings.ZONE_TYPE_BOND:
            self.metrics['spent_on_bond'] += cost
        elif zone.zone_type == settings.ZONE_TYPE_STOCK:
            self.metrics['spent_on_stock'] += cost
        elif zone.zone_type == settings.ZONE_TYPE_CRYPTO:
            self.metrics['spent_on_crypto'] += cost
        
    def _sell_shares(self, zone: InvestmentZone, intensity: float):
        """Sprzedaje akcje za gotówkę (z prowizją transakcyjną)."""
        held = self.portfolio.get(zone.zone_id, 0)
        if held <= 0:
            return
            
        num_to_sell = max(1, int(held * intensity))
        num_to_sell = min(num_to_sell, held)
        
        gross_revenue = num_to_sell * zone.share_price
        fee = gross_revenue * settings.TRANSACTION_FEE_RATE
        net_revenue = gross_revenue - fee  # Przychód po prowizji
        
        self.cash += net_revenue
        self.portfolio[zone.zone_id] -= num_to_sell
        
        if self.portfolio[zone.zone_id] <= 0:
            del self.portfolio[zone.zone_id]
            
        self.metrics['trades'] += 1
        self.metrics['sells'] += 1
        self.metrics['fees_paid'] += fee
        
    def update_physics(self):
        if self.is_bankrupt:
            return
            
        # Ruch
        self.pos += self.vel
        self.vel *= 0.95  # Tarcie
        
        # Odbicia od ścian
        if self.pos.x <= 0:
            self.pos.x = 0
            self.vel.x *= -1
        elif self.pos.x >= settings.WORLD_WIDTH:
            self.pos.x = settings.WORLD_WIDTH
            self.vel.x *= -1
            
        if self.pos.y <= 0:
            self.pos.y = 0
            self.vel.y *= -1
        elif self.pos.y >= settings.WINDOW_HEIGHT:
            self.pos.y = settings.WINDOW_HEIGHT
            self.vel.y *= -1
            
        # Inflacja — gotówka traci wartość
        self.cash *= (1.0 - settings.INFLATION_RATE)
        self.survival_time += 1
        
        if self.cash <= 0.01 and not self.portfolio:
            self.is_bankrupt = True
            
    def handle_market_metrics(self, zones: list[InvestmentZone]):
        """Aktualizuje statystyki czasu przebywania w konkretnych typach aktywów."""
        if self.is_bankrupt:
            return
        
        current_worth = self.get_total_worth(zones)
        if current_worth > self.metrics['max_wealth']:
            self.metrics['max_wealth'] = current_worth

        for zone_id, shares in self.portfolio.items():
            if shares > 0:
                for z in zones:
                    if z.zone_id == zone_id:
                        if z.zone_type == settings.ZONE_TYPE_BOND:
                            self.metrics['time_in_bond'] += 1
                        elif z.zone_type == settings.ZONE_TYPE_STOCK:
                            self.metrics['time_in_stock'] += 1
                        elif z.zone_type == settings.ZONE_TYPE_CRYPTO:
                            self.metrics['time_in_crypto'] += 1
                        break
            
    def handle_crash(self, crashed_zone: InvestmentZone):
        """Obsługuje krach strefy — akcje tracą wartość (nie znikają, cena jest już obniżona)."""
        # Przy korekcie rynkowej akcje nie znikają, ale ich wartość
        # jest już uwzględniona w obniżonej cenie share_price
        if crashed_zone.zone_id in self.portfolio:
            self.metrics['crash_losses'] += 1
        
    def calculate_fitness(self, zones: list[InvestmentZone]):
        total_worth = self.get_total_worth(zones)
        
        # Bazowy fitness: majątek + premia za przetrwanie
        if self.is_bankrupt:
            self.fitness = total_worth + (self.survival_time * settings.BONUS_SURVIVAL_PER_FRAME) - settings.PENALTY_BANKRUPTCY
        else:
            self.fitness = total_worth + (self.survival_time * settings.BONUS_SURVIVAL_PER_FRAME)
        
        # [NOWE] Bonus za dywersyfikację portfela
        # Agent dostaje premię za posiadanie aktywów z RÓŻNYCH klas
        asset_classes = self._get_held_asset_classes(zones)
        if len(asset_classes) > 1:
            self.fitness += (len(asset_classes) - 1) * settings.DIVERSIFICATION_BONUS
            
        return self.fitness

    def draw(self, screen):
        if self.is_bankrupt:
            color = settings.COLOR_AGENT_BANKRUPT
        elif self.portfolio:
            color = (50, 255, 100)  # Zielony = posiada akcje
        else:
            color = settings.COLOR_AGENT
            
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), settings.AGENT_RADIUS)
        
        if not self.is_bankrupt:
            dir_vec = Vector2(settings.AGENT_RADIUS * 1.5, 0).rotate(self.angle)
            end_pos = self.pos + dir_vec
            pygame.draw.line(screen, (255, 255, 255), 
                           (int(self.pos.x), int(self.pos.y)), 
                           (int(end_pos.x), int(end_pos.y)), 1)
