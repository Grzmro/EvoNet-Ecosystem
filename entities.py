import math
import random
from dataclasses import dataclass
import pygame
from pygame.math import Vector2
import settings

# Cache fontów — tworzymy raz, nie w każdej klatce
_font_cache = {}
_zone_id_counter = 0

def poisson(lam: float) -> int:
    """Generuje zmienną losową z rozkładu Poissona."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(0, k - 1)

def get_font(size: int):
    """Zwraca skeszowany font o danym rozmiarze."""
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont(None, size)
    return _font_cache[size]

def _next_zone_id():
    global _zone_id_counter
    _zone_id_counter += 1
    return _zone_id_counter

@dataclass
class InformationWave:
    pos: Vector2
    radius: float
    timestamp: int

class Guru:
    def __init__(self, x: float, y: float):
        self.pos = Vector2(x, y)
        self.waves = []
        self.cooldown = 0
    
    def update(self, frame_count: int):
        for wave in self.waves:
            wave.radius += settings.WAVE_SPEED
            
        max_dist = math.hypot(settings.WORLD_WIDTH, settings.WINDOW_HEIGHT)
        self.waves = [w for w in self.waves if w.radius < max_dist]
        
        if self.cooldown <= 0:
            self.waves.append(InformationWave(Vector2(self.pos), 0.0, frame_count))
            self.cooldown = settings.WAVE_COOLDOWN
        else:
            self.cooldown -= 1
            
    def draw(self, screen, wave_surface):
        pygame.draw.circle(screen, settings.COLOR_GURU, (int(self.pos.x), int(self.pos.y)), 10)
        for wave in self.waves:
            pygame.draw.circle(wave_surface, settings.COLOR_WAVE, 
                             (int(wave.pos.x), int(wave.pos.y)), int(wave.radius), 2)


class InvestmentZone:
    def __init__(self, x: float, y: float, zone_type: float = None):
        self.zone_id = _next_zone_id()
        self.pos = Vector2(x, y)
        self.radius = settings.ZONE_RADIUS
        self.zone_type = zone_type if zone_type is not None else self._pick_random_type()
        
        # System cenowy
        self.share_price = 10.0
        self.price_history = [10.0] * 30
        self.age = 0
        self.crashed = False  # True = właśnie nastąpiła korekta (trwa 1 klatkę, informacyjnie)
        self.crash_cooldown = 0  # Czas ochrony po krachu (zapobiega kaskadzie)
        
    def _pick_random_type(self):
        """Losuje typ strefy (tylko STOCK lub CRYPTO, obligacje tworzymy ręcznie)."""
        r = random.random()
        if r < 0.7:
            return settings.ZONE_TYPE_STOCK
        else:
            return settings.ZONE_TYPE_CRYPTO
        
    def update(self, num_investing_agents: int, frame_count: int):
        """Aktualizuje cenę akcji: Poisson + cykle rynkowe + wpływ tłumu + ryzyko krachu."""
        self.age += 1
        self.crashed = False  # Reset flagi krachu z poprzedniej klatki
        
        if self.crash_cooldown > 0:
            self.crash_cooldown -= 1
        
        # --- Określenie fazy rynku (bull / bear) ---
        is_bear = frame_count > settings.BEAR_MARKET_START
        
        # --- Bazowe lambdy Poissona w zależności od typu ---
        if self.zone_type == settings.ZONE_TYPE_BOND:
            lam_up = settings.BOND_LAMBDA_UP
            lam_down = settings.BOND_LAMBDA_DOWN
            jump_size = settings.BOND_JUMP_SIZE
        elif self.zone_type == settings.ZONE_TYPE_STOCK:
            lam_up = settings.STOCK_LAMBDA_UP
            lam_down = settings.STOCK_LAMBDA_DOWN
            jump_size = settings.STOCK_JUMP_SIZE
        else:  # CRYPTO
            lam_up = settings.CRYPTO_LAMBDA_UP
            lam_down = settings.CRYPTO_LAMBDA_DOWN
            jump_size = settings.CRYPTO_JUMP_SIZE
        
        # --- Modyfikacja lambdy przez cykl rynkowy ---
        if is_bear:
            lam_down *= settings.BEAR_LAMBDA_MULTIPLIER
        else:
            lam_up *= settings.BULL_LAMBDA_MULTIPLIER
            
        # --- Wahania Poissona ---
        jumps_up = poisson(lam_up)
        jumps_down = poisson(lam_down)
        price_change = (jumps_up - jumps_down) * jump_size
        
        # --- Wpływ tłumu (bańka spekulacyjna) ---
        # Obligacje są odporne na bańki
        if self.zone_type != settings.ZONE_TYPE_BOND and num_investing_agents > 0:
            crowd_boost = num_investing_agents * settings.CROWD_PRICE_BOOST
            price_change += crowd_boost
            
            # Ryzyko krachu rynkowego (korekta) przy przeludnieniu
            if num_investing_agents >= settings.CROWD_CRASH_THRESHOLD and self.crash_cooldown <= 0:
                excess = num_investing_agents - settings.CROWD_CRASH_THRESHOLD + 1
                crash_prob = settings.CROWD_CRASH_BASE_PROB * excess
                if random.random() < crash_prob:
                    # KRACH! Korekta rynkowa — cena spada o X%
                    self.share_price *= settings.CRASH_PRICE_DROP
                    self.crashed = True
                    self.crash_cooldown = 60  # 1 sekunda ochrony przed kolejnym krachem
                    self._update_price_history()
                    return True
        
        self.share_price += price_change
        
        # Cena nie może spaść poniżej 0.01
        self.share_price = max(self.share_price, 0.01)
        self._update_price_history()
        return False
        
    def _update_price_history(self):
        """Aktualizuje historię cen (do obliczania trendu)."""
        self.price_history.append(self.share_price)
        if len(self.price_history) > 30:
            self.price_history.pop(0)
            
    def get_price_trend(self) -> float:
        """Zwraca trend cenowy: +1.0 (silny wzrost), -1.0 (silny spadek), 0.0 (stabilna)."""
        if len(self.price_history) < 10:
            return 0.0
        recent = sum(self.price_history[-5:]) / 5.0
        older = sum(self.price_history[-15:-10]) / 5.0
        if older == 0:
            return 0.0
        change = (recent - older) / max(older, 0.01)
        return max(-1.0, min(1.0, change * 5.0))
                           
    def get_color(self):
        if self.crashed:
            return (255, 60, 60)  # Czerwony błysk przy krachu
            
        if self.zone_type == settings.ZONE_TYPE_BOND:
            return (80, 140, 255)
        elif self.zone_type == settings.ZONE_TYPE_CRYPTO:
            intensity = min(self.share_price / 100.0, 1.0)
            return (180, int(50 + 180 * intensity), 255)
        else:  # STOCK
            risk = min(self.share_price / 50.0, 1.0)
            r = int(255 * risk)
            g = int(255 * (1 - risk))
            return (r, g, 0)
        
    def draw(self, screen):
        color = self.get_color()
        
        # Przezroczyste wypełnienie
        surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*color, 50), (self.radius, self.radius), self.radius)
        screen.blit(surface, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))
        
        # Obramowanie (grubsze przy krachu)
        thickness = 4 if self.crashed else 2
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), self.radius, thickness)
        
        # Etykieta z typem i ceną
        font = get_font(22)
        type_labels = {
            settings.ZONE_TYPE_BOND: "BOND",
            settings.ZONE_TYPE_STOCK: "STOCK",
            settings.ZONE_TYPE_CRYPTO: "CRYPTO"
        }
        label = type_labels.get(self.zone_type, "?")
        
        if self.crashed:
            text = font.render(f"{label} KRACH!", True, (255, 50, 50))
        else:
            text = font.render(f"{label}", True, settings.COLOR_TEXT)
        text_rect = text.get_rect(center=(int(self.pos.x), int(self.pos.y) - 10))
        screen.blit(text, text_rect)
        
        # Cena
        price_font = get_font(20)
        price_text = price_font.render(f"${self.share_price:.2f}", True, settings.COLOR_TEXT)
        price_rect = price_text.get_rect(center=(int(self.pos.x), int(self.pos.y) + 10))
        screen.blit(price_text, price_rect)
