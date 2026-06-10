import math
import random
from dataclasses import dataclass
import pygame
from pygame.math import Vector2
import settings

_font_cache = {}
_zone_id_counter = 0


def poisson(lam: float) -> int:
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
    target_zone_id: int
    direction: float


class Guru:
    def __init__(self, x: float, y: float):
        self.pos = Vector2(x, y)
        self.waves = []
        self.cooldown = 0
        self.current_signal = None

    def update(self, frame_count: int, zones: list):
        self.current_signal = None

        for wave in self.waves:
            wave.radius += settings.WAVE_SPEED

        max_dist = math.hypot(settings.WORLD_WIDTH, settings.WINDOW_HEIGHT)
        self.waves = [w for w in self.waves if w.radius < max_dist]

        if self.cooldown <= 0:
            if zones:
                target_zone = max(zones, key=lambda z: abs(z.get_price_trend()))
                raw_direction = 1.0 if target_zone.get_price_trend() > 0 else -1.0
                noisy_direction = raw_direction + random.gauss(0, settings.GURU_SIGNAL_NOISE)
                noisy_direction = max(-1.0, min(1.0, noisy_direction))
                wave = InformationWave(
                    pos=Vector2(self.pos),
                    radius=0.0,
                    timestamp=frame_count,
                    target_zone_id=target_zone.zone_id,
                    direction=noisy_direction,
                )
                self.waves.append(wave)
                self.current_signal = (target_zone.zone_id, noisy_direction)
            self.cooldown = settings.WAVE_COOLDOWN
        else:
            self.cooldown -= 1


class InvestmentZone:
    def __init__(self, x: float, y: float, zone_type: float = None):
        self.zone_id = _next_zone_id()
        self.pos = Vector2(x, y)
        self.radius = settings.ZONE_RADIUS
        self.zone_type = zone_type if zone_type is not None else self._pick_random_type()
        self.share_price = 10.0
        self.price_history = [10.0] * 30
        self.chart_history = [10.0]
        self.age = 0
        self.crashed = False
        self.crash_cooldown = 0

    def _pick_random_type(self):
        return settings.ZONE_TYPE_STOCK if random.random() < 0.7 else settings.ZONE_TYPE_CRYPTO

    def update(self, num_investing_agents: int, frame_count: int):
        self.age += 1
        self.crashed = False

        if self.crash_cooldown > 0:
            self.crash_cooldown -= 1

        is_bear = frame_count > settings.BEAR_MARKET_START

        if self.zone_type == settings.ZONE_TYPE_BOND:
            lam_up = settings.BOND_LAMBDA_UP
            lam_down = settings.BOND_LAMBDA_DOWN
            jump_size = settings.BOND_JUMP_SIZE
        elif self.zone_type == settings.ZONE_TYPE_STOCK:
            lam_up = settings.STOCK_LAMBDA_UP
            lam_down = settings.STOCK_LAMBDA_DOWN
            jump_size = settings.STOCK_JUMP_SIZE
        else:
            lam_up = settings.CRYPTO_LAMBDA_UP
            lam_down = settings.CRYPTO_LAMBDA_DOWN
            jump_size = settings.CRYPTO_JUMP_SIZE

        if is_bear:
            lam_down *= settings.BEAR_LAMBDA_MULTIPLIER
        else:
            lam_up *= settings.BULL_LAMBDA_MULTIPLIER

        jumps_up = poisson(lam_up)
        jumps_down = poisson(lam_down)
        price_change = (jumps_up - jumps_down) * jump_size

        if self.zone_type != settings.ZONE_TYPE_BOND and num_investing_agents > 0:
            price_change += num_investing_agents * settings.CROWD_PRICE_BOOST

            if num_investing_agents >= settings.CROWD_CRASH_THRESHOLD and self.crash_cooldown <= 0:
                excess = num_investing_agents - settings.CROWD_CRASH_THRESHOLD + 1
                crash_prob = settings.CROWD_CRASH_BASE_PROB * excess
                if random.random() < crash_prob:
                    self.share_price *= settings.CRASH_PRICE_DROP
                    self.crashed = True
                    self.crash_cooldown = 60
                    self._update_price_history()
                    return True

        self.share_price += price_change
        self.share_price = max(self.share_price, 0.01)
        self._update_price_history()
        return False

    def _update_price_history(self):
        self.price_history.append(self.share_price)
        if len(self.price_history) > 30:
            self.price_history.pop(0)
        self.chart_history.append(self.share_price)

    def get_price_trend(self) -> float:
        if len(self.price_history) < 10:
            return 0.0
        recent = sum(self.price_history[-5:]) / 5.0
        older = sum(self.price_history[-15:-10]) / 5.0
        if older == 0:
            return 0.0
        change = (recent - older) / max(older, 0.01)
        return max(-1.0, min(1.0, change * 5.0))
