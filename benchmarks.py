import random
import settings
from agent import Agent
from entities import InvestmentZone

STRATEGY_HOLD = "hold6040"
STRATEGY_MOMENTUM = "momentum"
STRATEGY_RANDOM = "random"

ALL_STRATEGIES = [STRATEGY_HOLD, STRATEGY_MOMENTUM, STRATEGY_RANDOM]


class BenchmarkAgent(Agent):
    def __init__(self, strategy: str, genome_id):
        super().__init__(genome_id, net=None, has_info_advantage=False)
        self.strategy = strategy
        self._frame = 0

    def decide(self, zones: list[InvestmentZone]):
        if self.is_bankrupt:
            return
        self._frame += 1

        if self.strategy == STRATEGY_HOLD:
            if self._frame == 1:
                self._apply_target_weights(self._weights_hold(zones), zones)
        elif self.strategy == STRATEGY_MOMENTUM:
            if (self._frame - 1) % settings.MOMENTUM_REBALANCE_INTERVAL == 0:
                self._apply_target_weights(self._weights_momentum(zones), zones)
        elif self.strategy == STRATEGY_RANDOM:
            if (self._frame - 1) % settings.MOMENTUM_REBALANCE_INTERVAL == 0:
                self._apply_target_weights(self._weights_random(zones), zones)

    def _weights_hold(self, zones: list[InvestmentZone]) -> list[float]:
        bond_zones = [i for i, z in enumerate(zones) if z.zone_type == settings.ZONE_TYPE_BOND]
        stock_zones = [i for i, z in enumerate(zones) if z.zone_type == settings.ZONE_TYPE_STOCK]
        weights = [0.0] * (len(zones) + 1)
        if bond_zones:
            for i in bond_zones:
                weights[i] = 0.4 / len(bond_zones)
        if stock_zones:
            for i in stock_zones:
                weights[i] = 0.6 / len(stock_zones)
        return weights

    def _weights_momentum(self, zones: list[InvestmentZone]) -> list[float]:
        weights = [0.0] * (len(zones) + 1)
        trends = [z.get_price_trend() for z in zones]
        best_idx = max(range(len(zones)), key=lambda i: trends[i])
        if trends[best_idx] > 0:
            weights[best_idx] = 0.9
            weights[-1] = 0.1
        else:
            weights[-1] = 1.0
        return weights

    def _weights_random(self, zones: list[InvestmentZone]) -> list[float]:
        raw = [random.random() for _ in range(len(zones) + 1)]
        total = sum(raw)
        return [r / total for r in raw]


def build_benchmark_agents() -> list[BenchmarkAgent]:
    agents = []
    for strategy in ALL_STRATEGIES:
        for k in range(settings.BENCHMARK_AGENTS_PER_STRATEGY):
            agents.append(BenchmarkAgent(strategy, genome_id=f"{strategy}_{k}"))
    return agents
