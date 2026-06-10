import math
import random
import settings
from entities import InvestmentZone


def _softmax(values: list[float]) -> list[float]:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class Agent:
    def __init__(self, genome_id, net, has_info_advantage: bool):
        self.genome_id = genome_id
        self.net = net
        self.has_info_advantage = has_info_advantage

        self.cash = settings.STARTING_CAPITAL
        self.portfolio = {}
        self.is_bankrupt = False
        self.survival_time = 0
        self.fitness = 0.0

        self.wealth_history = []
        self.guru_signal_zone_id = None
        self.guru_signal_direction = 0.0
        self.guru_freshness = 0.0

        self.metrics = {
            'trades': 0,
            'buys': 0,
            'sells': 0,
            'fees_paid': 0.0,
            'spent_on_bond': 0.0,
            'spent_on_stock': 0.0,
            'spent_on_crypto': 0.0,
        }

    def receive_guru_signal(self, zone_id: int, direction: float):
        self.guru_signal_zone_id = zone_id
        self.guru_signal_direction = direction
        self.guru_freshness = 1.0

    def get_total_worth(self, zones: list[InvestmentZone]) -> float:
        zone_map = {z.zone_id: z for z in zones}
        portfolio_value = sum(
            shares * zone_map[zid].share_price
            for zid, shares in self.portfolio.items()
            if zid in zone_map
        )
        return self.cash + portfolio_value

    def decide(self, zones: list[InvestmentZone]):
        if self.is_bankrupt:
            return
        inputs = self._build_inputs(zones)
        outputs = self.net.activate(inputs)
        self._execute_allocation(outputs, zones)

    def _build_inputs(self, zones: list[InvestmentZone]) -> tuple:
        total_worth = self.get_total_worth(zones)
        total_worth_safe = max(total_worth, 1e-6)

        prices_norm = [min(z.share_price / 10.0, 3.0) for z in zones]
        trends = [z.get_price_trend() for z in zones]
        portfolio_weights = []
        for z in zones:
            val = self.portfolio.get(z.zone_id, 0) * z.share_price
            portfolio_weights.append(val / total_worth_safe)

        cash_ratio = min(self.cash / total_worth_safe, 1.0)

        guru_freshness = self.guru_freshness if self.has_info_advantage else 0.0
        if self.has_info_advantage and self.guru_signal_zone_id is not None:
            zone_ids = [z.zone_id for z in zones]
            if self.guru_signal_zone_id in zone_ids:
                idx = zone_ids.index(self.guru_signal_zone_id)
                guru_target = (idx / max(len(zones) - 1, 1)) * 2.0 - 1.0
            else:
                guru_target = -1.0
            guru_direction = self.guru_signal_direction
        else:
            guru_target = -1.0
            guru_direction = 0.0

        return tuple(prices_norm + trends + portfolio_weights + [cash_ratio, guru_freshness, guru_target, guru_direction])

    def _execute_allocation(self, outputs: list[float], zones: list[InvestmentZone]):
        weights = _softmax(list(outputs))
        self._apply_target_weights(weights, zones)

    def _apply_target_weights(self, weights: list[float], zones: list[InvestmentZone]):
        total_worth = self.get_total_worth(zones)
        if total_worth <= 0:
            return

        zone_targets = weights[:len(zones)]

        for i, zone in enumerate(zones):
            current_value = self.portfolio.get(zone.zone_id, 0) * zone.share_price
            target_value = zone_targets[i] * total_worth

            if current_value > target_value + zone.share_price:
                excess_value = current_value - target_value
                shares_to_sell = max(1, int(excess_value / zone.share_price))
                self._sell_shares(zone, shares_to_sell)
            elif target_value > current_value + zone.share_price and self.cash > zone.share_price:
                budget = min(target_value - current_value, self.cash * 0.95)
                self._buy_shares(zone, budget)

    def _buy_shares(self, zone: InvestmentZone, budget: float):
        if zone.share_price <= 0 or budget <= 0 or zone.crashed:
            return
        fee = budget * settings.TRANSACTION_FEE_RATE
        effective_budget = budget - fee
        num_shares = int(effective_budget / zone.share_price)
        if num_shares <= 0:
            return
        cost = num_shares * zone.share_price
        total_cost = cost + fee
        if total_cost > self.cash:
            return
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

    def _sell_shares(self, zone: InvestmentZone, num_to_sell: int):
        held = self.portfolio.get(zone.zone_id, 0)
        if held <= 0:
            return
        num_to_sell = min(num_to_sell, held)
        gross = num_to_sell * zone.share_price
        fee = gross * settings.TRANSACTION_FEE_RATE
        self.cash += gross - fee
        self.portfolio[zone.zone_id] = held - num_to_sell
        if self.portfolio[zone.zone_id] <= 0:
            del self.portfolio[zone.zone_id]
        self.metrics['trades'] += 1
        self.metrics['sells'] += 1
        self.metrics['fees_paid'] += fee

    def update_state(self, zones: list[InvestmentZone]):
        if self.is_bankrupt:
            return
        self.cash *= (1.0 - settings.INFLATION_RATE)
        self.survival_time += 1
        if self.guru_freshness > 0:
            self.guru_freshness -= settings.GURU_SIGNAL_FRESHNESS_DECAY
            if self.guru_freshness < 0:
                self.guru_freshness = 0.0

        worth = self.get_total_worth(zones)
        self.wealth_history.append(worth)

        if self.cash <= 0.01 and not self.portfolio:
            self.is_bankrupt = True

    def sharpe_ratio(self) -> float:
        if len(self.wealth_history) < 10:
            return 0.0
        log_returns = [
            math.log(max(self.wealth_history[i], 1e-6) / max(self.wealth_history[i - 1], 1e-6))
            for i in range(1, len(self.wealth_history))
        ]
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns)
        std_r = math.sqrt(variance)
        return (mean_r / (std_r + 1e-6)) * math.sqrt(len(log_returns))

    def calculate_fitness(self, zones: list[InvestmentZone]) -> float:
        worth = self.get_total_worth(zones)

        log_growth = math.log(max(worth, 1e-6) / settings.STARTING_CAPITAL)
        growth_term = log_growth * settings.LOG_GROWTH_SCALE
        sharpe_term = self.sharpe_ratio() * settings.SHARPE_SCALE

        asset_classes = set()
        zone_map = {z.zone_id: z for z in zones}
        for zid, shares in self.portfolio.items():
            if shares > 0 and zid in zone_map:
                asset_classes.add(zone_map[zid].zone_type)
        diversification = (len(asset_classes) - 1) * settings.DIVERSIFICATION_BONUS if len(asset_classes) > 1 else 0.0

        base = growth_term + sharpe_term + diversification
        self.fitness = base - settings.PENALTY_BANKRUPTCY if self.is_bankrupt else base
        return self.fitness
