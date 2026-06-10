import pygame
import random
import neat
import settings
from entities import Guru, InvestmentZone, get_font
from agent import Agent
from benchmarks import build_benchmark_agents, ALL_STRATEGIES


def build_zones():
    bond_zone = InvestmentZone(
        settings.WORLD_WIDTH // 2,
        settings.WINDOW_HEIGHT - 80,
        zone_type=settings.ZONE_TYPE_BOND,
    )
    bond_zone.radius = 80

    stock_zones = [
        InvestmentZone(200 + i * 300, 300, zone_type=settings.ZONE_TYPE_STOCK)
        for i in range(3)
    ]

    crypto_zone = InvestmentZone(
        settings.WORLD_WIDTH // 2,
        100,
        zone_type=settings.ZONE_TYPE_CRYPTO,
    )

    return [bond_zone] + stock_zones + [crypto_zone]


class Ecosystem:
    def __init__(self, genomes, config, generation=1, best_fitness=0.0):
        self.frame_count = 0
        self.generation = generation
        self.best_fitness = best_fitness
        self.config = config

        self.guru = Guru(settings.WORLD_WIDTH / 2, settings.WINDOW_HEIGHT / 2)
        self.last_signal = None

        self.all_zones = build_zones()
        self.bond_zone = self.all_zones[0]
        self.stock_zones = self.all_zones[1:4]
        self.crypto_zone = self.all_zones[4]

        self.agents = []
        num_genomes = len(genomes)
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            has_adv = (i / max(num_genomes - 1, 1)) < settings.INFORMATION_ADVANTAGE_FRACTION
            agent = Agent(genome_id, net, has_info_advantage=has_adv)
            genome.fitness = 0.0
            self.agents.append((genome, agent))

    def run_frame(self, screen=None, _draw_vision=False):
        self.frame_count += 1

        self.guru.update(self.frame_count, self.all_zones)
        current_signal = self.guru.current_signal
        if current_signal:
            self.last_signal = (current_signal[0], current_signal[1], self.frame_count)

        investing_counts = {zone.zone_id: 0 for zone in self.all_zones}
        for _, agent in self.agents:
            if agent.is_bankrupt:
                continue
            for zid, shares in agent.portfolio.items():
                if shares > 0 and zid in investing_counts:
                    investing_counts[zid] += 1

        crashed_zones = []
        for zone in self.all_zones:
            did_crash = zone.update(investing_counts.get(zone.zone_id, 0), self.frame_count)
            if did_crash:
                crashed_zones.append(zone)

        current_best = -9999.0
        for genome, agent in self.agents:
            if agent.is_bankrupt:
                continue

            if current_signal and agent.has_info_advantage:
                agent.receive_guru_signal(*current_signal)

            agent.decide(self.all_zones)
            agent.update_state(self.all_zones)

            worth = agent.get_total_worth(self.all_zones)
            if worth > current_best:
                current_best = worth

        if screen:
            self._draw(screen, current_best)

        return current_best

    def _investor_count(self, zone):
        return sum(
            1 for _, a in self.agents
            if not a.is_bankrupt and a.portfolio.get(zone.zone_id, 0) > 0
        )

    def _draw(self, screen, current_best):
        screen.fill(settings.COLOR_BG)

        self._draw_header(screen)
        self._draw_price_chart(screen, pygame.Rect(55, 78, 890, 352))
        self._draw_crowd_bars(screen, pygame.Rect(55, 452, 890, 132))
        self._draw_wealth_distribution(screen, pygame.Rect(55, 606, 890, 176))
        self._draw_side_panel(screen, current_best)

        pygame.display.flip()

    def _draw_header(self, screen):
        is_bear = self.frame_count > settings.BEAR_MARKET_START
        phase_color = settings.COLOR_BEAR if is_bear else settings.COLOR_BULL
        phase_label = "BESSA (BEAR)" if is_bear else "HOSSA (BULL)"

        screen.blit(get_font(34).render("EvoNet Trading Desk", True, settings.COLOR_PANEL_TEXT), (20, 14))

        badge = get_font(26).render(phase_label, True, (15, 15, 20))
        bw, bh = badge.get_size()
        badge_rect = pygame.Rect(settings.WORLD_WIDTH - bw - 40, 16, bw + 24, bh + 8)
        pygame.draw.rect(screen, phase_color, badge_rect, border_radius=6)
        screen.blit(badge, (badge_rect.x + 12, badge_rect.y + 4))

    def _draw_price_chart(self, screen, rect):
        pygame.draw.rect(screen, settings.COLOR_CHART_BG, rect, border_radius=8)

        title = get_font(22).render("Ceny aktywow (wzgledem ceny startowej)", True, settings.COLOR_TEXT)
        screen.blit(title, (rect.x + 12, rect.y + 8))

        plot = pygame.Rect(rect.x + 55, rect.y + 42, rect.width - 70, rect.height - 70)

        max_frames = max(settings.TIME_LIMIT_FRAMES, 1)

        # Przesuwajace sie okno: pokazuj ostatnie max_frames probek, dzieki czemu
        # wykres dalej "jedzie" po przekroczeniu limitu klatek (np. ciagly replay.py).
        history_offset = 0
        series = []
        norm_min, norm_max = 1.0, 1.0
        for zone in self.all_zones:
            hist = zone.chart_history
            base = hist[0] if hist and hist[0] > 0 else 1.0
            history_offset = max(0, len(hist) - 1 - max_frames)
            window = hist[history_offset:]
            norm = [p / base for p in window]
            series.append(norm)
            if norm:
                norm_min = min(norm_min, min(norm))
                norm_max = max(norm_max, max(norm))

        pad = (norm_max - norm_min) * 0.1 or 0.1
        y_lo = max(0.0, norm_min - pad)
        y_hi = norm_max + pad
        span = max(y_hi - y_lo, 1e-6)

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            gy = plot.bottom - frac * plot.height
            pygame.draw.line(screen, settings.COLOR_GRID, (plot.x, gy), (plot.right, gy), 1)
            val = y_lo + frac * span
            lbl = get_font(16).render(f"{val:.1f}x", True, settings.COLOR_AXIS)
            screen.blit(lbl, (rect.x + 10, gy - 8))

        bear_frame = settings.BEAR_MARKET_START - history_offset
        if 0 <= bear_frame <= max_frames:
            bear_x = plot.x + (bear_frame / max_frames) * plot.width
            if plot.x <= bear_x < plot.right:
                pygame.draw.line(screen, settings.COLOR_BEAR, (bear_x, plot.y), (bear_x, plot.bottom), 1)
                screen.blit(get_font(15).render("bessa", True, settings.COLOR_BEAR), (bear_x + 3, plot.y + 2))

        for idx, norm in enumerate(series):
            if len(norm) < 2:
                continue
            color = settings.ASSET_COLORS[idx % len(settings.ASSET_COLORS)]
            points = [
                (plot.x + (i / max_frames) * plot.width,
                 plot.bottom - ((v - y_lo) / span) * plot.height)
                for i, v in enumerate(norm)
            ]
            pygame.draw.lines(screen, color, False, points, 2)

        lx = plot.x + 8
        ly = plot.y + 6
        for idx, zone in enumerate(self.all_zones):
            color = settings.ASSET_COLORS[idx % len(settings.ASSET_COLORS)]
            label = settings.ASSET_LABELS[idx % len(settings.ASSET_LABELS)]
            pygame.draw.rect(screen, color, (lx, ly, 12, 12), border_radius=2)
            txt = get_font(15).render(f"{label}  ${zone.share_price:.1f}", True, settings.COLOR_TEXT)
            screen.blit(txt, (lx + 16, ly - 2))
            ly += 19

    def _draw_crowd_bars(self, screen, rect):
        pygame.draw.rect(screen, settings.COLOR_CHART_BG, rect, border_radius=8)
        screen.blit(get_font(22).render(
            f"Inwestorzy w aktywie  (prog krachu: {settings.CROWD_CRASH_THRESHOLD}, linia czerwona)",
            True, settings.COLOR_TEXT), (rect.x + 12, rect.y + 8))

        n = len(self.all_zones)
        slot_w = (rect.width - 40) / n
        base_y = rect.bottom - 26
        max_h = rect.height - 70
        max_count = max(len(self.agents), settings.CROWD_CRASH_THRESHOLD * 2, 1)

        threshold_y = base_y - (settings.CROWD_CRASH_THRESHOLD / max_count) * max_h
        pygame.draw.line(screen, settings.COLOR_BEAR, (rect.x + 20, threshold_y), (rect.right - 20, threshold_y), 1)

        for idx, zone in enumerate(self.all_zones):
            count = self._investor_count(zone)
            color = settings.ASSET_COLORS[idx % len(settings.ASSET_COLORS)]
            cx = rect.x + 20 + idx * slot_w + slot_w / 2
            h = min(count / max_count, 1.0) * max_h
            bar_w = slot_w * 0.5
            over = count >= settings.CROWD_CRASH_THRESHOLD
            bar_color = settings.COLOR_BEAR if (over or zone.crashed) else color
            pygame.draw.rect(screen, bar_color, (cx - bar_w / 2, base_y - h, bar_w, h), border_radius=3)
            screen.blit(get_font(16).render(str(count), True, settings.COLOR_PANEL_TEXT),
                        (cx - 6, base_y - h - 18))
            lbl = get_font(15).render(settings.ASSET_LABELS[idx], True, settings.COLOR_AXIS)
            screen.blit(lbl, (cx - lbl.get_width() / 2, base_y + 4))

    def _draw_wealth_distribution(self, screen, rect):
        pygame.draw.rect(screen, settings.COLOR_CHART_BG, rect, border_radius=8)
        screen.blit(get_font(22).render("Rozklad majatku populacji", True, settings.COLOR_TEXT),
                    (rect.x + 12, rect.y + 8))

        worths = [a.get_total_worth(self.all_zones) for _, a in self.agents]
        if not worths:
            return

        plot = pygame.Rect(rect.x + 20, rect.y + 44, rect.width - 40, rect.height - 70)
        hi = max(worths)
        lo = min(worths)
        span = max(hi - lo, 1e-6)
        num_bins = 24
        bins = [0] * num_bins
        for w in worths:
            b = min(int((w - lo) / span * num_bins), num_bins - 1)
            bins[b] += 1
        max_bin = max(bins) or 1
        bar_w = plot.width / num_bins

        for i, c in enumerate(bins):
            h = (c / max_bin) * plot.height
            x = plot.x + i * bar_w
            pygame.draw.rect(screen, (90, 150, 255), (x, plot.bottom - h, bar_w - 2, h))

        avg = sum(worths) / len(worths)
        avg_x = plot.x + ((avg - lo) / span) * plot.width
        pygame.draw.line(screen, settings.COLOR_BULL, (avg_x, plot.y), (avg_x, plot.bottom), 2)
        screen.blit(get_font(15).render(f"sr. ${avg:.0f}", True, settings.COLOR_BULL), (avg_x + 4, plot.y))
        screen.blit(get_font(15).render(f"${lo:.0f}", True, settings.COLOR_AXIS), (plot.x, plot.bottom + 4))
        screen.blit(get_font(15).render(f"${hi:.0f}", True, settings.COLOR_AXIS), (plot.right - 50, plot.bottom + 4))

    def _draw_side_panel(self, screen, current_best):
        px = settings.WORLD_WIDTH
        pygame.draw.rect(screen, settings.COLOR_PANEL, (px, 0, settings.PANEL_WIDTH, settings.WINDOW_HEIGHT))
        pygame.draw.line(screen, settings.COLOR_BG, (px, 0), (px, settings.WINDOW_HEIGHT), 2)

        x = px + 16
        y = 16
        ft = get_font(20)
        fh = get_font(24)

        active = sum(1 for _, a in self.agents if not a.is_bankrupt)
        total = len(self.agents)
        informed = sum(1 for _, a in self.agents if a.has_info_advantage and not a.is_bankrupt)
        avg_worth = (sum(a.get_total_worth(self.all_zones) for _, a in self.agents if not a.is_bankrupt) / active) if active else 0.0

        screen.blit(fh.render("SYGNAL GURU", True, settings.COLOR_GURU), (x, y))
        y += 30
        y = self._draw_guru_box(screen, x, y)

        y += 14
        screen.blit(fh.render("NAJLEPSZY PORTFEL", True, settings.COLOR_PANEL_TEXT), (x, y))
        y += 30
        y = self._draw_best_portfolio(screen, x, y)

        y += 14
        screen.blit(fh.render("STATYSTYKI", True, settings.COLOR_PANEL_TEXT), (x, y))
        y += 30
        for text in [
            f"Generacja: {self.generation}",
            f"Klatka: {self.frame_count}/{settings.TIME_LIMIT_FRAMES}",
            f"Aktywni: {active}/{total}",
            f"Z info edge: {informed}",
            f"Sr. majatek: ${avg_worth:.1f}",
            f"Best majatek: ${current_best:.1f}",
        ]:
            screen.blit(ft.render(text, True, settings.COLOR_PANEL_TEXT), (x, y))
            y += 24

        y = settings.WINDOW_HEIGHT - 110
        screen.blit(fh.render("STEROWANIE", True, settings.COLOR_PANEL_TEXT), (x, y))
        y += 28
        for text in ["SPACJA - Pauza", "F - Fast Forward", "S - Skip generacji"]:
            screen.blit(ft.render(text, True, settings.COLOR_AXIS), (x, y))
            y += 22

    def _draw_guru_box(self, screen, x, y):
        box = pygame.Rect(x, y, settings.PANEL_WIDTH - 32, 78)
        pygame.draw.rect(screen, settings.COLOR_CHART_BG, box, border_radius=6)
        ft = get_font(20)

        if not self.last_signal:
            screen.blit(ft.render("Brak sygnalu", True, settings.COLOR_AXIS), (x + 12, y + 28))
            return box.bottom

        zone_id, direction, emit_frame = self.last_signal
        idx = next((i for i, z in enumerate(self.all_zones) if z.zone_id == zone_id), 0)
        color = settings.ASSET_COLORS[idx % len(settings.ASSET_COLORS)]
        label = settings.ASSET_LABELS[idx % len(settings.ASSET_LABELS)]

        bullish = direction >= 0
        dir_color = settings.COLOR_BULL if bullish else settings.COLOR_BEAR
        dir_text = "KUPUJ ^" if bullish else "SPRZEDAJ v"

        pygame.draw.rect(screen, color, (x + 12, y + 14, 14, 14), border_radius=3)
        screen.blit(ft.render(label, True, settings.COLOR_PANEL_TEXT), (x + 32, y + 12))
        screen.blit(ft.render(dir_text, True, dir_color), (x + 12, y + 38))

        elapsed = self.frame_count - emit_frame
        freshness = max(0.0, 1.0 - elapsed * settings.GURU_SIGNAL_FRESHNESS_DECAY)
        bar_x = x + 130
        pygame.draw.rect(screen, settings.COLOR_GRID, (bar_x, y + 42, 130, 10), border_radius=4)
        pygame.draw.rect(screen, settings.COLOR_GURU, (bar_x, y + 42, int(130 * freshness), 10), border_radius=4)
        return box.bottom

    def _draw_best_portfolio(self, screen, x, y):
        if not self.agents:
            return y
        best = max((a for _, a in self.agents), key=lambda a: a.get_total_worth(self.all_zones))
        zone_map = {z.zone_id: z for z in self.all_zones}
        bond = stock = crypto = 0.0
        for zid, sh in best.portfolio.items():
            if zid in zone_map:
                val = sh * zone_map[zid].share_price
                zt = zone_map[zid].zone_type
                if zt == settings.ZONE_TYPE_BOND:
                    bond += val
                elif zt == settings.ZONE_TYPE_STOCK:
                    stock += val
                else:
                    crypto += val
        cash = best.cash
        total = bond + stock + crypto + cash
        if total <= 0:
            return y

        segments = [
            ("Bond", bond, settings.ASSET_COLORS[0]),
            ("Stock", stock, settings.ASSET_COLORS[1]),
            ("Crypto", crypto, settings.ASSET_COLORS[4]),
            ("Cash", cash, (140, 140, 150)),
        ]
        bar = pygame.Rect(x, y, settings.PANEL_WIDTH - 32, 26)
        cx = bar.x
        for _, val, color in segments:
            w = (val / total) * bar.width
            if w > 0:
                pygame.draw.rect(screen, color, (cx, bar.y, w, bar.height))
                cx += w
        y = bar.bottom + 8

        fs = get_font(18)
        for name, val, color in segments:
            pct = val / total * 100
            pygame.draw.rect(screen, color, (x, y + 2, 12, 12), border_radius=2)
            screen.blit(fs.render(f"{name}: {pct:.0f}%", True, settings.COLOR_PANEL_TEXT), (x + 18, y))
            y += 20
        screen.blit(fs.render(f"Majatek: ${best.get_total_worth(self.all_zones):.0f}", True, settings.COLOR_BULL), (x, y))
        return y + 22

    def all_dead(self):
        return all(agent.is_bankrupt for _, agent in self.agents)

    def get_group_stats(self):
        informed_fits = [a.fitness for _, a in self.agents if a.has_info_advantage and a.fitness is not None]
        uninformed_fits = [a.fitness for _, a in self.agents if not a.has_info_advantage and a.fitness is not None]
        avg_informed = sum(informed_fits) / len(informed_fits) if informed_fits else 0.0
        avg_uninformed = sum(uninformed_fits) / len(uninformed_fits) if uninformed_fits else 0.0
        return avg_informed, avg_uninformed

    def get_avg_worth(self):
        worths = [a.get_total_worth(self.all_zones) for _, a in self.agents]
        return sum(worths) / len(worths) if worths else 0.0

    def get_portfolio_composition(self):
        zone_map = {z.zone_id: z for z in self.all_zones}
        tot_bond = tot_stock = tot_crypto = tot_cash = 0.0
        total_trades = 0
        num_agents = len(self.agents)
        for _, agent in self.agents:
            for zid, shares in agent.portfolio.items():
                if zid in zone_map:
                    val = shares * zone_map[zid].share_price
                    zt = zone_map[zid].zone_type
                    if zt == settings.ZONE_TYPE_BOND:
                        tot_bond += val
                    elif zt == settings.ZONE_TYPE_STOCK:
                        tot_stock += val
                    else:
                        tot_crypto += val
            tot_cash += agent.cash
            total_trades += agent.metrics['trades']

        total = tot_bond + tot_stock + tot_crypto + tot_cash
        avg_trades = total_trades / num_agents if num_agents else 0.0
        if total <= 0:
            return (0.0, 0.0, 0.0, 0.0, avg_trades)
        return (
            tot_bond / total * 100,
            tot_stock / total * 100,
            tot_crypto / total * 100,
            tot_cash / total * 100,
            avg_trades,
        )


def run_benchmark_market(seed: int, frames: int = None) -> dict:
    frames = frames or settings.TIME_LIMIT_FRAMES
    random.seed(seed)

    zones = build_zones()
    guru = Guru(settings.WORLD_WIDTH / 2, settings.WINDOW_HEIGHT / 2)
    agents = build_benchmark_agents()

    for frame in range(1, frames + 1):
        guru.update(frame, zones)

        investing_counts = {z.zone_id: 0 for z in zones}
        for agent in agents:
            if agent.is_bankrupt:
                continue
            for zid, shares in agent.portfolio.items():
                if shares > 0 and zid in investing_counts:
                    investing_counts[zid] += 1

        for zone in zones:
            zone.update(investing_counts.get(zone.zone_id, 0), frame)

        for agent in agents:
            if agent.is_bankrupt:
                continue
            agent.decide(zones)
            agent.update_state(zones)

    results = {}
    for strategy in ALL_STRATEGIES:
        worths = [a.get_total_worth(zones) for a in agents if a.strategy == strategy]
        results[strategy] = sum(worths) / len(worths) if worths else 0.0
    return results
