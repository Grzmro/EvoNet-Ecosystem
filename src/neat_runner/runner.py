import glob
import math
import os
import pickle
import random
from statistics import mean, pstdev

import neat
import pygame
from neat.population import CompleteExtinctionException

from src.core.settings import (
    CHECKPOINT_EVERY_N_GENS,
    CHECKPOINT_PREFIX,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_TEXT,
    ENABLE_CHECKPOINTS,
    FPS,
    HEADLESS_MODE,
    LEFT_PANEL_WIDTH,
    MAX_TRAINING_GENERATIONS,
    RENDER_EVERY_N_GENS,
    REPLAY_GENOME_PATH,
    REPLAY_MODE,
    RIGHT_PANEL_WIDTH,
    RuntimeParams,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from src.entities.agent import Agent
from src.simulation.simulation import Simulation
from src.ui import Button, ReadOnlyValue, Slider
from src.ui.widgets import NumericInput


def load_config(config_path):
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def find_latest_checkpoint(project_root):
    pattern = os.path.join(project_root, f"{CHECKPOINT_PREFIX}*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def generation_key(path):
        base = os.path.basename(path)
        suffix = base.replace(CHECKPOINT_PREFIX, "")
        try:
            return int(suffix)
        except ValueError:
            return -1

    candidates.sort(key=generation_key)
    return candidates[-1]


def format_value(value, fmt, integer):
    if integer:
        return str(int(round(value)))
    return fmt.format(value)


class NeatEngine:
    def __init__(self, config, project_root, resume_from_checkpoint=True):
        self.config = config
        self.project_root = project_root
        self.best_genome_path = os.path.join(project_root, REPLAY_GENOME_PATH)
        self.best_fitness_ever = float("-inf")

        self.stats_reporter = neat.StatisticsReporter()
        self.population = self._create_population(resume_from_checkpoint)
        self.population.add_reporter(self.stats_reporter)

        if ENABLE_CHECKPOINTS:
            prefix = os.path.join(project_root, CHECKPOINT_PREFIX)
            self.population.add_reporter(
                neat.Checkpointer(
                    generation_interval=CHECKPOINT_EVERY_N_GENS,
                    filename_prefix=prefix,
                )
            )

        self.latest_stats = {
            "population": len(self.population.population),
            "generation": self.population.generation,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
            "std_fitness": 0.0,
            "species": [],
        }

    def _create_population(self, resume_from_checkpoint):
        if resume_from_checkpoint:
            latest = find_latest_checkpoint(self.project_root)
            if latest is not None:
                return neat.Checkpointer.restore_checkpoint(latest)
        return neat.Population(self.config)

    def reset(self):
        self.population = neat.Population(self.config)
        self.population.add_reporter(self.stats_reporter)

        if ENABLE_CHECKPOINTS:
            prefix = os.path.join(self.project_root, CHECKPOINT_PREFIX)
            self.population.add_reporter(
                neat.Checkpointer(
                    generation_interval=CHECKPOINT_EVERY_N_GENS,
                    filename_prefix=prefix,
                )
            )

    def build_generation(self, params):
        self.population.reporters.start_generation(self.population.generation)

        agents = []
        for _, genome in self.population.population.items():
            genome.fitness = 0.0
            network = neat.nn.FeedForwardNetwork.create(genome, self.config)
            x = random.uniform(0.0, WORLD_WIDTH)
            y = random.uniform(0.0, WORLD_HEIGHT)
            angle = random.uniform(-math.pi, math.pi)
            agents.append(
                Agent(
                    x,
                    y,
                    angle,
                    genome,
                    network,
                    params,
                    WORLD_WIDTH,
                    WORLD_HEIGHT,
                )
            )

        return Simulation(agents, params)

    def _update_stats(self):
        genomes = list(self.population.population.values())
        fitnesses = [g.fitness if g.fitness is not None else 0.0 for g in genomes]

        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_fitness = mean(fitnesses) if fitnesses else 0.0
        std_fitness = pstdev(fitnesses) if len(fitnesses) > 1 else 0.0

        species_rows = []
        for sid, species in self.population.species.species.items():
            sf = [g.fitness if g.fitness is not None else 0.0 for g in species.members.values()]
            species_rows.append(
                {
                    "id": sid,
                    "age": self.population.generation - species.created,
                    "size": len(species.members),
                    "fitness": max(sf) if sf else 0.0,
                }
            )

        species_rows.sort(key=lambda row: row["fitness"], reverse=True)
        self.latest_stats = {
            "population": len(self.population.population),
            "generation": self.population.generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "std_fitness": std_fitness,
            "species": species_rows,
        }

    def finish_generation(self):
        best = max(self.population.population.values(), key=lambda g: g.fitness or -1e9)

        if (best.fitness or -1e9) > self.best_fitness_ever:
            self.best_fitness_ever = best.fitness or -1e9
            with open(self.best_genome_path, "wb") as handle:
                pickle.dump(best, handle)

        self.population.reporters.post_evaluate(
            self.config,
            self.population.population,
            self.population.species,
            best,
        )
        self._update_stats()

        new_population = self.population.reproduction.reproduce(
            self.config,
            self.population.species,
            self.config.pop_size,
            self.population.generation,
        )

        if not new_population:
            self.population.reporters.complete_extinction()
            if self.config.reset_on_extinction:
                new_population = self.population.reproduction.create_new(
                    self.config.genome_type,
                    self.config.genome_config,
                    self.config.pop_size,
                )
            else:
                raise CompleteExtinctionException()

        self.population.population = new_population
        self.population.species.speciate(
            self.config,
            self.population.population,
            self.population.generation,
        )
        self.population.reporters.end_generation(
            self.config,
            self.population.population,
            self.population.species,
        )

        self.population.generation += 1


class HeadlessTrainer:
    def __init__(self, engine, params):
        self.engine = engine
        self.params = params
        self.stop_requested = False

    def _run_generation_fast(self, simulation):
        dt = 1.0 / FPS
        while not simulation.finished:
            simulation.update(dt)

    def _run_generation_rendered(self, simulation, generation):
        pygame.init()
        pygame.display.set_caption(f"Preview generation {generation}")
        screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)

        while not simulation.finished and not self.stop_requested:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_requested = True
                    break

            dt = clock.tick(FPS) / 1000.0
            simulation.update(dt)
            simulation.draw_world(screen)

            label = font.render(
                f"Generation: {generation} | Alive: {simulation.alive_count()} | Time: {simulation.elapsed_time:.1f}s",
                True,
                COLOR_TEXT,
            )
            screen.blit(label, (8, 8))
            pygame.display.flip()

        pygame.quit()

    def run(self):
        while not self.stop_requested:
            if MAX_TRAINING_GENERATIONS > 0 and self.engine.population.generation >= MAX_TRAINING_GENERATIONS:
                break

            generation = self.engine.population.generation
            simulation = self.engine.build_generation(self.params)

            render_now = RENDER_EVERY_N_GENS > 0 and generation % RENDER_EVERY_N_GENS == 0
            if render_now:
                self._run_generation_rendered(simulation, generation)
            else:
                self._run_generation_fast(simulation)

            if self.stop_requested:
                break

            self.engine.finish_generation()


class EcosystemApp:
    def __init__(self, engine):
        self.engine = engine
        self.params = RuntimeParams()
        self.default_params = RuntimeParams()

        pygame.init()
        pygame.display.set_caption("NEAT Ecosystem - GUI Control")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.world_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)

        self.running = True
        self.started = False
        self.paused = False
        self.request_reset = False

        self.simulation = None
        self.generation_stats = self.engine.latest_stats.copy()

        self._build_ui()

    def _build_ui(self):
        self.btn_start = Button((20, 20, 76, 34), "Start")
        self.btn_pause = Button((104, 20, 76, 34), "Pauza")
        self.btn_reset = Button((188, 20, 76, 34), "Reset")

        y = 92
        step = 55
        slider_width = 106
        input_width = 52
        default_width = 52
        input_x = 20 + slider_width + 8
        default_x = input_x + input_width + 8

        controls_spec = [
            ("food_count", "Food count", 10, 120, "{:.0f}", True),
            ("max_generation_time", "Gen time (s)", 5, 60, "{:.1f}", False),
            ("base_energy_drain", "Energy drain", 0.05, 0.8, "{:.2f}", False),
            ("food_energy_gain", "Food energy", 5, 80, "{:.2f}", False),
            ("food_fitness_reward", "Food fitness", 1, 20, "{:.2f}", False),
            ("turn_rate", "Turn rate", 0.04, 0.5, "{:.2f}", False),
            ("acceleration", "Acceleration", 0.05, 0.6, "{:.2f}", False),
            ("max_speed", "Max speed", 1, 8, "{:.2f}", False),
        ]

        self.controls = []
        for index, (field, label, minimum, maximum, fmt, integer) in enumerate(controls_spec):
            top = y + index * step
            current_value = getattr(self.params, field)
            default_value = getattr(self.default_params, field)

            slider = Slider((20, top, slider_width, 16), label, minimum, maximum, current_value, fmt, integer)
            input_box = NumericInput((input_x, top - 1, input_width, 18), minimum, maximum, current_value, fmt, integer)
            default_box = ReadOnlyValue((default_x, top - 1, default_width, 18), format_value(default_value, fmt, integer))

            self.controls.append((field, slider, input_box, default_box))

    def _reset_population(self):
        self.engine.reset()
        self.simulation = None
        self.started = False
        self.paused = False
        self.generation_stats = self.engine.latest_stats.copy()

    def _start_generation(self):
        self.simulation = self.engine.build_generation(self.params)
        self.generation_stats["population"] = len(self.simulation.agents)
        self.generation_stats["generation"] = self.engine.population.generation

    def _apply_slider_values(self):
        for field, slider, _, _ in self.controls:
            setattr(self.params, field, slider.value)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if self.btn_start.handle_event(event):
                self.started = True
                self.paused = False
                if self.simulation is None:
                    self._start_generation()

            if self.btn_pause.handle_event(event):
                self.paused = not self.paused

            if self.btn_reset.handle_event(event):
                self.request_reset = True

            for _, slider, input_box, _ in self.controls:
                moved = slider.handle_event(event)
                if moved:
                    input_box.set_value(slider.value)

                committed, value = input_box.handle_event(event)
                if committed and value is not None:
                    slider.value = value
                    input_box.set_value(value)

        self._apply_slider_values()

    def _draw_left_panel(self):
        panel_rect = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.line(
            self.screen,
            COLOR_BORDER,
            (LEFT_PANEL_WIDTH, 0),
            (LEFT_PANEL_WIDTH, WINDOW_HEIGHT),
            2,
        )

        title = self.font.render("Panel sterowania", True, COLOR_TEXT)
        self.screen.blit(title, (20, 64))

        now_label = self.small_font.render("Now", True, COLOR_MUTED)
        def_label = self.small_font.render("Def", True, COLOR_MUTED)
        self.screen.blit(now_label, (136, 72))
        self.screen.blit(def_label, (194, 72))

        self.btn_start.active = self.started and not self.paused
        self.btn_pause.active = self.started and self.paused
        self.btn_reset.active = False

        self.btn_start.draw(self.screen, self.small_font)
        self.btn_pause.draw(self.screen, self.small_font)
        self.btn_reset.draw(self.screen, self.small_font)

        for _, slider, input_box, default_box in self.controls:
            slider.draw(self.screen, self.small_font)
            input_box.draw(self.screen, self.small_font)
            default_box.draw(self.screen, self.small_font)

    def _draw_right_panel(self):
        x0 = LEFT_PANEL_WIDTH + WORLD_WIDTH
        panel_rect = pygame.Rect(x0, 0, RIGHT_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.line(self.screen, COLOR_BORDER, (x0, 0), (x0, WINDOW_HEIGHT), 2)

        y = 20
        line_h = 24

        lines = [
            "Statystyki na zywo",
            f"Generation: {self.generation_stats['generation']}",
            f"Population: {self.generation_stats['population']}",
            f"Best fitness: {self.generation_stats['best_fitness']:.2f}",
            f"Avg fitness: {self.generation_stats['avg_fitness']:.2f}",
            f"Std fitness: {self.generation_stats['std_fitness']:.2f}",
            f"Alive: {self.simulation.alive_count() if self.simulation else 0}",
            (
                f"Gen time: {self.simulation.elapsed_time:.1f}s"
                if self.simulation
                else "Gen time: 0.0s"
            ),
        ]

        for index, line in enumerate(lines):
            color = COLOR_TEXT if index == 0 else COLOR_MUTED
            font = self.font if index == 0 else self.small_font
            self.screen.blit(font.render(line, True, color), (x0 + 16, y))
            y += line_h

        y += 10
        species_label = "Species (age / size / fitness):"
        self.screen.blit(self.small_font.render(species_label, True, COLOR_TEXT), (x0 + 16, y))
        y += 24

        for row in self.generation_stats["species"][:10]:
            text = (
                f"S{row['id']:>2} | age {row['age']:>3} | "
                f"size {row['size']:>3} | fit {row['fitness']:.2f}"
            )
            self.screen.blit(self.small_font.render(text, True, COLOR_MUTED), (x0 + 16, y))
            y += 20

    def _draw(self):
        self.screen.fill(COLOR_BG)

        if self.simulation is not None:
            self.simulation.draw_world(self.world_surface)
        else:
            from src.core.settings import COLOR_WORLD

            self.world_surface.fill(COLOR_WORLD)

        self.screen.blit(self.world_surface, (LEFT_PANEL_WIDTH, 0))
        self._draw_left_panel()
        self._draw_right_panel()
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()

            if self.request_reset:
                self._reset_population()
                self.request_reset = False

            if self.started and not self.paused:
                if self.simulation is None:
                    self._start_generation()

                self.simulation.update(dt)
                if self.simulation.finished:
                    self.engine.finish_generation()
                    self.generation_stats = self.engine.latest_stats.copy()
                    self._start_generation()

            self._draw()

        pygame.quit()


def replay_best(config, project_root):
    genome_path = os.path.join(project_root, REPLAY_GENOME_PATH)
    if not os.path.exists(genome_path):
        raise FileNotFoundError(
            f"Nie znaleziono zapisanego najlepszego genomu: {genome_path}"
        )

    with open(genome_path, "rb") as handle:
        best_genome = pickle.load(handle)

    params = RuntimeParams()
    network = neat.nn.FeedForwardNetwork.create(best_genome, config)
    agent = Agent(
        random.uniform(0.0, WORLD_WIDTH),
        random.uniform(0.0, WORLD_HEIGHT),
        random.uniform(-math.pi, math.pi),
        best_genome,
        network,
        params,
        WORLD_WIDTH,
        WORLD_HEIGHT,
    )
    simulation = Simulation([agent], params)

    pygame.init()
    pygame.display.set_caption("Replay best genome")
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    running = True

    while running and not simulation.finished:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        dt = clock.tick(FPS) / 1000.0
        simulation.update(dt)
        simulation.draw_world(screen)
        text = font.render(
            f"Replay | Alive: {simulation.alive_count()} | Time: {simulation.elapsed_time:.1f}s",
            True,
            COLOR_TEXT,
        )
        screen.blit(text, (8, 8))
        pygame.display.flip()

    pygame.quit()


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, "config-feedforward.txt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Brak pliku config-feedforward.txt. Umiesc go w katalogu projektu."
        )

    config = load_config(config_path)

    if REPLAY_MODE:
        replay_best(config, project_root)
        return

    engine = NeatEngine(config, project_root, resume_from_checkpoint=True)

    if HEADLESS_MODE:
        trainer = HeadlessTrainer(engine, RuntimeParams())
        trainer.run()
        return

    app = EcosystemApp(engine)
    app.run()
