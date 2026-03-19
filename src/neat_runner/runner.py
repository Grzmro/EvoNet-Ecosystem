import glob
import math
import os
import pickle
import random
from statistics import mean, pstdev

import neat
from neat.population import CompleteExtinctionException

from src.core.settings import (
    CHECKPOINT_EVERY_N_GENS,
    CHECKPOINT_PREFIX,
    CHECKPOINTS_DIR,
    ENABLE_CHECKPOINTS,
    HEADLESS_MODE,
    REPLAY_GENOME_PATH,
    REPLAY_MODE,
    RuntimeParams,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from src.entities.agent import Agent
from src.neat_runner.trainer import HeadlessTrainer
from src.neat_runner.replay import replay_best
from src.simulation.simulation import Simulation
from src.ui.ecosystem_app import EcosystemApp


def load_config(config_path):
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def find_latest_checkpoint(project_root):
    checkpoints_path = os.path.join(project_root, CHECKPOINTS_DIR)
    pattern = os.path.join(checkpoints_path, f"{CHECKPOINT_PREFIX}*")
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
            checkpoints_path = os.path.join(project_root, CHECKPOINTS_DIR)
            prefix = os.path.join(checkpoints_path, CHECKPOINT_PREFIX)
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
            checkpoints_path = os.path.join(self.project_root, CHECKPOINTS_DIR)
            prefix = os.path.join(checkpoints_path, CHECKPOINT_PREFIX)
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


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, "config-feedforward.txt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Brak pliku config-feedforward.txt. Umiesc go w katalogu projektu."
        )

    # Utwórz wymagane foldery
    checkpoints_dir = os.path.join(project_root, CHECKPOINTS_DIR)
    models_dir = os.path.dirname(os.path.join(project_root, REPLAY_GENOME_PATH))
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

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
