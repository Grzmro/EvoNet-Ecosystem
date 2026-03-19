import os
import pickle
import random
import math

import pygame

from src.core.settings import (
    FPS,
    REPLAY_GENOME_PATH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    COLOR_TEXT,
    RuntimeParams,
)
from src.entities.agent import Agent
from src.simulation.simulation import Simulation


def replay_best(config, project_root):
    genome_path = os.path.join(project_root, REPLAY_GENOME_PATH)
    if not os.path.exists(genome_path):
        raise FileNotFoundError(
            f"Nie znaleziono zapisanego najlepszego genomu: {genome_path}"
        )

    with open(genome_path, "rb") as handle:
        best_genome = pickle.load(handle)

    params = RuntimeParams()
    import neat

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
