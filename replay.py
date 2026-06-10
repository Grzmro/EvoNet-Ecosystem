import pygame
import neat
import pickle
import os
import settings
from simulation import Ecosystem
from benchmarks import BenchmarkAgent, STRATEGY_HOLD, STRATEGY_MOMENTUM, STRATEGY_RANDOM

settings.HEADLESS_MODE = False

pygame.init()
pygame.display.set_caption("NEAT Portfolio — Replay vs Benchmarki")


def run_replay():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    model_path = "models/best_agent_latest.pkl"
    if not os.path.exists(model_path):
        print("[BLAD] Brak pliku models/best_agent_latest.pkl — uruchom najpierw trening.")
        return

    with open(model_path, "rb") as f:
        best_genome = pickle.load(f)

    def new_episode():
        best_genome.fitness = 0.0
        sim = Ecosystem([(best_genome.key, best_genome)], config, generation=0, best_fitness=0.0)
        # Ten sam rynek — dodaj po jednym agencie z kazdej strategii benchmarkowej.
        # genome=None jest bezpieczne: run_frame nigdy nie odwoluje sie do genomu w petli.
        for strategy in [STRATEGY_HOLD, STRATEGY_MOMENTUM, STRATEGY_RANDOM]:
            sim.agents.append((None, BenchmarkAgent(strategy, genome_id=f"bench_{strategy}")))
        return sim

    sim = new_episode()
    episode = 1

    screen = pygame.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    fast_forward = False
    paused = False

    print("[INFO] Sterowanie: SPACJA=pauza, F=fast forward, Q=wyjscie")
    print("[INFO] Gracze: NEAT + Hold60/40 + Momentum + Random (ten sam rynek)")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    fast_forward = not fast_forward

        if paused:
            font = pygame.font.SysFont(None, 64)
            text = font.render("PAUZA", True, (255, 100, 100))
            screen.blit(text, text.get_rect(center=(settings.WORLD_WIDTH // 2, settings.WINDOW_HEIGHT // 2)))
            pygame.display.flip()
            clock.tick(15)
            continue

        if fast_forward:
            sim.run_frame(None)
            if sim.frame_count % 10 == 0:
                sim.run_frame(screen)
                pygame.event.pump()
        else:
            sim.run_frame(screen)
            clock.tick(settings.FPS)

        if sim.frame_count >= settings.TIME_LIMIT_FRAMES or sim.all_dead():
            episode += 1
            print(f"[INFO] Nowy epizod #{episode}")
            sim = new_episode()


if __name__ == "__main__":
    run_replay()
