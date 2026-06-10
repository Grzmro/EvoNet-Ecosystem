import pygame
import neat
import os
import csv
import glob
import pickle
import time
import random
import settings
from simulation import Ecosystem, run_benchmark_market
from benchmarks import STRATEGY_HOLD, STRATEGY_MOMENTUM, STRATEGY_RANDOM

pygame.init()
pygame.display.set_caption("NEAT Portfolio Evolution — Information Asymmetry")

global_generation = 0
global_best_fitness = 0.0
fast_forward = False

last_group_stats = (0.0, 0.0)
last_neat_worth = 0.0
last_benchmark = {STRATEGY_HOLD: 0.0, STRATEGY_MOMENTUM: 0.0, STRATEGY_RANDOM: 0.0}
last_composition = (0.0, 0.0, 0.0, 0.0, 0.0)


class HeadlessReporter(neat.reporting.BaseReporter):
    def __init__(self, total_generations, csv_path):
        self.total_generations = total_generations
        self.current_generation = 0
        self.started = False
        self.gen_start_time = time.time()
        self.csv_path = csv_path

        csv_file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_file_exists:
                writer.writerow([
                    "generacja", "najlepszy_fitness", "sredni_fitness",
                    "liczba_gatunkow", "czas_epoki_s",
                    "avg_fitness_informed", "avg_fitness_uninformed",
                    "neat_avg_worth", "bench_hold6040", "bench_momentum", "bench_random",
                    "pct_bond", "pct_stock", "pct_crypto", "pct_cash", "avg_trades",
                ])

    def start_generation(self, generation):
        self.current_generation = generation
        self.gen_start_time = time.time()
        if not self.started:
            print(f"\n[START] Epoka {generation}, cel: {self.total_generations}.", flush=True)
            self.started = True

    def post_evaluate(self, config, population, species, best_genome):
        elapsed = time.time() - self.gen_start_time
        percent = (self.current_generation + 1) / self.total_generations * 100
        all_fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        avg_fitness = sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0.0
        num_species = len(species.species)

        informed_f = last_group_stats[0]
        uninformed_f = last_group_stats[1]
        b_hold = last_benchmark[STRATEGY_HOLD]
        b_mom = last_benchmark[STRATEGY_MOMENTUM]
        b_rnd = last_benchmark[STRATEGY_RANDOM]
        pct_bond, pct_stock, pct_crypto, pct_cash, avg_trades = last_composition

        print(
            f"Epoka {self.current_generation + 1}/{self.total_generations} ({percent:.1f}%) | "
            f"Best: {best_genome.fitness:.2f} | Avg: {avg_fitness:.2f} | "
            f"Gatunki: {num_species} | {elapsed:.1f}s",
            flush=True,
        )
        print(
            f"   > Informed: {informed_f:.1f} | Uninformed: {uninformed_f:.1f} | "
            f"NEAT worth: ${last_neat_worth:.0f} vs Hold: ${b_hold:.0f} / Mom: ${b_mom:.0f} / Rnd: ${b_rnd:.0f}",
            flush=True,
        )
        print(
            f"   > Portfel: Bond {pct_bond:.0f}% / Stock {pct_stock:.0f}% / "
            f"Crypto {pct_crypto:.0f}% / Cash {pct_cash:.0f}% | Trades: {avg_trades:.1f}",
            flush=True,
        )

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_generation + 1,
                f"{best_genome.fitness:.2f}",
                f"{avg_fitness:.2f}",
                num_species,
                f"{elapsed:.2f}",
                f"{informed_f:.2f}",
                f"{uninformed_f:.2f}",
                f"{last_neat_worth:.2f}",
                f"{b_hold:.2f}",
                f"{b_mom:.2f}",
                f"{b_rnd:.2f}",
                f"{pct_bond:.2f}",
                f"{pct_stock:.2f}",
                f"{pct_crypto:.2f}",
                f"{pct_cash:.2f}",
                f"{avg_trades:.2f}",
            ])


def eval_genomes(genomes, config):
    global global_generation, global_best_fitness, fast_forward
    global last_group_stats, last_neat_worth, last_benchmark, last_composition
    global_generation += 1

    gen_seed = settings.BASE_SEED + global_generation
    random.seed(gen_seed)

    sim = Ecosystem(genomes, config, generation=global_generation, best_fitness=global_best_fitness)

    screen = None
    clock = None
    if not settings.HEADLESS_MODE:
        screen = pygame.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
        clock = pygame.time.Clock()

    running = True
    paused = False
    skip_gen = False

    while running and sim.frame_count < settings.TIME_LIMIT_FRAMES and not sim.all_dead() and not skip_gen:
        if settings.HEADLESS_MODE:
            sim.run_frame(None)
            if sim.frame_count % 100 == 0:
                pygame.event.pump()
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    fast_forward = not fast_forward
                elif event.key == pygame.K_s:
                    skip_gen = True

        if paused:
            font = pygame.font.SysFont(None, 64)
            text = font.render("PAUZA", True, (255, 100, 100))
            screen.blit(text, text.get_rect(center=(settings.WORLD_WIDTH // 2, settings.WINDOW_HEIGHT // 2)))
            pygame.display.flip()
            clock.tick(15)
            continue

        if fast_forward:
            sim.run_frame(None)
            if sim.frame_count % 30 == 0:
                sim.run_frame(screen)
                pygame.event.pump()
        else:
            sim.run_frame(screen)
            clock.tick(settings.FPS)

    current_best = -9999.0
    for genome, agent in sim.agents:
        genome.fitness = agent.calculate_fitness(sim.all_zones)
        if genome.fitness > current_best:
            current_best = genome.fitness

    global_best_fitness = max(global_best_fitness, current_best)
    last_group_stats = sim.get_group_stats()
    last_neat_worth = sim.get_avg_worth()
    last_composition = sim.get_portfolio_composition()
    last_benchmark = run_benchmark_market(gen_seed)


def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not settings.LOAD_CHECKPOINT and os.path.exists(settings.CSV_LOG_FILE):
        os.remove(settings.CSV_LOG_FILE)

    checkpoints = glob.glob("checkpoints/neat-checkpoint-*")

    if settings.LOAD_CHECKPOINT and checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        loaded = False
        for checkpoint in checkpoints:
            try:
                print(f"\n[INFO] Wczytywanie: {checkpoint}...")
                p = neat.Checkpointer.restore_checkpoint(checkpoint)
                p.reporters.reporters.clear()
                global global_generation
                global_generation = p.generation
                loaded = True
                break
            except Exception as e:
                print(f"[BLAD] {checkpoint}: {e}")
        if not loaded:
            print("\n[INFO] Brak waznych checkpointow. Start od zera.")
            p = neat.Population(config)
    else:
        if not settings.LOAD_CHECKPOINT:
            print("\n[INFO] LOAD_CHECKPOINT=False. Start od zera.")
        else:
            print("\n[INFO] Brak checkpointow. Start od zera.")
        p = neat.Population(config)

    p.add_reporter(HeadlessReporter(settings.MAX_GENERATIONS, settings.CSV_LOG_FILE))

    if not settings.HEADLESS_MODE:
        p.add_reporter(neat.StatisticsReporter())

    p.add_reporter(neat.Checkpointer(50, filename_prefix="checkpoints/neat-checkpoint-"))

    remaining = max(1, settings.MAX_GENERATIONS - p.generation)
    winner = p.run(eval_genomes, remaining)

    print("\n[INFO] Trening zakonczony!")
    print(f"Najlepszy genom:\n{winner!s}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"models/best_agent_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)
    with open("models/best_agent_latest.pkl", "wb") as f:
        pickle.dump(winner, f)

    print(f"[INFO] Model zapisany: {model_path}")
    print(f"[INFO] Dane: {settings.CSV_LOG_FILE}")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
