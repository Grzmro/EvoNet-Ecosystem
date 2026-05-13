import pygame
import neat
import os
import csv
import glob
import pickle
import time
import settings
from simulation import Ecosystem

# Ustawienia Pygame
pygame.init()
pygame.display.set_caption("Ewolucja agentów w warunkach asymetrii informacyjnej")

# Globalne zmienne stanu UI
global_generation = 0
global_best_fitness = 0.0
fast_forward = False
draw_vision = True
last_run_metrics = {
    'avg_trades': 0.0,
    'bond_pct': 0.0,
    'stock_pct': 0.0,
    'crypto_pct': 0.0,
    'bond_spend_pct': 0.0,
    'stock_spend_pct': 0.0,
    'crypto_spend_pct': 0.0
}

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
                writer.writerow(["generacja", "najlepszy_fitness", "sredni_fitness", "liczba_gatunkow", "czas_epoki_s",
                                 "avg_trades", "time_bond_pct", "time_stock_pct", "time_crypto_pct",
                                 "budget_bond_pct", "budget_stock_pct", "budget_crypto_pct"])

    def start_generation(self, generation):
        self.current_generation = generation
        self.gen_start_time = time.time()
        if not self.started:
            print(f"\n[HEADLESS MODE] Trening w toku... Start od epoki {generation}, cel: {self.total_generations}.", flush=True)
            self.started = True

    def post_evaluate(self, config, population, species, best_genome):
        elapsed = time.time() - self.gen_start_time
        percent = (self.current_generation + 1) / self.total_generations * 100
        
        all_fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        avg_fitness = sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0.0
        num_species = len(species.species)
        
        m = last_run_metrics
        
        print(f"Epoka {self.current_generation + 1}/{self.total_generations} ({percent:.1f}%) | "
              f"Best: {best_genome.fitness:.2f} | Avg: {avg_fitness:.2f} | "
              f"Gatunki: {num_species} | Czas: {elapsed:.1f}s", flush=True)
        print(f"   > Strategia: Transakcje: {m['avg_trades']:.1f} | Czas[B/S/C]: {m['bond_pct']:.1f}%/{m['stock_pct']:.1f}%/{m['crypto_pct']:.1f}% | Wydatki[B/S/C]: {m['bond_spend_pct']:.1f}%/{m['stock_spend_pct']:.1f}%/{m['crypto_spend_pct']:.1f}%", flush=True)
        
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_generation + 1, 
                f"{best_genome.fitness:.2f}", 
                f"{avg_fitness:.2f}", 
                num_species, 
                f"{elapsed:.2f}",
                f"{m['avg_trades']:.2f}",
                f"{m['bond_pct']:.2f}",
                f"{m['stock_pct']:.2f}",
                f"{m['crypto_pct']:.2f}",
                f"{m['bond_spend_pct']:.2f}",
                f"{m['stock_spend_pct']:.2f}",
                f"{m['crypto_spend_pct']:.2f}"
            ])

def eval_genomes(genomes, config):
    global global_generation, global_best_fitness, fast_forward, draw_vision
    global_generation += 1
    
    # Tworzymy symulację
    sim = Ecosystem(genomes, config, generation=global_generation, best_fitness=global_best_fitness)
    
    # Okno
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
            sim.run_frame(None, False)
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
                elif event.key == pygame.K_v:
                    draw_vision = not draw_vision
                    
        if paused:
            font = pygame.font.SysFont(None, 64)
            text = font.render("PAUZA", True, (255, 100, 100))
            text_rect = text.get_rect(center=(settings.WORLD_WIDTH//2, settings.WINDOW_HEIGHT//2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(15)
            continue
            
        if fast_forward:
            sim.run_frame(None, False)
            if sim.frame_count % 30 == 0:
                sim.run_frame(screen, draw_vision)
                pygame.event.pump()
        else:
            sim.run_frame(screen, draw_vision)
            clock.tick(settings.FPS)
            
    # Zapisz najlepszy fitness w bieżącej generacji
    current_best = -9999.0
    for _, genome in genomes:
        if genome.fitness is not None and genome.fitness > current_best:
            current_best = genome.fitness
            
    # Agregacja statystyk dla całej populacji
    total_trades = 0
    t_bond = 0
    t_stock = 0
    t_crypto = 0
    s_bond = 0.0
    s_stock = 0.0
    s_crypto = 0.0
    for _, agent in sim.agents:
        total_trades += agent.metrics.get('trades', 0)
        t_bond += agent.metrics.get('time_in_bond', 0)
        t_stock += agent.metrics.get('time_in_stock', 0)
        t_crypto += agent.metrics.get('time_in_crypto', 0)
        s_bond += agent.metrics.get('spent_on_bond', 0.0)
        s_stock += agent.metrics.get('spent_on_stock', 0.0)
        s_crypto += agent.metrics.get('spent_on_crypto', 0.0)
    
    num_a = len(sim.agents)
    global last_run_metrics
    if num_a > 0:
        total_spent = s_bond + s_stock + s_crypto
        if total_spent == 0:
            total_spent = 1.0  # Zapobiega dzieleniu przez zero
            
        last_run_metrics = {
            'avg_trades': total_trades / num_a,
            'bond_pct': (t_bond / (num_a * settings.TIME_LIMIT_FRAMES)) * 100,
            'stock_pct': (t_stock / (num_a * settings.TIME_LIMIT_FRAMES)) * 100,
            'crypto_pct': (t_crypto / (num_a * settings.TIME_LIMIT_FRAMES)) * 100,
            'bond_spend_pct': (s_bond / total_spent) * 100,
            'stock_spend_pct': (s_stock / total_spent) * 100,
            'crypto_spend_pct': (s_crypto / total_spent) * 100
        }
        
    global_best_fitness = current_best

def run(config_file):
    # Inicjalizacja NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Upewnij się, że foldery istnieją
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Wyszukiwanie najnowszego checkpointu w folderze
    checkpoints = glob.glob("checkpoints/neat-checkpoint-*")
    
    if settings.LOAD_CHECKPOINT and checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        loaded = False
        for checkpoint in checkpoints:
            try:
                print(f"\n[INFO] Próba wczytania zapisu: {checkpoint}...")
                p = neat.Checkpointer.restore_checkpoint(checkpoint)
                
                # Usuń stare reportery z checkpointu
                p.reporters.reporters.clear()
                
                global global_generation
                global_generation = p.generation
                loaded = True
                break
            except Exception as e:
                print(f"[BŁĄD] Plik zapisu {checkpoint} jest uszkodzony. Error: {e}")
                
        if not loaded:
            print("\n[INFO] Żaden plik zapisu nie działał. Rozpoczynanie nowej ewolucji od zera...")
            p = neat.Population(config)
    else:
        if not settings.LOAD_CHECKPOINT:
            print("\n[INFO] LOAD_CHECKPOINT=False. Rozpoczynanie nowej ewolucji od zera...")
        else:
            print("\n[INFO] Brak zapisów w folderze 'checkpoints/'. Rozpoczynanie od generacji 0...")
        p = neat.Population(config)

    # --- CSV Logger ---
    csv_path = settings.CSV_LOG_FILE

    # Zawsze dodajemy HeadlessReporter, aby mieć logi w konsoli i zapis do CSV!
    p.add_reporter(HeadlessReporter(settings.MAX_GENERATIONS, csv_path))
    
    if not settings.HEADLESS_MODE:
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

    # Automatyczny zapis postępów co 50 generacji do folderu checkpoints
    p.add_reporter(neat.Checkpointer(50, filename_prefix="checkpoints/neat-checkpoint-"))

    # Trening
    remaining_generations = max(1, settings.MAX_GENERATIONS - p.generation)
    winner = p.run(eval_genomes, remaining_generations)
    
    print('\n[INFO] Trening zakończony!')
    print('Najlepszy genom:\n{!s}'.format(winner))
    
    # Zapisz najlepszego agenta do folderu models z datą
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"models/best_agent_{timestamp}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(winner, f)
    print(f"\n[INFO] Najlepszy agent został zapisany jako '{model_filename}'")
    
    # Kopia jako 'latest' dla ułatwienia
    with open("models/best_agent_latest.pkl", "wb") as f:
        pickle.dump(winner, f)
    
    print(f"[INFO] Dane treningowe zapisano do '{csv_path}'")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
