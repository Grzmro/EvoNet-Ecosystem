import pygame
import neat
import os
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
            # Wyświetl napis PAUZA na środku ekranu
            font = pygame.font.SysFont(None, 64)
            text = font.render("PAUZA", True, (255, 100, 100))
            text_rect = text.get_rect(center=(settings.WORLD_WIDTH//2, settings.WINDOW_HEIGHT//2))
            
            # Aby nie migało, nie czyścimy ekranu, tylko nadpisujemy
            # (można to ulepszyć, ale wystarczy jako wskaźnik)
            screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(15)
            continue
            
        if fast_forward:
            # Advance logic without rendering every frame
            sim.run_frame(None, False)
            # Render every 30 frames to keep UI responsive
            if sim.frame_count % 30 == 0:
                sim.run_frame(screen, draw_vision)
                pygame.event.pump() # Zapobiega zawieszaniu się okna
        else:
            sim.run_frame(screen, draw_vision)
            clock.tick(settings.FPS)
            
    # Zapisz najlepszy fitness w bieżącej generacji
    # genomes to lista krotek (genome_id, genome), sprawdzamy genome.fitness
    current_best = -9999.0
    for _, genome in genomes:
        if genome.fitness is not None and genome.fitness > current_best:
            current_best = genome.fitness
    global_best_fitness = current_best

def run(config_file):
    # Inicjalizacja NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    # Dodanie raportowania
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Trening
    winner = p.run(eval_genomes, settings.MAX_GENERATIONS)
    print('\nNajlepszy genom:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
