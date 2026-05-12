import pygame
import neat
import os
import settings
from simulation import Ecosystem

# Ustawienia Pygame
pygame.init()
pygame.display.set_caption("Ewolucja agentów w warunkach asymetrii informacyjnej")
font = pygame.font.SysFont(None, 24)

def eval_genomes(genomes, config):
    # Tworzymy symulację
    sim = Ecosystem(genomes, config)
    
    # Tworzymy okno, jeśli chcemy renderować
    screen = pygame.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running and sim.frame_count < settings.TIME_LIMIT_FRAMES and not sim.all_dead():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
        sim.run_frame(screen)
        clock.tick(settings.FPS)

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
