import pygame

from src.core.settings import FPS, RENDER_EVERY_N_GENS, WORLD_HEIGHT, WORLD_WIDTH, RuntimeParams, COLOR_TEXT
from src.simulation.simulation import Simulation


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
        from src.core.settings import MAX_TRAINING_GENERATIONS

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
