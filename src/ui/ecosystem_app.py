import pygame

from src.core.settings import (
    COLOR_BG,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_TEXT,
    COLOR_WORLD,
    FPS,
    LEFT_PANEL_WIDTH,
    RIGHT_PANEL_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    RuntimeParams,
)
from src.ui import Button, ReadOnlyValue, Slider
from src.ui.widgets import NumericInput


def format_value(value, fmt, integer):
    if integer:
        return str(int(round(value)))
    return fmt.format(value)


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
