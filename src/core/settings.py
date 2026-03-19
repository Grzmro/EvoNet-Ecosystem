from dataclasses import dataclass


WORLD_WIDTH = 800
WORLD_HEIGHT = 600
FPS = 60

LEFT_PANEL_WIDTH = 280
RIGHT_PANEL_WIDTH = 320
WINDOW_WIDTH = LEFT_PANEL_WIDTH + WORLD_WIDTH + RIGHT_PANEL_WIDTH
WINDOW_HEIGHT = WORLD_HEIGHT

FOOD_RADIUS = 4
AGENT_SIZE = 10

COLOR_BG = (238, 240, 244)
COLOR_PANEL = (224, 228, 236)
COLOR_WORLD = (245, 245, 245)
COLOR_GREEN = (40, 170, 40)
COLOR_BLUE = (50, 100, 230)
COLOR_TEXT = (20, 20, 20)
COLOR_MUTED = (90, 95, 110)
COLOR_BTN = (120, 145, 210)
COLOR_BTN_ACTIVE = (84, 118, 210)
COLOR_SLIDER_BG = (200, 205, 215)
COLOR_SLIDER_FILL = (95, 130, 220)
COLOR_BORDER = (160, 166, 180)


@dataclass
class RuntimeParams:
	food_count: int = 35
	max_generation_time: float = 15.0

	max_energy: float = 120.0
	initial_energy: float = 80.0
	base_energy_drain: float = 0.18
	move_energy_drain_factor: float = 0.03
	food_energy_gain: float = 35.0

	survival_fitness_per_frame: float = 0.01
	food_fitness_reward: float = 8.0
	death_fitness_penalty: float = 1.0

	turn_rate: float = 0.16
	acceleration: float = 0.18
	friction: float = 0.92
	max_speed: float = 4.0

	@property
	def eat_distance(self) -> float:
		return AGENT_SIZE + FOOD_RADIUS + 2
