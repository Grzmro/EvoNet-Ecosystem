import pygame

# Okno symulacji
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
FPS = 60

# Kolory
COLOR_BG = (20, 20, 30)
COLOR_TEXT = (200, 200, 200)
COLOR_AGENT = (100, 150, 255)
COLOR_AGENT_BANKRUPT = (50, 50, 50)
COLOR_GURU = (255, 215, 0)
COLOR_WAVE = (255, 215, 0, 100) # żółty, częściowo przezroczysty (będzie rysowany na osobnej powierzchni)

# Agent
STARTING_CAPITAL = 100.0
ENERGY_DRAIN_PER_FRAME = 0.05
AGENT_RADIUS = 5
MAX_SPEED = 5.0
ACCELERATION_FACTOR = 0.5
TURN_RATE = 0.2
SENSOR_RADIUS = 100.0 # Zasięg czujników (np. do wykrywania tłumu)

# Guru i Fale (Asymetria Informacyjna)
WAVE_SPEED = 2.0
WAVE_COOLDOWN = 120 # Liczba klatek między falami

# Strefy Inwestycyjne (Pump & Dump)
ZONE_RADIUS = 60
GROWTH_FACTOR = 0.1 # Przyrost wartości bazowej w każdej klatce za każdego agenta
MAX_CAPACITY = 2000.0 # Maksymalna skumulowana wartość strefy przed obowiązkowym krachem
CRASH_PROBABILITY_BASE = 0.001 # Bazowe prawdopodobieństwo krachu na klatkę (rośnie z czasem bańki)

# NEAT
MAX_GENERATIONS = 300
TIME_LIMIT_FRAMES = 1200 # Czas życia generacji w klatkach (ok. 20 sekund przy 60 FPS)

# Nagrody i Kary (Fitness)
BONUS_SURVIVAL_PER_FRAME = 0.01
PENALTY_BANKRUPTCY = 50.0
