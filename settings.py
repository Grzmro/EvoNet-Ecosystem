import pygame

# Okno symulacji
WORLD_WIDTH = 1000
PANEL_WIDTH = 300
WINDOW_WIDTH = WORLD_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = 800
FPS = 60
HEADLESS_MODE = False
LOAD_CHECKPOINT = True  # RESET wymagany po zmianie architektury

# Kolory
COLOR_BG = (20, 20, 30)
COLOR_PANEL = (30, 30, 40)
COLOR_PANEL_TEXT = (220, 220, 220)
COLOR_TEXT = (200, 200, 200)
COLOR_AGENT = (100, 150, 255)
COLOR_AGENT_BANKRUPT = (50, 50, 50)
COLOR_GURU = (255, 215, 0)
COLOR_WAVE = (255, 215, 0, 100)

# Agent
STARTING_CAPITAL = 100.0
AGENT_RADIUS = 5
MAX_SPEED = 5.0
ACCELERATION_FACTOR = 0.5
TURN_RATE = 0.2
SENSOR_RADIUS = 100.0

# Inflacja (zamiast stałego kosztu życia)
INFLATION_RATE = 0.0003  # 0.03% za klatkę — po 1200 klatkach gotówka traci ~30%

# Guru i Fale (Asymetria Informacyjna)
WAVE_SPEED = 2.0
WAVE_COOLDOWN = 120

# Typy Aktywów
ZONE_TYPE_BOND = 0.0     # Obligacje państwowe (stałe, bezpieczne)
ZONE_TYPE_STOCK = 0.5    # Akcje giełdowe
ZONE_TYPE_CRYPTO = 1.0   # Spekulacyjne kryptowaluty
ZONE_RADIUS = 60

# ==========================================
# Parametry Poissona (bazowe, modyfikowane przez cykl rynkowy)
# ==========================================
BOND_LAMBDA_UP = 0.02
BOND_LAMBDA_DOWN = 0.005
BOND_JUMP_SIZE = 0.01

STOCK_LAMBDA_UP = 0.12
STOCK_LAMBDA_DOWN = 0.10
STOCK_JUMP_SIZE = 0.1

CRYPTO_LAMBDA_UP = 0.20
CRYPTO_LAMBDA_DOWN = 0.21
CRYPTO_JUMP_SIZE = 0.5

# ==========================================
# [NOWE] Wpływ agentów na rynek (bańka spekulacyjna)
# ==========================================
CROWD_PRICE_BOOST = 0.03      # Dodatkowy wzrost ceny za każdego inwestora w strefie (na klatkę)
CROWD_CRASH_THRESHOLD = 8     # Powyżej tylu inwestorów rośnie ryzyko krachu (bańka)
CROWD_CRASH_BASE_PROB = 0.005 # Bazowa szansa krachu przy przeludnieniu (za klatkę)
CRASH_PRICE_DROP = 0.5        # Przy krachu cena spada o 50% (korekta, nie śmierć)

# ==========================================
# [NOWE] Koszty transakcyjne
# ==========================================
TRANSACTION_FEE_RATE = 0.02   # 2% prowizja od każdej transakcji (kupno i sprzedaż)

# ==========================================
# [NOWE] Bonus za dywersyfikację portfela
# ==========================================
DIVERSIFICATION_BONUS = 15.0  # Bonus fitness za każdą dodatkową KLASĘ aktywów w portfelu

# ==========================================
# [NOWE] Cykle rynkowe (Bull / Bear market)
# ==========================================
BEAR_MARKET_START = 600       # Po tej klatce zaczyna się bear market
BULL_LAMBDA_MULTIPLIER = 1.3  # W bull market: lambda_up *= 1.3
BEAR_LAMBDA_MULTIPLIER = 1.5  # W bear market: lambda_down *= 1.5

# NEAT
MAX_GENERATIONS = 1000
TIME_LIMIT_FRAMES = 1200

# Nagrody i Kary (Fitness)
BONUS_SURVIVAL_PER_FRAME = 0.01
PENALTY_BANKRUPTCY = 50.0

# CSV Logging
CSV_LOG_FILE = "training_log.csv"
