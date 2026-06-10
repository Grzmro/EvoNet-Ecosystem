# EvoNet-Ecosystem

**Ewolucja sieci neuronowych sterujących zachowaniem agentów w ekosystemie z wykorzystaniem algorytmu NEAT.**

Symulacja wieloagentowa, w której sterowniki w postaci sieci neuronowych są ewoluowane
algorytmem **NEAT** (NeuroEvolution of Augmenting Topologies) w celu rozwiązania
sekwencyjnego problemu **alokacji portfela** na niestacjonarnym rynku finansowym.

**Pytanie badawcze:** czy agenci z dostępem do sygnału informacyjnego (*Guru*) ewoluują
inne, lepsze strategie niż agenci bez niego?

Każdy agent to portfolio manager — dostaje 19 wejść opisujących stan rynku i portfela,
a zwraca 6 wyjść (alokacja na 5 aktywów + gotówka przez softmax). Rynek oparty jest na
złożonym procesie Poissona ze skokami (model zbliżony do skokowo-dyfuzyjnego Mertona),
z bańkami spekulacyjnymi napędzanymi tłumem, krachami oraz zmianą reżimu hossa/bessa.

## Struktura projektu

| Plik / katalog | Opis |
|---|---|
| `train.py` | Sterownik treningu NEAT (logowanie, checkpointy, serializacja) |
| `simulation.py` | Klasa `Ecosystem` — pętla epizodu, równoległy rynek benchmarkowy, dashboard GUI |
| `agent.py` | Agent portfolio-manager (obserwacja, alokacja softmax, fitness) |
| `entities.py` | `Guru` (fale informacyjne) i `InvestmentZone` (proces cen) |
| `benchmarks.py` | Strategie referencyjne: Hold 60/40, Momentum, Random |
| `settings.py` | Scentralizowana konfiguracja |
| `visualize_results.py` | Generuje wykresy wyników i graf topologii sieci |
| `replay.py` | Odtwarza najlepszego agenta w interaktywnym dashboardzie |
| `config-feedforward.txt` | Hiperparametry NEAT |
| `requirements.txt` | Zależności Pythona |
| `report/` | Sprawozdanie (PDF + źródło LaTeX + wykresy) |
| `models/` | Najlepsze wyewoluowane genomy (`best_agent_latest.pkl`) |
| `checkpoints/` | Checkpointy populacji NEAT |
| `training_log.csv` | Statystyki per generacja |

## Instalacja

```bash
pip install -r requirements.txt
```
Python 3.12 — `pygame`, `neat-python` (rdzeń) oraz `pandas`, `matplotlib`, `networkx`
(wizualizacja wyników).

## Uruchomienie

**Trening** (headless, domyślnie):
```bash
python train.py
```
Parametry w `settings.py`: `MAX_GENERATIONS` (domyślnie 1000), `HEADLESS_MODE`,
`LOAD_CHECKPOINT`. Statystyki trafiają do `training_log.csv`, najlepszy genom do
`models/best_agent_latest.pkl`, a checkpointy co 50 generacji do `checkpoints/`.

GUI: ustaw `HEADLESS_MODE = False` w `settings.py`. Sterowanie: **SPACJA** (pauza),
**F** (fast forward), **S** (skip generacji).

**Wykresy** po treningu:
```bash
python visualize_results.py     # -> training_charts.png, neural_network.png
```

**Podgląd** najlepszego agenta w dashboardzie tradingowym:
```bash
python replay.py
```

## Reprodukowalność

Każda generacja `g` jest ziarnowana wartością `BASE_SEED + g`, więc zarówno przebieg NEAT,
jak i równoległy rynek benchmarkowy są w pełni odtwarzalne. Po zmianie
`num_inputs`/`num_outputs` w `config-feedforward.txt` ustaw `LOAD_CHECKPOINT = False`
(stare checkpointy są niekompatybilne).

## Sprawozdanie

Pełny raport (artykuł, sekcje 1–4 + Abstrakt i Słowa kluczowe) znajduje się w
[`report/report.pdf`](report/report.pdf); źródło LaTeX w `report/report.tex`.
Gotowe archiwum do oddania: `EvoNet-Ecosystem-submission.zip`.
