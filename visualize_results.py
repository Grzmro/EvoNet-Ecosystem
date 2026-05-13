import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os

print("Generowanie statystyk z training_log.csv...")
try:
    df = pd.read_csv('training_log.csv')
    
    plt.figure(figsize=(12, 12))
    
    # 1. Fitness plot
    plt.subplot(3, 1, 1)
    plt.plot(df['generacja'], df['najlepszy_fitness'], label='Najlepszy Fitness', color='green', linewidth=2)
    plt.plot(df['generacja'], df['sredni_fitness'], label='Średni Fitness', color='orange', linewidth=2)
    plt.title('Fitness w czasie ewolucji (Wyniki)')
    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Strategy plot
    plt.subplot(3, 1, 2)
    plt.plot(df['generacja'], df['time_bond_pct'], label='Obligacje (Bond)', color='blue', linewidth=2)
    plt.plot(df['generacja'], df['time_stock_pct'], label='Akcje (Stock)', color='purple', linewidth=2)
    plt.plot(df['generacja'], df['time_crypto_pct'], label='Krypto (Crypto)', color='gold', linewidth=2)
    plt.title('Preferencje inwestycyjne populacji (% czasu spędzonego w aktywach)')
    plt.xlabel('Generacja')
    plt.ylabel('% Czasu')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Species plot
    plt.subplot(3, 1, 3)
    plt.plot(df['generacja'], df['liczba_gatunkow'], label='Liczba gatunków', color='red', linewidth=2)
    plt.plot(df['generacja'], df['avg_trades'], label='Średnia liczba transakcji', color='teal', linewidth=2, linestyle='--')
    plt.title('Różnorodność genetyczna i aktywność na rynku')
    plt.xlabel('Generacja')
    plt.ylabel('Ilość')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_charts.png', dpi=150)
    print("-> Zapisano wykresy statystyk do 'training_charts.png'")
except Exception as e:
    print(f"Błąd przy generowaniu statystyk: {e}")

print("\nGenerowanie struktury sieci neuronowej najlepszego agenta...")
try:
    # Wczytanie najlepszego agenta
    with open('models/best_agent_latest.pkl', 'rb') as f:
        genome = pickle.load(f)
        
    G = nx.DiGraph()
    
    # Definicja warstw i węzłów
    inputs = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
    outputs = [0, 1, 2, 3]
    
    input_names = {
        -9: "Freshness\n(Sygnał)", -8: "Dist\n(Dystans)", -7: "VecX\n(Kierunek X)", -6: "VecY\n(Kierunek Y)",
        -5: "Density\n(Tłum)", -4: "Cash\n(Gotówka)", -3: "ZoneType\n(Typ Strefy)", -2: "Portfolio\n(Aktywa)", -1: "Trend\n(Cena)"
    }
    output_names = {
        0: "Acc\n(Gaz)", 1: "Turn\n(Skręt)", 2: "Decision\n(Kup/Sprzedaj)", 3: "Intensity\n(Budżet)"
    }
    
    # Dodawanie węzłów
    for n in inputs:
        G.add_node(n, layer=0, label=input_names.get(n, str(n)))
        
    for n in outputs:
        G.add_node(n, layer=2, label=output_names.get(n, str(n)))
        
    for cg in genome.connections.values():
        if cg.enabled:
            n1, n2 = cg.key
            if n1 not in G.nodes:
                G.add_node(n1, layer=1, label=f"Hidden {n1}")
            if n2 not in G.nodes:
                G.add_node(n2, layer=1, label=f"Hidden {n2}")
            G.add_edge(n1, n2, weight=cg.weight)
            
    # Pozycjonowanie wierszami/warstwami
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    
    # Rysowanie
    plt.figure(figsize=(14, 10))
    labels = nx.get_node_attributes(G, 'label')
    
    edges = G.edges(data=True)
    # Kolor zielony dla dodatnich wag, czerwony dla ujemnych
    colors = ['green' if d['weight'] > 0 else 'red' for u,v,d in edges]
    # Grubość linii zależna od wagi
    weights = [min(abs(d['weight']), 5.0) for u,v,d in edges]
    
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, 
            node_color='#a0c8f0', font_size=8, font_weight='bold',
            edge_color=colors, width=weights, arrows=True, arrowsize=15)
            
    plt.title('Sieć Neuronowa Najlepszego Agenta (Zielony = +, Czerwony = -)', fontsize=16)
    plt.tight_layout()
    plt.savefig('neural_network.png', dpi=150)
    print("-> Zapisano graf sieci neuronowej do 'neural_network.png'")
    
except Exception as e:
    print(f"Błąd przy generowaniu grafu sieci: {e}")
