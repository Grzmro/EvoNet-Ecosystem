import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os

print("Generowanie statystyk z training_log.csv...")
try:
    df = pd.read_csv('training_log.csv')

    plt.figure(figsize=(13, 20))

    plt.subplot(5, 1, 1)
    plt.plot(df['generacja'], df['najlepszy_fitness'], label='Najlepszy Fitness', color='green', linewidth=2)
    plt.plot(df['generacja'], df['sredni_fitness'], label='Sredni Fitness', color='orange', linewidth=2)
    plt.title('Fitness w czasie ewolucji')
    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 2)
    plt.plot(df['generacja'], df['neat_avg_worth'], label='NEAT (sr. majatek)', color='green', linewidth=2.5)
    plt.plot(df['generacja'], df['bench_hold6040'], label='Hold 60/40', color='steelblue', linewidth=1.8, linestyle='--')
    plt.plot(df['generacja'], df['bench_momentum'], label='Momentum', color='purple', linewidth=1.8, linestyle='--')
    plt.plot(df['generacja'], df['bench_random'], label='Random', color='gray', linewidth=1.5, linestyle=':')
    plt.title('NEAT vs strategie referencyjne (koncowy majatek, ten sam rynek)')
    plt.xlabel('Generacja')
    plt.ylabel('Sredni koncowy majatek ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 3)
    plt.plot(df['generacja'], df['avg_fitness_informed'], label='Z info edge (Guru)', color='gold', linewidth=2)
    plt.plot(df['generacja'], df['avg_fitness_uninformed'], label='Bez info edge', color='steelblue', linewidth=2)
    plt.fill_between(df['generacja'], df['avg_fitness_informed'], df['avg_fitness_uninformed'],
                     alpha=0.15, color='gold')
    plt.title('Asymetria informacyjna: Informed vs Uninformed')
    plt.xlabel('Generacja')
    plt.ylabel('Sredni Fitness grupy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 4)
    plt.stackplot(
        df['generacja'],
        df['pct_bond'], df['pct_stock'], df['pct_crypto'], df['pct_cash'],
        labels=['Bond', 'Stock', 'Crypto', 'Cash'],
        colors=['#5078ff', '#ff9030', '#b432ff', '#888888'],
        alpha=0.8,
    )
    plt.title('Wyewoluowana strategia: sredni sklad portfela populacji')
    plt.xlabel('Generacja')
    plt.ylabel('% kapitalu')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 5)
    plt.plot(df['generacja'], df['liczba_gatunkow'], label='Liczba gatunkow', color='red', linewidth=2)
    plt.plot(df['generacja'], df['avg_trades'], label='Sr. liczba transakcji', color='teal', linewidth=1.8, linestyle='--')
    plt.title('Roznorodnosc genetyczna i aktywnosc handlowa')
    plt.xlabel('Generacja')
    plt.ylabel('Ilosc')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_charts.png', dpi=150)
    print("-> Zapisano wykresy do 'training_charts.png'")
except Exception as e:
    print(f"Blad przy generowaniu statystyk: {e}")

print("\nGenerowanie struktury sieci neuronowej najlepszego agenta...")
try:
    with open('models/best_agent_latest.pkl', 'rb') as f:
        genome = pickle.load(f)

    G = nx.DiGraph()

    num_inputs = 19
    num_outputs = 6
    inputs = list(range(-num_inputs, 0))
    outputs = list(range(num_outputs))

    input_names = {
        -19: "Price[bond]", -18: "Price[s1]", -17: "Price[s2]", -16: "Price[s3]", -15: "Price[crypto]",
        -14: "Trend[bond]", -13: "Trend[s1]", -12: "Trend[s2]", -11: "Trend[s3]", -10: "Trend[crypto]",
        -9: "W[bond]", -8: "W[s1]", -7: "W[s2]", -6: "W[s3]", -5: "W[crypto]",
        -4: "Cash%", -3: "Guru fresh", -2: "Guru asset", -1: "Guru dir",
    }
    output_names = {
        0: "Alloc[bond]", 1: "Alloc[s1]", 2: "Alloc[s2]", 3: "Alloc[s3]", 4: "Alloc[crypto]", 5: "Cash",
    }

    for n in inputs:
        G.add_node(n, layer=0, label=input_names.get(n, str(n)))
    for n in outputs:
        G.add_node(n, layer=2, label=output_names.get(n, str(n)))

    for cg in genome.connections.values():
        if cg.enabled:
            n1, n2 = cg.key
            if n1 not in G.nodes:
                G.add_node(n1, layer=1, label=f"H{n1}")
            if n2 not in G.nodes:
                G.add_node(n2, layer=1, label=f"H{n2}")
            G.add_edge(n1, n2, weight=cg.weight)

    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

    plt.figure(figsize=(16, 12))
    labels = nx.get_node_attributes(G, 'label')
    edge_data = list(G.edges(data=True))
    edge_colors = ['green' if d['weight'] > 0 else 'red' for _, _, d in edge_data]
    edge_widths = [min(abs(d['weight']), 4.0) for _, _, d in edge_data]

    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2500,
            node_color='#a0c8f0', font_size=7, font_weight='bold',
            edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=12)

    plt.title('Siec Neuronowa Najlepszego Agenta  |  Zielony = waga+  /  Czerwony = waga-', fontsize=14)
    plt.tight_layout()
    plt.savefig('neural_network.png', dpi=150)
    print("-> Zapisano graf sieci do 'neural_network.png'")

except Exception as e:
    print(f"Blad przy generowaniu grafu: {e}")
