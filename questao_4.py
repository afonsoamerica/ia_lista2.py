from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt

# Criar ontologia em português
onto = get_ontology("http://exemplo.org/ontologia_vinhos_comidas.owl")

with onto:
    # Classes
    class Vinho(Thing): pass
    class Prato(Thing): pass
    class CaracteristicaDoVinho(Thing): pass

    # Subclasses de vinho
    class CabernetSauvignon(Vinho): pass
    class Chardonnay(Vinho): pass
    class Malbec(Vinho): pass

    # Subclasses de prato
    class CarnesGrelhadas(Prato): pass
    class FrutosDoMar(Prato): pass
    class Massas(Prato): pass
    class Queijos(Prato): pass

    # Características
    tânico = CaracteristicaDoVinho("Tânico")
    leve = CaracteristicaDoVinho("Leve")
    frutado = CaracteristicaDoVinho("Frutado")
    encorpado = CaracteristicaDoVinho("Encorpado")

    # Propriedades
    class éAdequadoCom(ObjectProperty):
        domain = [Vinho]
        range = [Prato]

    class temCaracterística(ObjectProperty):
        domain = [Vinho]
        range = [CaracteristicaDoVinho]

    # Instanciando vinhos com características
    cs = CabernetSauvignon("cabernet")
    cs.temCaracterística = [tânico, encorpado]
    cs.éAdequadoCom = [CarnesGrelhadas("carnes")]

    ch = Chardonnay("chardonnay")
    ch.temCaracterística = [leve, frutado]
    ch.éAdequadoCom = [FrutosDoMar("frutos")]

    malb = Malbec("malbec")
    malb.temCaracterística = [tânico, frutado]
    malb.éAdequadoCom = [Queijos("queijos"), Massas("massas")]

# Inferência
sync_reasoner()

# Impressão das harmonizações
print("\n--- Harmonizações ---")
for vinho in onto.Vinho.instances():
    print(f"{vinho.name} harmoniza com:")
    for prato in vinho.éAdequadoCom:
        print(f"  - {prato.name}")

# Plot da ontologia
G = nx.DiGraph()
cores = {"Vinho": "skyblue", "Prato": "lightgreen", "CaracteristicaDoVinho": "orange"}

for vinho in onto.Vinho.subclasses():
    G.add_node(vinho.name, color=cores["Vinho"])
for prato in onto.Prato.subclasses():
    G.add_node(prato.name, color=cores["Prato"])
for carac in onto.CaracteristicaDoVinho.instances():
    G.add_node(carac.name, color=cores["CaracteristicaDoVinho"])

# Ligações entre objetos
for vinho in onto.Vinho.instances():
    for prato in vinho.éAdequadoCom:
        G.add_edge(vinho.__class__.__name__, prato.__class__.__name__, label="éAdequadoCom")
    for carac in vinho.temCaracterística:
        G.add_edge(vinho.__class__.__name__, carac.name, label="temCaracterística")

# Plot
colors = [G.nodes[n]['color'] for n in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2500, font_size=10, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})
plt.title("Ontologia de Vinhos e Pratos (em Português)")
plt.show()
