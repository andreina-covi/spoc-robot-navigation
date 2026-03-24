import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def draw_default_graph():
    # Create graph
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(["A", "B", "C", "D"])
    # Add edges
    G.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D")
    ])
    # Draw
    nx.draw(
        G,
        with_labels=True,
        node_size=2000,
        font_size=12
    )
    plt.show()

def get_edge_ids(edges_dict, node_ids):
    edge_ids = []
    for edge in edges_dict:
        source = edge.get("source")
        target = edge.get("target")
        if source is not None and target is not None:
            edge_ids.append((node_ids[source], node_ids[target], {"relation": edge.get("relation", "")}))
    return edge_ids

def draw_graph(graph_dict):
    # print("Graph dict: ", graph_dict)
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    for timestep in range(len(nodes)):
        node_ids = list(nodes[timestep].keys())
        node_ids = {value:i for i, value in enumerate(node_ids)}
        # print("Node IDs: ", node_ids)
        edge_ids = get_edge_ids(edges[timestep], node_ids)
        # print("Edges dict: ", edges[timestep])
        # print("Edge IDs: ", edge_ids)
        G = nx.DiGraph()
        G.add_nodes_from(node_ids.values())
        G.add_edges_from(edge_ids)
        print(f"Nodes at timestep {timestep}: {node_ids}")
        print(f"Graph at timestep {timestep}: {G.edges(data=True)}")
        nx.draw(
            G,
            with_labels=True,
            node_size=2000,
            font_size=12
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=nx.spring_layout(G),
            edge_labels={(u, v): d["relation"] for u, v, d in G.edges(data=True)},
            font_color='red'
        )
        plt.show()
        # net = Network(height="600px", width="100%")
        # net.from_nx(G)
        # net.show("graph.html", notebook=False)
        break

if __name__ == "__main__":
    draw_default_graph()

