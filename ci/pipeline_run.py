import networkx as nx


done = set({})
name2process = {}
name2cmd = {}


def add_pipelie(graph, tasks):
    for (t1, t2) in zip(tasks[0:-1], tasks[1::]):
        print(t1, t2)
        graph.add_edge(t1["name"], t2["name"])


G = nx.Graph()
add_pipelie(G, [{"name": "cpu build"}, {"name": "xla build"}, {"name": "gpu build"}])
for n in G:
    print(n)
