import networkx as nx
import torch
import random

"""
Implementation of the HEAT-RAY Optimization based mitigation selection

    (1) The original implementation logic is:

        - We use a particular notation of Hajiaghayi-Räcke sparsest cut formulation
        - Start by initialising u(e) uniform
        - Apply non-linear transformation over edge u(e) values for d(e)
        - Compute gamma using shortest paths with d(e) as weights using Dijkstra’s
        - Apply perturbation factor of 0.95/1.05 to weights to avoid degeneracies
        - Compute beta = sum_e c(e) * d(e)
        - Gradient descent is used to refine the edge distances iteratively
        - Output the demand for each edge. If: d(e) > 0.5 := cut-set, else: continue
        - Params : max_it=1, max_tree=1000, max_depth=1000, lr=1

    (2) The following differ from the original:

        - Use ADAM optimiser instead of SGD
        - Remove tree and depth upper bounds
        - Move max_it=100, lr=0.01
        - Use Torch
        - Instead of SVM learned cost, we use heuristics (#parallel edges)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")


def d_from_u_tensor(u_tensor):
    """
    Eq (5): Apply non-linear transformation vectorized over edge u values
    """
    return torch.where(u_tensor > 1, u_tensor, torch.exp(u_tensor - 1))

def compute_beta(G, u_dict, edge_idx_map):
    """
    Eq (7): Compute beta = sum_e c(e) * d(e)
    """
    u_tensor = torch.stack([u_dict[e] for e in G.edges()])
    d_tensor = d_from_u_tensor(u_tensor)
    c_tensor = torch.tensor([G.edges[u, v, key].get('cost', 1.0) for u, v, key in G.edges(keys=True)], dtype=torch.float32, device=device)
    return torch.sum(d_tensor * c_tensor)

def compute_gamma(G, u_dict, edge_idx_map, sampled_pairs):
    """
    Eq (8): Compute gamma using shortest paths with d(e) as weights 
    """
    u_tensor = torch.stack([u_dict[e] for e in G.edges()])
    d_tensor = d_from_u_tensor(u_tensor).detach().cpu().numpy()

    """
    Shortest Path Contribution to Denominator (Benefit b(e))
    """
    edge_weights = {(u, v, key): d_tensor[i] for i, (u, v, key) in enumerate(G.edges(keys=True))}
    nx.set_edge_attributes(G, edge_weights, "weight")

    """
    Perturb the edge weights by applying a random multiplicative factor between 0.95 and 1.05.
    """
    for u, v, key, data in G.edges(data=True, keys=True):
        perturb_factor = random.uniform(0.95, 1.05)
        data['weight'] = data.get('weight', 1.0) * perturb_factor

    """
    For each sampled pair (si, ti), the algorithm conducts a bounded-horizon search 
    using Dijkstra’s algorithm to compute the shortest path distance x(si, ti).
    """
    total = 0.0
    for s, t in sampled_pairs:
        try:
            total += nx.shortest_path_length(G, s, t, weight="weight")
        except nx.NetworkXNoPath:
            total += 1e6 
    return torch.tensor(total, dtype=torch.float32)


def sparsest_cut_torch(G, lr=10e-5, num_iter=10, sample_size=10):
    edges = list(G.edges())
    edge_idx_map = {e: i for i, e in enumerate(edges)}

    """
    Heat-ray begins with an initial uniform assignment of edge distances
    We directly apply the perturbation here since we do not need the SVM part from the paper.
    """
    u_dict = {e: torch.nn.Parameter(torch.tensor(random.uniform(0.475, 0.525), dtype=torch.float32, device=device)) for e in edges}

    optimizer = torch.optim.Adam(u_dict.values(), lr=lr)
    nodes = list(G.nodes())
    loss_history = []

    for it in range(num_iter):
        optimizer.zero_grad()

        """
        The algorithm samples a small set of node pairs and estimates the gradient from them. 
        This helps avoid computing gradients for all pairs, which can be computationally expensive.
        """
        sampled_pairs = random.sample([(u, v) for u in nodes for v in nodes if u != v], sample_size)

        beta = compute_beta(G, u_dict, edge_idx_map)
        gamma_val = compute_gamma(G, u_dict, edge_idx_map, sampled_pairs)

        loss = torch.log(beta / gamma_val)
        loss.backward()

        """
        Gradient descent is used to refine the edge distances iteratively (ADAM)
        """
        optimizer.step()

    final_u = {e: u.detach().cpu().item() for e, u in u_dict.items()}
    final_d = {e: d_from_u_tensor(torch.tensor(final_u[e])).item() for e in edges}
    return final_d, loss_history

if __name__ == "__main__":
    
    """
    Sample code to generate a mock directed multigraph
    """
    n = 1000  
    p = 0.1  
    G = nx.gnp_random_graph(n, p, directed=True)
    G_multigraph = nx.MultiDiGraph()
    for u, v in G.edges():
        for i in range(random.randint(1, 3)):  
            key = random.randint(0, 10000)  
            G_multigraph.add_edge(u, v, key=key, cost=1.0)

    """
    Cost : In the original paper, cost is learned by SVM from operator feedback
    We compute cost using heuristics instead (cost = #parallele edges)
    """
    for u, v, key, data in G_multigraph.edges(data=True, keys=True):
        parallel_edges_count = len(G_multigraph.get_edge_data(u, v))
        data['cost'] = 1-(1/parallel_edges_count)

    """
    Contribution : Use ADAM instead of SGD
    """
    d_result, loss_adam = sparsest_cut_torch(G_multigraph)
    
    """
    Return the set
    """
    cut_set = [(e, d) for e, d in d_result.items() if d >= 0.5]
    non_set = [(e, d) for e, d in d_result.items() if d < 0.5]

    """
    Filter the cut set based on the condition: cost > demand
    """
    selected_edges = []
    for e, d in cut_set:
        if len(e) == 3: # ugly fix but I'm late
            u, v, key = e
            cost = G_multigraph[u][v][key]['cost'] 
        else:
            u, v = e
            key = list(G_multigraph[u][v].keys())[0]  
            cost = G_multigraph[u][v][key]['cost']
        if cost > d:
            selected_edges.append((e,cost,d))

    """
    Return suggested mitigations
    """
    print(f"\n---\n[Selected edges based on cost > demand condition (= {len(selected_edges)}/{len(G_multigraph.edges)})]\n---\n")
    for (e,cost,d) in selected_edges[:10]:
        print(f"Edge {e}\twith cost > demand \t({cost:.2f} > {d:.2f})")