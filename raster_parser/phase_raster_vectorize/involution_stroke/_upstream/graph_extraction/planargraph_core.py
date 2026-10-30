# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/graph_extraction/planargraph_core.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import math
from collections import Counter

import logging as _logging
_LOG = _logging.getLogger("raster_parser.involution_stroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))




def add_subgraph_with_pos(
    G: nx.Graph,
    pos: np.ndarray,
    edges: list[tuple[int, int]],
    G_to_new_edges: None | list = None,
    **kwargs,
) -> None:
    """
    Add a subgraph to G where each row in `pos` becomes a new node with a
    'pos' attribute, and `edges` connect those new nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph to extend.
    pos : np.ndarray
        Array of shape (n, d), where each row is a node position.
    edges : list of (int, int)
        Edge list referring to indices in `pos`.
    """
    # Offset for new node indices
    offset = max(G.nodes)+1 if G.nodes else 0

    # Add nodes with position attributes
    for i, p in enumerate(pos):
        G.add_node(offset + i, pos=tuple(p), **kwargs)

    # Add edges with offset indices
    for u, v in edges:
        assert G.has_node(offset + u)
        assert G.has_node(offset + v)
        G.add_edge(offset + u, offset + v)
    if G_to_new_edges is not None:
        for u, v in G_to_new_edges:
            assert offset + v in G.nodes
            G.add_edge(u, offset+v)
        

    for i in G.nodes:
        assert G.nodes[i]["pos"] is not None

def add_path_subgraph(G, points: np.ndarray,  G_to_new_edges: None | list[tuple[int, int]] = None):
    l = list(
        zip(
            range(len(points)), range(1, len(points))
            )
        )
    return add_subgraph_with_pos(G, points, l, G_to_new_edges)

def add_subgraph_to_G(G: nx.Graph, H: nx.Graph, G_to_H_edges = None):
    """
    Adds subG to G in-place. 
    Relabels subG nodes to avoid collisions with existing nodes in G.
    Preserves all node and edge attributes.
    """
    # 1. Determine the starting ID for the new nodes
    # If G is empty, start at 0. Otherwise, start at max(nodes) + 1.
    start_id = max(G.nodes) + 1 if len(G) > 0 else 0

    # 2. Create a mapping from old IDs to new IDs
    # This maintains the relative structure of subG
    mapping = {node: i + start_id for i, node in enumerate(H.nodes())}

    # 3. Create a temporary relabeled version of the subgraph
    # copy=True ensures we don't modify the original subG object
    new_sub_part = nx.relabel_nodes(H, mapping, copy=True)

    # 4. Merge into G in-place
    # update() adds nodes, edges, and all associated attributes
    G.update(new_sub_part)

    if G_to_H_edges:
        for u, v in G_to_H_edges:
            assert u in G
            assert mapping[v] in G
            G.add_edge(u, mapping[v])

def remove_path(G: nx.Graph, path):
    if len(path) == 2:
        G.remove_edge(path[0], path[-1])
    if len(path) >2:
        nodes_to_remove = path[1:-1]
        G.remove_nodes_from(nodes_to_remove)

def remove_shortest_path(G: nx.Graph, source, target):
    path = nx.shortest_path(G, source=source, target=target)
    if len(path) == 2:
        G.remove_edge(path[0], path[-1])
    if len(path) >2:
        nodes_to_remove = path[1:-1]
        G.remove_nodes_from(nodes_to_remove)


def has_degree1_path(G: nx.Graph, u, nxt):
    visited = [u]
    current = nxt
    prev = u
    it =0
    while it < 10_000_000:
        it+=1
        visited.append(current)
        # print("current", current, G.nodes[current]["pos"])
        deg = G.degree(current)
        if deg == 1:
            # Found another leaf -> valid path
            return visited
        elif deg == 2:
            # Move to the next neighbor that we haven’t visited
            next_nodes = [n for n in G.neighbors(current) if n != prev]
            if not next_nodes:
                return None
            prev = current
            current = next_nodes[0]
            if current == u:
                return None

        else:
            # Node with degree != 1,2 blocks the path
            return None

    raise ValueError("long iteration, possibly infinite loop")


def find_degree3_pairs_within_distance(
    G: nx.Graph,
    d: int,
) -> list[tuple[int, int, int]]:
    """
    Find pairs of degree-3 nodes whose shortest-path distance is <= d.
    """
    deg3_nodes = {n for n, deg in G.degree() if deg == 3}
    pairs = []

    for u in deg3_nodes:
        # Shortest path lengths from u up to distance d
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=d)
        for v, d in lengths.items():
            if v in deg3_nodes and v > u:
                pairs.append((u, v, d))

    return pairs


def collapse_nodes(G: nx.Graph, nodes, **node_attrs):
    """
    Collapse a set of nodes into a single node in-place.

    Parameters
    ----------
    G : networkx.Graph (or DiGraph / MultiGraph / MultiDiGraph)
        Graph to modify in-place.
    nodes : iterable
        Nodes to collapse.
    new_node : hashable
        Label of the new collapsed node.

    Notes
    -----
    - Internal edges are removed
    - External edges are preserved
    - No self-loops are created
    """
    nodes = set(nodes)

    new_node = max(G.nodes)+1
    G.add_node(new_node, **node_attrs)

    for u in nodes:
        for v in list(G.neighbors(u)):
            if v not in nodes:
                G.add_edge(new_node, v)

    # Remove old nodes
    G.remove_nodes_from(nodes)


def find_disjoint_paths_to_set(G, source, target_set):
    # 1. Create a copy so we don't modify the original graph
    H = G.copy()

    # 2. Add the Super-Sink
    super_sink = "_SINK_"
    for target in target_set:
        H.add_edge(target, super_sink)

    # 3. Compute node-disjoint paths
    # This uses a flow-based algorithm internally
    try:
        paths = list(nx.node_disjoint_paths(H, source, super_sink))

        # 4. Clean up: Remove the Super-Sink from the resulting paths
        # Each path will look like [source, ..., target, "_SINK_"]
        cleaned_paths = [p[:-1] for p in paths]
        return cleaned_paths
    except (nx.NetworkXNoPath, nx.NetworkXError):
        return []


def find_weighted_disjoint_paths_to_set(G, source, target_set, weight_func):
    aux = nx.DiGraph()
    for node in G.nodes():
        aux.add_edge(f"{node}_in", f"{node}_out", capacity=1, weight=0)
    for u, v in G.edges():
        w = weight_func(u, v)
        aux.add_edge(f"{u}_out", f"{v}_in", capacity=1, weight=w)
        aux.add_edge(f"{v}_out", f"{u}_in", capacity=1, weight=w)

    super_sink = "_SINK_"
    for t in target_set:
        if t in G:
            aux.add_edge(f"{t}_out", super_sink, capacity=1, weight=0)

    source_node = f"{source}_out"

    # FIX: Use max_flow_min_cost directly
    try:

        flow_dict = nx.max_flow_min_cost(aux, source_node, super_sink)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return []
    print("done")
    # Reconstruct paths (same as before, but checking for flow > 0)
    paths = []
    # To find out how many paths we have, check flow out of source
    total_flow = sum(flow_dict[source_node].values())

    for _ in range(int(total_flow)):
        path = [source]
        curr = source_node
        while curr != super_sink:
            found_next = False
            for next_node, flow in flow_dict[curr].items():
                if flow > 0:
                    flow_dict[curr][next_node] -= 1
                    if "_in" in next_node:
                        node_name = next_node.split("_in")[0]
                        if node_name != str(path[-1]):
                            path.append(node_name)
                    curr = next_node
                    found_next = True
                    break
            if not found_next: break
        paths.append(path)

    return paths


def solve_multi_source_greedy(G, source_to_targets, weight_func=None):
    """
    Greedily finds disjoint paths for multiple source-target groups.

    source_to_targets: Dict {s1: [t1, t2...], s2: [t4, t5...]}
    """
    # 1. Sort sources by the number of targets descending (Heuristic)
    sorted_sources = sorted(source_to_targets.items(), key=lambda x: len(x[1]), reverse=True)

    working_graph = G.copy()
    final_results = {}

    for source, targets in sorted_sources:
        # 2. Find paths for the current source in the remaining graph
        #print("getting paths for source", source)
        if weight_func is None:
            paths = find_disjoint_paths_to_set(working_graph, source, targets)
        else:
            paths = find_weighted_disjoint_paths_to_set(working_graph, source, targets, weight_func)


        if paths:
            final_results[source] = paths

            # 3. Remove used nodes from the working graph to maintain vertex-disjointness
            for path in paths:
                nodes_to_remove = path[:-1]
                working_graph.remove_nodes_from(nodes_to_remove)
        else:
            final_results[source] = []

    return final_results


def remove_all_except_paths(G: nx.Graph, V_set: set[int], paths: list[list[int]]):
    """
    Removes all nodes and edges in G that are in V but not present in the given paths.
    Modifies G in-place.
    """
    # 1. Extract all nodes and edges present in the provided paths
    nodes_in_paths = set()
    edges_in_paths = set()

    for path in paths:
        for i in range(len(path)):
            nodes_in_paths.add(path[i])
            if i < len(path) - 1:
                # Use a sorted tuple to ensure undirected edge consistency (u, v) == (v, u)
                u, v = path[i], path[i+1]
                edges_in_paths.add(tuple(sorted((u, v))))


    # 2. Identify and remove nodes
    # Logic: If node is in V and NOT in the paths, remove it.
    nodes_to_remove = [n for n in V_set if n in G and n not in nodes_in_paths]
    G.remove_nodes_from(nodes_to_remove)

    # 3. Identify and remove edges
    # Logic: If an edge involves a node in V and is NOT in the paths, remove it.
    # We iterate over a list of current edges because we cannot modify the graph while iterating.
    edges_to_remove = []
    for u, v in G.edges():
        if u in V_set and v in V_set:
            if tuple(sorted((u, v))) not in edges_in_paths:
                edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)


def get_neighbors_of_neighbors(G: nx.Graph, v: int) -> list[int]:
    # This returns a dictionary: {node: distance}
    # cutoff=2 prevents searching the whole graph
    lengths = nx.single_source_shortest_path_length(G, v, cutoff=2)

    # Filter for nodes where distance is exactly 2
    return [node for node, dist in lengths.items() if dist == 2]



def shortest_average_path_restricted(G, source, target, nodes, weight):
    """
    Computes the shortest path from source to target restricted to a specific node set,
    minimizing the average cost per edge.
    
    Uses Layered Dynamic Programming (Bellman-Ford style) to run in O(|V|*|E|) time.

    Parameters:
    - G: NetworkX graph
    - source: Starting node
    - target: Ending node
    - nodes: Iterable of allowed nodes
    - weight: Function (u, v, d) -> number. Returns None if edge is impassable.

    Returns:
    - List of nodes [source, ..., target]
    """
    
    # 1. Validation and Setup
    nodes_set = set(nodes)
    if source not in nodes_set or target not in nodes_set:
        raise ValueError("Source and Target must be in the restricted node set.")

    # Create a view restricted to the node set
    H = G.subgraph(nodes_set)
    n = len(nodes_set)
    
    # dp[k][u] = min cost to reach node u with exactly k edges
    # parents[k][u] = the predecessor node to allow path reconstruction
    dp = [{} for _ in range(n)] 
    parents = [{} for _ in range(n)]
    
    # Base case: 0 cost to reach source with 0 edges
    dp[0][source] = 0
    
    # 2. Dynamic Programming (Layered iteration)
    # We iterate k from 1 to n-1. A simple path can have at most n-1 edges.
    for k in range(1, n):
        prev_nodes = dp[k-1]
        
        # If no nodes were reachable in the previous step, we are done
        if not prev_nodes:
            break
            
        # Iterate over all nodes reached in the previous step (k-1)
        for u, cost_u in prev_nodes.items():
            
            # CONSTRAINT: Do not overshoot. 
            # If we reached the target at step k-1, we do not extend from it.
            if u == target:
                continue

            # Check all neighbors v of u
            for v in H[u]:
                # Calculate weight
                w = weight(u, v, H[u][v])
                
                # If weight is None, edge is effectively blocked
                if w is None:
                    continue
                
                new_cost = cost_u + w
                
                # If v is not reached at step k yet, or we found a cheaper way for step k
                if v not in dp[k] or new_cost < dp[k][v]:
                    dp[k][v] = new_cost
                    parents[k][v] = u

    # 3. Find the Global Minimum Average
    best_avg = float('inf')
    best_k = -1
    found = False

    # Check the cost at the target node for every possible path length k
    for k in range(1, n):
        if target in dp[k]:
            total_cost = dp[k][target]
            avg = total_cost / k
            
            if avg < best_avg:
                best_avg = avg
                best_k = k
                found = True

    if not found:
        raise nx.NetworkXNoPath(f"No path found between {source} and {target}.")

    # 4. Reconstruct Path
    path = [target]
    curr = target
    k = best_k
    
    while k > 0:
        parent = parents[k][curr]
        path.append(parent)
        curr = parent
        k -= 1
        
    path.reverse()
    return path




def shortest_minimax_path_restricted(G: nx.Graph, source, target, nodes, weight):
    """
    Computes the path from source to target restricted to a node set V,
    minimizing the maximum edge weight (Bottleneck Path).
    
    Uses Binary Search combined with NetworkX's built-in shortest_path.
    
    Parameters:
    - G: NetworkX graph
    - source: Starting node
    - target: Ending node
    - nodes: Iterable of allowed nodes
    - weight: Function (u, v, d) -> number. Returns None if impassable.

    Returns:
    - List of nodes [source, ..., target]
    """

    # 1. Validation
    nodes_set = set(nodes)
    if source not in nodes_set or target not in nodes_set:
        raise ValueError("Source and Target must be in the restricted node set.")
    
    # Create a view of the graph restricted to the node set
    # (This is virtually free, no data copying)
    H = G.subgraph(nodes_set)
    
    if source == target:
        return [source]

    # 2. Precompute weights
    # We store a list of ((u, v), w) tuples to efficiently filter later.
    weighted_edges = []
    unique_weights = set()

    for u, v, d in H.edges(data=True):
        w = weight(u, v, d)
        if w is not None:
            weighted_edges.append(((u, v), w))
            unique_weights.add(w)

    if not unique_weights:
        raise nx.NetworkXNoPath("No valid edges in the restricted subgraph.")

    # Sort weights for Binary Search
    sorted_weights = sorted(list(unique_weights))

    # 3. Binary Search
    low = 0
    high = len(sorted_weights) - 1
    best_path = None
    
    while low <= high:
        mid_index = (low + high) // 2
        threshold = sorted_weights[mid_index]
        
        # Filter: Identify edges in H that satisfy the constraint <= threshold
        valid_edges = [edge for edge, w in weighted_edges if w <= threshold]
        
        # Create a temporary view restricted to these valid edges.
        # H.edge_subgraph(...) returns a view, it does NOT copy the graph structure,
        # making this operation very efficient.
        H_filtered = H.edge_subgraph(valid_edges)
        try:
            # Use built-in unweighted shortest_path (BFS)
            # If multiple paths satisfy the threshold, this picks the one with fewest hops.
            if not source in H_filtered.nodes or not target in H_filtered.nodes:
                raise nx.NetworkXNoPath
            path = nx.shortest_path(H_filtered, source=source, target=target)
            
            # If successful, store path and try to find a tighter bottleneck (lower weight)
            best_path = path
            high = mid_index - 1
            
        except nx.NetworkXNoPath:
            # If not reachable, we need to increase the weight limit
            low = mid_index + 1

    if best_path is None:
        raise nx.NetworkXNoPath(f"No path found between {source} and {target}.")

    return best_path


def decompose_into_paths(G: nx.Graph):
    paths = []
    visited_edges = set()

    def edge_key(u, v):
        return tuple(sorted((u, v)))

    endpoints = [n for n in G.nodes if G.degree(n) != 2]

    # Walk from all endpoints
    for start in endpoints:
        for nbr in G.neighbors(start):
            if edge_key(start, nbr) in visited_edges:
                continue

            path = [start, nbr]
            visited_edges.add(edge_key(start, nbr))

            prev, curr = start, nbr
            while G.degree(curr) == 2:
                next_node = next(n for n in G.neighbors(curr) if n != prev)
                if edge_key(curr, next_node) in visited_edges:
                    break
                path.append(next_node)
                visited_edges.add(edge_key(curr, next_node))
                prev, curr = curr, next_node

            paths.append(path)

    # Handle pure cycles (all degree 2)
    for u, v in G.edges:
        if edge_key(u, v) not in visited_edges:
            path = [u, v]
            visited_edges.add(edge_key(u, v))
            prev, curr = u, v

            while curr != u:
                next_node = next(n for n in G.neighbors(curr) if n != prev)
                path.append(next_node)
                visited_edges.add(edge_key(curr, next_node))
                prev, curr = curr, next_node

            paths.append(path)

    return paths


def add_point(G: nx.Graph, point, connect_to):
    offset = max(G.nodes)+1
    G.add_node(offset, pos = tuple(point))
    for i in connect_to:
        assert i in G.nodes
        G.add_edge(offset, i)


def add_vectors_to_graph_debug(G: nx.Graph, p, vecs, scale=10):
    assert p.shape == vecs.shape

    nw = max(G.nodes)+1
    for i in range(0, len(vecs)):
        vi = p[i] + scale * vecs[i]
        G.add_node(i+ nw, pos = tuple(p[i]))
        G.add_node(i + len(vecs) + nw, pos = tuple(vi))
        G.add_edge(i+ nw,i + len(vecs) + nw)



def optimize_disjoint_paths(G, paths, weight):
    """
    Greedily optimizes a list of disjoint paths (where endpoints may be shared)
    by attempting to find a shortest path for each, respecting the constraints
    imposed by other paths.
    
    Args:
        G (nx.Graph): The networkx graph.
        paths (list of lists): A list of paths (lists of node IDs). 
                               This list is modified in-place.
        weight (str or callable): The edge weight attribute or function 
                                  to minimize. Default is 'weight'.
    """
    # 1. Track global node usage to ensure disjointness
    # We use a Counter because endpoints can be shared, so a node 
    # might have a count > 1 (e.g., if it's a start for one path and end for another).
    node_usage = Counter()
    for path in paths:
        node_usage.update(path)

    # 2. Iterate through each path to optimize it
    for i in range(len(paths)):
        old_path = paths[i]
        source = old_path[0]
        target = old_path[-1]

        # Temporarily remove the current path's contribution to usage
        # We handle this manually to ensure we don't delete keys purely for safety,
        # though Counter allows 0 values.
        node_usage.subtract(old_path)
        
        # 3. Define a dynamic weight function that considers blocking
        # This function wraps the user's weight logic but returns infinity
        # if the target node 'v' is occupied by another path.
        def dynamic_weight(u, v, d):
            # Collision Check:
            # If v is currently used by another path (>0) and is NOT the 
            # target of our current path, it is effectively a wall.
            if node_usage[v] > 0 and v != target:
                return float('inf')

            return weight(u, v, d)


        # 4. Attempt to find a new shortest path
        try:
            # We use dijkstra_path. If dynamic_weight returns inf, 
            # the algorithm treats the edge as non-existent.
            new_path = nx.dijkstra_path(G, source, target, weight=dynamic_weight)
            
            # If successful, update the list and the usage counter
            paths[i] = new_path
            node_usage.update(new_path)
            
        except nx.NetworkXNoPath:
            # If no path exists that avoids other paths, revert to the original
            # (The original is guaranteed valid relative to the others 
            # because we started with a valid set, assuming input was valid)
            node_usage.update(old_path)

def add_grid_edges(G, V):
    """
    Adds edges to G between nodes in subset V if they are 
    exactly 1 unit apart and align on the x or y axis.
    """
    # Dictionaries to group nodes by coordinate
    # cols: key = x, value = list of (y, node_id)
    # rows: key = y, value = list of (x, node_id)
    cols = defaultdict(list)
    rows = defaultdict(list)

    # 1. Group nodes
    for node in V:
        # Verify node has 'pos' attribute
        if 'pos' not in G.nodes[node]:
            continue
            
        x, y = G.nodes[node]['pos']
        
        # We store the coordinate that differs and the node ID
        cols[x].append((y, node))
        rows[y].append((x, node))

    # 2. Process Columns (Same X, check Y distance)
    for x_val, nodes in cols.items():
        # Sort by y coordinate to check adjacent nodes
        nodes.sort(key=lambda k: k[0])
        
        for i in range(len(nodes) - 1):
            y1, n1 = nodes[i]
            y2, n2 = nodes[i+1]
            
            # Exact comparison: distance must be exactly 1
            if y2 - y1 == 1:
                G.add_edge(n1, n2)

    # 3. Process Rows (Same Y, check X distance)
    for y_val, nodes in rows.items():
        # Sort by x coordinate to check adjacent nodes
        nodes.sort(key=lambda k: k[0])
        
        for i in range(len(nodes) - 1):
            x1, n1 = nodes[i]
            x2, n2 = nodes[i+1]
            
            # Exact comparison: distance must be exactly 1
            if x2 - x1 == 1:
                G.add_edge(n1, n2)
                



def split_opposite_edges(G, v):
    """
    Iterates through all planar triangles incident to node v by sorting 
    neighbors angularly. Adds a split node at the midpoint of the edge 
    opposite to v for each triangle found.
    """
    # 1. Get position of center node v
    if 'pos' not in G.nodes[v]:
        raise ValueError(f"Node {v} does not have a 'pos' attribute.")
    vx, vy = G.nodes[v]['pos']

    # 2. Get neighbors and calculate their angles relative to v
    #    atan2 returns values in (-pi, pi]
    neighbor_data = []
    for u in G.neighbors(v):
        if 'pos' not in G.nodes[u]:
            continue
        ux, uy = G.nodes[u]['pos']
        angle = math.atan2(uy - vy, ux - vx)
        neighbor_data.append((angle, u))

    # 3. Sort neighbors by angle (Clockwise/Counter-Clockwise)
    neighbor_data.sort(key=lambda x: x[0])
    
    # Extract just the sorted nodes
    sorted_neighbors = [u for angle, u in neighbor_data]
    count = len(sorted_neighbors)
    
    if count < 2:
        return

    # List to store edges to split: (node_u, node_w)
    # We collect them first to avoid modifying the graph topology 
    # while iterating over the neighbors logic.
    edges_to_split = []

    # 4. Iterate through consecutive pairs (wrapping around to close the loop)
    for i in range(count):
        u = sorted_neighbors[i]
        w = sorted_neighbors[(i + 1) % count]

        # If an edge exists between two consecutive angular neighbors,
        # then (v, u, w) forms a triangle face in the planar embedding.
        if G.has_edge(u, w):
            edges_to_split.append((u, w))

    # 5. Perform the splitting
    # Note: We must check if edge exists again, in case a previous iteration
    # affected it (though in a valid triangulation, edges opposite v 
    # shouldn't overlap in a way that breaks this logic).
    for u, w in edges_to_split:
        if not G.has_edge(u, w):
            continue

        new_node = max(G.nodes()) + 1
        

        # B. Calculate Midpoint
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[w]['pos']
        midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

        # C. Update Graph
        # 1. Remove the old edge opposite to v
        G.remove_edge(u, w)
        
        # 2. Add the new node
        G.add_node(new_node, pos=midpoint)
        
        # 3. Add edges connecting the split node to the original endpoints
        G.add_edge(u, new_node)
        G.add_edge(new_node, w)
        G.add_edge(v, new_node) 



def add_intersection_nodes_fast(G: nx.Graph, node_subset: set[int] = None, maxdist=3.01):
    if node_subset is None:
        node_subset = set(G.nodes())

    # Use a fixed list of edges to ensure consistent indexing
    g = G.subgraph(node_subset)
    pos_dict = nx.get_node_attributes(g, "pos")
    edges_list = list(g.edges())
    if not edges_list:
        return

    # 1. Prepare Midpoints & Segment Data
    # Shape: (NumEdges, 2 coordinates [u, v], 2D pos [x, y])
    edge_coords = np.array([ [pos_dict[u], pos_dict[v]] for u, v in edges_list ])
    midpoints = edge_coords.mean(axis=1)

    # 2. Spacial Query
    tree = cKDTree(midpoints)

    candidate_pairs = list(tree.query_ball_tree(tree, r=maxdist))

    # Flatten the list of lists into two arrays of indices
    idx_i = []
    idx_j = []
    for i, neighbors in enumerate(candidate_pairs):
        for j in neighbors:
            if i < j:  # Avoid self-comparison and double-counting
                idx_i.append(i)
                idx_j.append(j)

    if not idx_i:
        return

    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)

    # 3. Filter Shared Endpoints (Pre-vectorization)
    # We retrieve the original node IDs to check for connectivity
    u_ids = np.array([edges_list[i][0] for i in idx_i])
    v_ids = np.array([edges_list[i][1] for i in idx_i])
    s_ids = np.array([edges_list[j][0] for j in idx_j])
    t_ids = np.array([edges_list[j][1] for j in idx_j])

    no_shared_endpoints = (u_ids != s_ids) & (u_ids != t_ids) & \
                          (v_ids != s_ids) & (v_ids != t_ids)

    idx_i = idx_i[no_shared_endpoints]
    idx_j = idx_j[no_shared_endpoints]

    # 4. Vectorized Intersection Logic (Cramer's Rule / Determinants)
    # Segment 1: P_u -> P_v (Vector A), Segment 2: P_s -> P_t (Vector B)
    # P_u + k1*A = P_s + k2*B
    Pu = edge_coords[idx_i, 0, :]
    Pv = edge_coords[idx_i, 1, :]
    Ps = edge_coords[idx_j, 0, :]
    Pt = edge_coords[idx_j, 1, :]

    A = Pv - Pu
    B = Pt - Ps
    C = Ps - Pu

    # 2D Cross Product (Determinant) function
    def cross_2d(m1, m2):
        return m1[:, 0] * m2[:, 1] - m1[:, 1] * m2[:, 0]

    det = cross_2d(A, B)

    # Avoid division by zero (parallel lines)
    non_parallel = np.abs(det) > 1e-9

    # Calculate k1 and k2 for all pairs
    # k1 = (C x B) / (A x B)
    # k2 = (C x A) / (A x B)
    k1 = np.zeros_like(det)
    k2 = np.zeros_like(det)
    k1[non_parallel] = cross_2d(C[non_parallel], B[non_parallel]) / det[non_parallel]
    k2[non_parallel] = cross_2d(C[non_parallel], A[non_parallel]) / det[non_parallel]

    # Valid intersection mask (strictly between 0 and 1 to exclude endpoints)
    valid_mask = non_parallel & (k1 > 0) & (k1 < 1) & (k2 > 0) & (k2 < 1)

    # 5. Process Intersections and Update Graph
    valid_idx_i = idx_i[valid_mask]
    valid_idx_j = idx_j[valid_mask]
    valid_k1 = k1[valid_mask]
    valid_k2 = k2[valid_mask]
    # Calculate actual intersection points: Pu + k1*A
    intersect_pts = Pu[valid_mask] + valid_k1[:, np.newaxis] * A[valid_mask]

    next_node_id = max(G.nodes()) + 1
    segmentdict = defaultdict(list)
    nodes_to_add = []

    for idx in range(len(valid_idx_i)):
        new_node = next_node_id
        pos = tuple(intersect_pts[idx])

        nodes_to_add.append((new_node, {"pos": pos}))
        node_subset.add(new_node)

        # Add to segmentdict for splitting later
        edge_a = edges_list[valid_idx_i[idx]]
        edge_b = edges_list[valid_idx_j[idx]]
        segmentdict[edge_a].append((valid_k1[idx], new_node))
        segmentdict[edge_b].append((valid_k2[idx], new_node))

        next_node_id += 1

    # Apply updates to G
    G.add_nodes_from(nodes_to_add)
    for (u, v), intersections in segmentdict.items():
        # Sort by k value along segment
        intersections.sort(key=lambda x: x[0])
        path = [u] + [node_id for k, node_id in intersections] + [v]

        for start, end in zip(path, path[1:]):
            G.add_edge(start, end)

        if G.has_edge(u, v):
            G.remove_edge(u, v)
    
    return [i[0] for i in nodes_to_add]


def add_intersection_nodes_slow(G: nx.Graph, node_subset: set[int]):
    next_node_id = max(G.nodes()) + 1

    g = G.subgraph(node_subset)
    pos = nx.get_node_attributes(g, "pos")
    nodes_to_add = []
    curredges = list(g.edges())
    segmentdict = {t: [] for t in curredges}


    for idx, (u, v) in enumerate(curredges):
        for idx2, (s, t) in enumerate(curredges):
            if  idx2 < idx or u == s or v == t or u == t or v == s:
                continue

            pu, pv = np.array(pos[u]), np.array(pos[v])
            ps, pt = np.array(pos[s]), np.array(pos[t])

            r, k1, k2 = shapeutils.segment_intersection(pu, pv, ps, pt)
            if r is not None:
                nodes_to_add.append((next_node_id, {"pos": tuple(r)}))
                node_subset.add(next_node_id)
                segmentdict[(u, v)].append((k1, next_node_id))
                segmentdict[(s, t)].append((k2, next_node_id))
                next_node_id+=1

    G.add_nodes_from(nodes_to_add)

    for (u, v), inter in segmentdict.items():
        if inter:
            inter = [(0, u)] + inter + [(1, v)]
            inter.sort(key=lambda x: x[0])
            for prev, nxt in zip(inter, inter[1:]):
                G.add_edge(prev[1], nxt[1])
            assert isinstance(u, int)
            assert isinstance(v, int)
            G.remove_edge(u, v)
            
def find_degree1_degree3_pairs(G: nx.Graph):
    """
    Find all pairs (u, v) where u has degree 1, v has degree 3,
    connected by a simple path of only degree-2 nodes.
    
    Returns list of dicts with:
      - 'path': list of nodes from u to v (inclusive)
      - 'other_neighbors': tuple (a, b) of v's 2 neighbors not in path
    """
    results = []
    
    degree1_nodes = [n for n in G.nodes if G.degree(n) == 1]
    for u in degree1_nodes:
        # Walk the chain from u until we hit a non-degree-2 node
        path = [u]
        prev = u
        current = list(G.neighbors(u))[0]  # u has exactly 1 neighbor
        
        while G.degree(current) == 2:
            neighbors = list(G.neighbors(current))
            # Move to the neighbor that isn't where we came from
            nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
            prev = current
            path.append(current)
            current = nxt
        
        path.append(current)
        v = current
        # if path[0] == 10275:
        #     print(path)
        #     print(G.degree(path[1]), list(G.neighbors(path[1])))
        if G.degree(v) == 3:
            other_neighbors = tuple(n for n in G.neighbors(v) if n not in path)
            results.append({
                'path': path,
                'other_neighbors': other_neighbors
            })
    
    return results


