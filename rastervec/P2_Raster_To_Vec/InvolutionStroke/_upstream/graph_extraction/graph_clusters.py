# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/graph_extraction/graph_clusters.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

from contextlib import contextmanager
from time import perf_counter
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from .combinatorics import solve_exact_cover, subset_cover
from .planargraph_core import add_subgraph_to_G, collapse_nodes, find_degree3_pairs_within_distance, get_neighbors_of_neighbors, has_degree1_path, remove_all_except_paths, remove_shortest_path, solve_multi_source_greedy
from . import planargraph_core
from . import shapeutils
from itertools import chain
import math
from pprint import pprint
from .. import _utils as utils

from dataclasses import dataclass

import logging as _logging
_LOG = _logging.getLogger("rastervec.P2.InvolutionStroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))



@contextmanager
def timer(desc=""):
    start = perf_counter()
    yield
    print(f"{desc+' ' if desc else ''}time:", perf_counter() - start)

def printpos(p):
    return float(p[0]), float(p[1])
def getpos(G, u):
    p = G.nodes[u]["pos"]
    return float(p[0]), float(p[1])

@dataclass
class Boundary:
    nodedict: dict[int, bool] #boundary node -> is outer node
    centroid: tuple[float, float] #useful for debugging
    inner_nodes : set[int]
    
    def unique_degree(self):
        return len(self.outer_nodes())
    
    def outer_nodes(self):
        return [k for k, v in self.nodedict.items() if v]
    
    def all_nodes(self):
        return set(self.nodedict.keys()).union(self.inner_nodes)
    
    def outer_node_nbrs(self, u, G:nx.Graph):
        assert u in self.nodedict
        for v in G.neighbors(u):
            if v not in self.nodedict and v not in self.inner_nodes:
                yield v
    
    def outer_points(self, G):
        return np.array([
            np.array(G.nodes[u]["pos"]) for u, v in self.nodedict.items() if v])
    
        
    def boundary_polygon(self, G):
        return np.array([G.nodes[u]["pos"] for u in self.nodedict.keys()])


def remove_collinear_edges1(G: nx.Graph):
    to_remove = []
    for u in G.nodes():
        offsets = {}
        for v in G.neighbors(u):
            x1, y1 = G.nodes[u]['pos']
            x2, y2 = G.nodes[v]['pos']
            dx, dy = x1 - x2, y1 - y2
            offsets[(dx, dy)] = (u, v)
        if (0, 2) in offsets and (0, 1) in offsets:
            to_remove.append(offsets[(0, 2)])
        if (0, -2) in offsets and (0, -1) in offsets:
            to_remove.append(offsets[(0, -2)])
        if (2, 0) in offsets and (1, 0) in offsets:
            to_remove.append(offsets[(2, 0)])
        if (-2, 0) in offsets and (-1, 0) in offsets:
            to_remove.append(offsets[(-2, 0)])
            
        if (-2, -2) in offsets and (-1, -1) in offsets:
            to_remove.append(offsets[(-2, -2)])
        if (-2, 2) in offsets and (-1, 1) in offsets:
            to_remove.append(offsets[(-2, 2)])
        if (2, -2) in offsets and (1, -1) in offsets:
            to_remove.append(offsets[(2, -2)])
        if (2, 2) in offsets and (1, 1) in offsets:
            to_remove.append(offsets[(2, 2)])
            
        if (-1, 2) in offsets and (-0.5, 1) in offsets:
            to_remove.append(offsets[(-1, 2)])
        if (1, 2) in offsets and (0.5, 1) in offsets:
            to_remove.append(offsets[(1, 2)])
        if (1, -2) in offsets and (0.5, -1) in offsets:
            to_remove.append(offsets[(1, -2)])
        if (-1, -2) in offsets and (-0.5,-1) in offsets:
            to_remove.append(offsets[(-1, -2)])
        
        if (0, 1) in offsets and (0, 0.5) in offsets:
            to_remove.append(offsets[(0, 1)])
        if (0, -1) in offsets and (0, -0.5) in offsets:
            to_remove.append(offsets[(0, -1)])
        if (1, 0) in offsets and (0.5, 0) in offsets:
            to_remove.append(offsets[(1, 0)])
        if (-1, 0) in offsets and (-0.5, 0) in offsets:
            to_remove.append(offsets[(-1, 0)])
    G.remove_edges_from(to_remove)


def remove_collinear_edges(G: nx.Graph):
    """
    For each node, if multiple edges exist in the same direction
    (8 compass directions) with length <= 2 and step size 0.5,
    remove the longer ones.
    Uses exact comparisons.
    """
    # 8 primitive directions
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    # Allowed lengths (multiples of 0.5 up to 2)
    lengths = [0.5, 1.0, 1.5, 2.0]

    to_remove = []

    for u in G.nodes():
        x1, y1 = G.nodes[u]['pos']

        offsets = {}
        for v in G.neighbors(u):
            x2, y2 = G.nodes[v]['pos']
            dx = x2 - x1
            dy = y2 - y1
            offsets[(dx, dy)] = (u, v)

        # For each direction
        for dirx, diry in directions:

            # Collect all existing lengths in this direction
            found = []
            for L in lengths:
                dx = dirx * L
                dy = diry * L
                if (dx, dy) in offsets:
                    found.append((L, offsets[(dx, dy)]))

            # If more than one edge in same direction,
            # remove all except the shortest
            if len(found) > 1:
                found.sort(key=lambda x: x[0])  # sort by length
                for _, edge in found[1:]:
                    to_remove.append(edge)

    G.remove_edges_from(to_remove)
    
def remove_nodes_if_masked(G: nx.Graph,  mask):
    
    pos = nx.get_node_attributes(G, "pos")
    to_rm = []
    for i in G.nodes:
        p = np.floor(np.array(pos[i])).astype(int)
        if mask[p[1], p[0]] and mask[p[1]-1, p[0]-1]:
            to_rm.append(i)

    G.remove_nodes_from(to_rm)


def build_graph_fast(points, mask):
    """
    Build a graph connecting points with Manhattan distance <= (1,2) or (2,1)
    using cKDTree for efficiency.
    """
    points = np.array(points)
    G = nx.Graph()
    
    # Add nodes with original coordinates
    for i, p in enumerate(points):
        G.add_node(i, pos=tuple(p))
    
    # Build KDTree
    tree = cKDTree(points)
    
    # Maximum Manhattan distance allowed = 3 (1+2 or 2+1)
    max_manhattan = 3
        
    # Query neighbors efficiently
    for i, p in enumerate(points):
        # Find all points within Euclidean distance <= sqrt(3^2 + 0^2) (upper bound)
        idxs = tree.query_ball_point(p, r=max_manhattan, p=1.)
        
        for j in idxs:
            if i >= j:  # Avoid double edges
                continue
            if G.has_edge(i, j):
                print("problem: possible double edge", i, j, points[i], points[j])
                raise ValueError
                
            # Compute Manhattan distance
            dx = abs(p[0] - points[j][0])
            dy = abs(p[1] - points[j][1])
            if (dx <= 2 and dy <= 2):
                #check if the edge covers pixels not active on the mask (rare)
                    
                pixels = shapeutils.segment_pixels(p, points[j], approx=False)
                
                cond =  all(mask[q[1], q[0]] for q in pixels)
                if cond or (dx <=1 and dy <=1):
                    G.add_edge(i, j)
    
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def clusters_check(G, clusters):
    for c in clusters:
        boundary_edges = c["boundary_edges"]
        nodes = c["nodes"]
        for u in nodes:
            assert G.has_node(u)
        for u, v in boundary_edges:
            if not u in nodes:
                print(G.nodes[u]["pos"])
                raise ValueError("boundary edge not found")

def find_triangle_clusters(G: nx.Graph):
    """
    Find clusters of triangles:
      - each node belongs to at least one triangle
      - clusters are connected through shared triangle nodes
    
    Returns:
        list of dicts, each with:
            {
                'nodes': set of node indices (cluster),
                'boundary_edges': list of (u, v) edges where u in cluster, v not in cluster
            }
    """
    # Find all triangles (3-cycles)
    
    # node_candidates = [n for (n, d) in G.degree() if d>2]
    # H = nx.subgraph(G, node_candidates)
    H = G
    with utils.timer("triangles"):
        triangles = [tuple(sorted(tri)) for tri in nx.enumerate_all_cliques(H) if len(tri) == 3]

    if not triangles:
        return []

    # Build a graph of nodes that share triangles
    tri_graph = nx.Graph()
    for tri in triangles:
        tri_graph.add_nodes_from(tri)
        tri_graph.add_edges_from([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
    blocks = list(nx.biconnected_components(tri_graph))
    # For each cluster, find boundary edges (u in cluster, v not in cluster)
    cluster_data = []
    for block in blocks:
        boundary_edges = []
        for u in block:
            for v in G.neighbors(u):
                if v not in block:
                    boundary_edges.append((u, v))
        cluster_data.append({
            "nodes": set(block),
            "boundary_edges": boundary_edges,
            "center": G.nodes[list(block)[0]]["pos"]
        })


    return cluster_data

def remove_degree_2(G: nx.Graph, clusters: list[dict]):
    
    to_remove = []
    for idx, cluster in enumerate(clusters):
        b =cluster["boundary_edges"]
        if len(b) == 2 and b[0][0] != b[1][0]:
            to_remove.append(idx)
            for u in cluster["nodes"]:
                G.nodes[u]['cluster'].remove(idx+1)
    
    for i in reversed(to_remove):
        del clusters[i]
            
            

def graph_outer_face(G: nx.Graph, allowed_nodes=None):
    """
    Traverse the outer boundary of a planar embedded graph using
    CCW neighbor selection ("graph convex hull").
    
    Parameters
    ----------
    G : nx.Graph
        Planar graph with 'pos' attribute for each node: (x, y)
    allowed_nodes : set or None
        If provided, restrict traversal to this subset of nodes.
        
    Returns
    -------
    boundary : list
        List of node IDs forming the outer face in CCW order.
    """
    if allowed_nodes is None:
        allowed_nodes = set(G.nodes)
    else:
        allowed_nodes = set(allowed_nodes)
        
    debug = False

    # Helper: angle between two vectors (in radians)
    def angle(v):
        return np.arctan2(v[1], v[0])

    def ccw_angle(from_vec, to_vec):
        """Counterclockwise angle from from_vec to to_vec in [0, 2π)."""
        a = (angle(to_vec) - angle(from_vec)) % (2 * np.pi)
        return a

    pos = nx.get_node_attributes(G, "pos")

    # Start with the lowest (lowest y, then x) node in allowed set
    start = min(allowed_nodes, key=lambda n: (pos[n][1], pos[n][0]))
    
    

    # Find initial direction: neighbor that is most clockwise (rightmost)
    start_pos = np.array(pos[start])
    # debug = start_pos[0] == 1280 and start_pos[1] == 1255
    # print("start: ", start, start_pos)
    
    neighbors = [nbr for nbr in G.neighbors(start) if nbr in allowed_nodes]
    if not neighbors:
        return [start]

    # Initial direction vector pointing from lowest node to -infinity
    angles = [ccw_angle(np.array([0, -1]), np.array(pos[v])-start_pos) for v in neighbors]
    init_neighbor = neighbors[np.argmin(angles)]

    
    boundary = [start]
    curr = init_neighbor
    it =0
    while it< 100_000:
        it+=1
        # print("curr", it, curr, pos[curr])
        boundary.append(curr)
        # Compute incoming direction vector
        v_in = np.array(pos[boundary[-2]]) - np.array(pos[curr])

        # All candidate next neighbors
        nbrs = [nbr for nbr in G.neighbors(curr) if nbr in allowed_nodes and nbr != boundary[-2]]
        if not nbrs:
            break

        # Pick neighbor with smallest CCW angle
        v_curr = np.array(pos[curr])
        angles = []
        for nbr in nbrs:
            v_next = np.array(pos[nbr]) - v_curr
            angles.append(ccw_angle(v_in, v_next))
        next_node = nbrs[np.argmin(angles)]
        
        if pos[next_node] == pos[boundary[-2]]:
            raise ValueError("problem", next_node, pos[next_node])
        if next_node == start:
            break  # completed loop

        curr = next_node
    if it == 100_000:
        raise ValueError("too many iterations for graph_outer_face")
    
    # if tuple(start_pos) == (82, 97):
    #     exit()

    return boundary

def remove_inside_of_boundary(G: nx.Graph, boundary: Boundary):
    
    G.remove_nodes_from(boundary.inner_nodes)
    bnodes = [u for u in boundary.nodedict.keys() if G.has_node(u)]
    
     # Precompute valid boundary edges (the outer face cycle)
    boundary_edges = {
        (bnodes[i], bnodes[(i + 1) % len(bnodes)]) for i in range(len(bnodes))
    }
    boundary_edges |= {(b, a) for (a, b) in boundary_edges}  # undirected version

    # Remove edges within the cluster that are not part of the boundary
    for u in set(bnodes):
        for v in list(G[u]):  # only iterate over actual neighbors
            if v in bnodes and (u, v) not in boundary_edges:
                G.remove_edge(u, v)
    

def get_boundary(G: nx.Graph, cluster, debug = False, add_virtual = 2.0) -> Boundary | None:
    """
    Remove all interior nodes and non-boundary edges from a planar cluster.

    Parameters
    ----------
    G : nx.Graph
        Graph with 'pos' coordinates.
    cluster : iterable
        Node IDs forming a connected region (subset of G.nodes).

    Returns
    -------
    boundary : list
        Ordered list of boundary node IDs (outer face).
    """

    # Compute outer boundary of the cluster
    boundary = graph_outer_face(G, allowed_nodes=cluster["nodes"])
    boundary_set = set(boundary)

    center = shapeutils.compute_centroid([G.nodes[x]["pos"] for x in boundary_set])
    # print()
    #print("center", center)
    # print([G.nodes[u]["pos"] for u in boundary])
    inner_nodes = set()
    for v in cluster["nodes"]:
        if G.has_node(v) and v not in boundary_set:
            inner_nodes.add(v)

    boundary_edges = cluster["boundary_edges"]
    nodedict = {k: False for k in boundary}
    for u, v in boundary_edges:
        if not u in boundary:
            to_rm = [x for x in cluster["nodes"] if len(G.nodes[x]["cluster"]) == 1]
            G.remove_nodes_from(to_rm)
            return None
        nodedict[u] = True
    
    nodedict_nxt = nodedict.copy()
    
    # outer = [k for k, v in nodedict.items() if v]
    # if len(outer)> 2 and add_virtual>0:
    #     start = outer[0]
    #     prev = start
    #     polyline = np.array([G.nodes[x]["pos"] for x in nodedict.keys()])
    #     node_to_idx = {nid: i for i, nid in enumerate(nodedict.keys())}
    #     idx_to_node = {i: nid for i, nid in enumerate(nodedict.keys())}
    #     # if debug:
    #     #     print("start", start, polyline[node_to_idx[start]])
    #     new_outer_nodes =set()
    #     for k, v in nodedict.items():
    #         if v and k != prev:
    #             a, b = node_to_idx[prev], node_to_idx[k]
    #             pl = polyline[a:b+1]
    #             idx, dist = shapeutils.furthest_point_signed(pl, mode=-1)
    #                 #print(start, a, b, polyline[a], polyline[b])
    #             if dist < -add_virtual:
    #                 if debug:
    #                     print(start, "distance between", pl[0], pl[-1], dist)
    #                 p = pl[idx]
    #                 nv = shapeutils.normal_towards_point([pl[0], pl[-1]], p)
    #                 newouter = p+0.1*nv
    #                 node = idx_to_node[a+idx]
    #                 assert tuple(G.nodes[node]["pos"]) == printpos(p)
    #                 print("added", newouter , " at", p)
    #                 planargraph_core.add_subgraph_with_pos(G, np.array([newouter]), [], [(node, 0)])
    #                 new_outer_nodes.add(node)
    #             prev = k
                
    #     if prev != start:
    #         a, b = node_to_idx[prev], node_to_idx[start]
    #         pl = np.concatenate([polyline[a:], polyline[:b+1]])
    #         #planargraph_core.add_path_subgraph(G, pl+0.5)
    #         idx, dist = shapeutils.furthest_point_signed(pl, mode=-1)
    #         if dist < -add_virtual:
    #             if debug:
    #                 print(start, "distance between", pl[0], pl[-1], dist)
    #             p = pl[idx]
    #             nv = shapeutils.normal_towards_point([pl[0], pl[-1]], p)
    #             newouter = p+0.1*nv
    #             node = idx_to_node[a+idx] if a + idx < len(nodedict) else idx_to_node[idx+a-len(nodedict)]
    #             print("added", newouter , " at", p)
    #             assert tuple(G.nodes[node]["pos"]) == printpos(p)
    #             planargraph_core.add_subgraph_with_pos(G, np.array([newouter]), [], [(node, 0)])
    #             new_outer_nodes.add(node)
        
    #     for i in new_outer_nodes:
    #         nodedict_nxt[i] = True
        
            
    return Boundary(nodedict=nodedict_nxt,
                                   centroid=(float(center[0]), float(center[1])),
                                   inner_nodes=inner_nodes)

def get_boundaries(G: nx.Graph, clusters, debug=False, add_virtual = 0) ->dict[int, Boundary]:
    boundaries = {}
    for i, c in enumerate(clusters):
        # print("getting cluster boundary for ", c["center"])
        res = get_boundary(G, c, debug=debug, add_virtual = add_virtual)
        if res != None:
            boundaries[i] = res
    return boundaries

def get_circle_arc(cycle: list[int], source: int, target: int):
    src_idx = cycle.index(source)
    tgt_idx = cycle.index(target)

    # Two arcs around the cycle
    if src_idx <= tgt_idx:
        arc1 = cycle[src_idx:tgt_idx + 1]
        arc2 = cycle[tgt_idx:] + cycle[:src_idx + 1]
    else:
        arc1 = cycle[tgt_idx:src_idx + 1]
        arc2 = cycle[src_idx:] + cycle[:tgt_idx + 1]
    #print(arc1)
    #print(arc2)
    # return the shorter arc
    return arc1 if len(arc1) < len(arc2) else arc2
            
def resolve_degree_2_boundaries(G: nx.Graph, boundaries: dict[int, Boundary], img):
    """
    For each cluster with exactly 2 outer connections:
      - compute the shortest path between the boundary nodes
      - remove all intermediate nodes and edges along the path, keep endpoints

    Args:
        G: networkx graph (modified in-place)
        cluster_data: output from find_triangle_clusters_with_boundary(G)

    Returns:
        None (modifies G in-place)
    """
    boundaries_filtered = [i for i, b in boundaries.items() if b.unique_degree()==2]
    
    pos = nx.get_node_attributes(G, "pos")

    def weight(a, b, att):
        pa = pos[a]
        pb = pos[b]
        #mid = shapeutils.compute_centroid([pa, pb])
        r = shapeutils.grid_sample(img, np.array([pa, pb]), mode="bilinear shifted")
        v = float(np.mean(r))
        #we want the path with the max average value
        return 255-v
        
    
    for i in boundaries_filtered:
        boundary = boundaries[i]
        #remove_inside_of_boundary(G, boundary)
        source, target = boundary.outer_nodes()
        assert G.has_node(source)
        if not G.has_node(target):
            print(target, G.nodes[source]["pos"])
            assert G.has_node(target)
            
        V = boundary.all_nodes()
        v1, v2 = outbound_vectors(G, boundary)
        #print(boundary.centroid)
        #print(source, target, printpos(pos[source]), printpos(pos[target]))
        if np.dot(v1, v2)> 0:
            cycle = list(boundary.nodedict.keys())
            src_idx = cycle.index(source)
            tgt_idx = cycle.index(target)

            # Two arcs around the cycle
            if src_idx <= tgt_idx:
                arc1 = cycle[src_idx:tgt_idx + 1]
                arc2 = cycle[tgt_idx:] + cycle[:src_idx + 1]
            else:
                arc1 = cycle[tgt_idx:src_idx + 1]
                arc2 = cycle[src_idx:] + cycle[:tgt_idx + 1]

            # Keep the longer arc
            path = arc1 if len(arc1) >= len(arc2) else arc2

            planargraph_core.remove_all_except_paths(G, V, [path])

            
        else:
            H = nx.subgraph(G, V)
            path = nx.shortest_path(H, source, target)
            # path = planargraph_core.shortest_minimax_path_restricted(G, source, target, V, weight=weight)
            planargraph_core.remove_all_except_paths(G, V, [path])
        # if len(boundary.nodedict) ==3:
        #     other = [k for k in boundary.nodedict.keys() if k not in [source, target]][0]
        #     G.remove_node(other)
        #     continue
        
        # for v in respath:
        #     G.nodes[v]["color"] = "red"


        
    for i in (boundaries_filtered):
        del boundaries[i]

def remove_degree_0_boundary(G: nx.Graph, boundary: Boundary, boundary_idx: int):
    
    if boundary.unique_degree()==0:
        G.remove_nodes_from(boundary.all_nodes())
        return True
    return False
        
def resolve_degree_1_boundaries(G: nx.Graph, boundaries: dict[int, Boundary], img):
    
    def euclidean_weight(u, v, data):
        x1, y1 = G.nodes[u]["pos"]
        x2, y2 = G.nodes[v]["pos"]
        return math.dist((x1, y1), (x2, y2))
    
    
    boundaries_filtered = [idx for idx, b in boundaries.items() if b.unique_degree() == 1]
    print("num of deg 1 boundaries: ", len(boundaries_filtered))
    for idx in boundaries_filtered:
        boundary = boundaries[idx]
        out_u = boundary.outer_nodes()[0]
        src = G.nodes[out_u]["pos"]
        nodes = list(boundary.nodedict.keys())
        points = [G.nodes[k]["pos"] for k in nodes]
        
        p, idx, dist = shapeutils.find_closest_furthest_point(points, src, closest=False)
        node_idx = nodes[idx]
        color_clusters(G, [{"nodes":[node_idx]}], color="black")

        
        path = nx.shortest_path(G, source=out_u, target=node_idx, weight=euclidean_weight)
        remove_all_except_paths(G, boundary.all_nodes(), [path])
        # if spurious_path_condition(path, G, img):
        #     G.remove_nodes_from(path[1:])
        
    for i in boundaries_filtered:
        del boundaries[i]
        

def shorten_endpoints(G: nx.Graph, img: np.ndarray, threshold: float = 25.0):
    starts = [n for (n, d) in G.degree() if d == 1]
    pos = nx.get_node_attributes(G, "pos")

    for s in starts:
        current = s
        while current is not None and G.has_node(current) and G.degree(current) <= 1:
            intensity = shapeutils.grid_sample(
                img, np.array([pos[current]]), mode="bilinear shifted"
            )
            # print(pos[current], "intensity ", intensity)
            if intensity[0] >= threshold:
                break
            neighbors = list(G.neighbors(current))
            G.remove_node(current)
            current = neighbors[0] if len(neighbors)==1 else None
    
        
def consistency_check(G:nx.Graph, boundaries: dict[int, Boundary]):
    for idx, boundary in boundaries.items():
        for i in boundary.nodedict:
            if not G.has_node(i):
                print("G is missing ", i)
                t = list(boundary.nodedict)
                r = [G.nodes[x]['pos'] for x in t if G.has_node(x)]
                print(r)
                
                raise ValueError
        for i in boundary.nodedict:
            assert not i in boundary.inner_nodes
    for i in G.nodes:
        try:
            G.nodes[i]['pos']
        except KeyError:
            print("mssing pos:", i, G.has_node(i))
            raise ValueError
        

def boundary_consistency(G: nx.Graph, cluster, boundary):
    for i in boundary:
        pos = G.nodes[i]["pos"]
        assert i in cluster, f"{i}, {pos} not in cluster"
   

def collapse_boundary_to_point(G: nx.Graph, boundary: Boundary, point):
    """
    Collapse triangle clusters into a single node at the centroid.
    
    Parameters:
    - G: NetworkX graph with 'pos' attribute on nodes
    - clusters: list of sets of node indices
    """
    # Track new node index
    next_node_id = max(G.nodes()) + 1
    
    to_rm = G.subgraph(boundary.nodedict.keys()).edges()
    G.remove_edges_from(to_rm)
    
    new_node = next_node_id
    # print("new node:", new_node)
    next_node_id += 1
    G.add_node(new_node, pos=tuple(point))
    
    outer = boundary.outer_nodes()
    
    # Connect new node to all outer neighbors
    for nbr in outer:
        G.add_edge(new_node, nbr)
    
    # Remove original cluster nodes
    to_rm = boundary.nodedict.keys() - outer
    G.remove_nodes_from(to_rm)
    
    return G        

def process_special_degree_3_cluster(G: nx.Graph, boundary: Boundary, mask):
    pos = nx.get_node_attributes(G, "pos")

    
    boundary_nodes = [k for k in boundary.nodedict.keys() if len(boundary.nodedict[k]) > 0]
    if len(boundary_nodes) != 3:
        return False
    a, b, c = boundary_nodes
    pairs = [(a, b), (a, c), (b, c)]
    dists = []
    #get the point pair in the boundary triangle whose length is the shortest
    for source, target in pairs:
        dist = np.linalg.norm(np.array(pos[source])-np.array(pos[target]))
        dists.append((source, target, dist))
    source, target, _  = sorted(dists, key=lambda x: x[2])[0]
    #node that is at the sharp point of the triangle
    sharp = list({a, b, c} - {source, target})[0]
    
    v_sharp = get_outbound_vector(sharp, boundary.outer_node_nbrs(sharp, G), G)
    v1 = get_outbound_vector(source,  boundary.outer_node_nbrs(source, G), G)
    v2 = get_outbound_vector(target, boundary.outer_node_nbrs(target, G), G)
    
    d1 = np.dot(v_sharp, v1)
    d2 = np.dot(v_sharp, v2)
    if d1 < -0.7 and d2 < -0.7:
        print("removed deg 3")
        # for i in boundary.nodedict:
        #     G.nodes[i]["color"] = "red"
        remove_inside_of_boundary(G, boundary)
        remove_shortest_path(G, source, target)
        G.nodes[source]["color"] = "red"
        G.nodes[target]["color"] = "red"
        return True
    return False

    # # we know there's only 1 outgoing edge
    # nxt = [
    #     u for u in G.neighbors(sharp) if u not in boundary.nodedict.keys() and u not in boundary.inner_nodes
    #     ][0]
    # # print("nxt pos:", pos[nxt])
    # #walk starting from the start node, to see if we reach a degree 1 node
    # path = has_degree1_path(G, sharp, nxt)

def process_special_deg_3_clusters(G: nx.Graph, boundaries: list[Boundary], mask):
    to_rm = []
    for idx, b in enumerate(boundaries):
        if process_special_degree_3_cluster(G, b, mask):
            to_rm.append(idx)
        
    
    for i in reversed(to_rm):
        del boundaries[i]


def edge_vector(in_node, out_node, G: nx.Graph):
    pos = nx.get_node_attributes(G, "pos")
    p0 = np.array(pos[in_node])
    p1 = np.array(pos[out_node])

    if G.degree(out_node) != 2:
        tmp = p1 - p0
        return tmp/np.linalg.norm(tmp)
    else:
        # better estimate for degree-2 nodes
        nbr = [x for x in G.neighbors(out_node) if x != in_node][0]
        p2 = np.array(pos[nbr])
        mid0 = shapeutils.lerp(p0, p1, 0.5)
        mid1 = shapeutils.lerp(p1, p2, 0.5)
        tmp = mid1 - mid0
        return tmp/np.linalg.norm(tmp)


def get_outbound_vector(node, nbrs, G: nx.Graph):
    pos = nx.get_node_attributes(G, "pos")

    p0 = np.array(pos[node])

    # single outgoing edge
    if len(nbrs) == 1:
        return edge_vector(node, nbrs[0], G)

    # multiple outgoing edges: use bisector of furthest-apart pair
    vecs = [edge_vector(node, nbr, G) for nbr in nbrs]

    # normalize only for angle comparison
    nvecs = [v / np.linalg.norm(v) for v in vecs]

    # find pair with smallest dot product (largest angle)
    min_dot = np.inf
    best_pair = (np.array([1, 0]), np.array([1, 0]))

    for i in range(len(nvecs)):
        for j in range(i + 1, len(nvecs)):
            d = np.dot(nvecs[i], nvecs[j])
            if d < min_dot:
                min_dot = d
                best_pair = (nvecs[i], nvecs[j])

    # bisector
    tmp = best_pair[0] + best_pair[1]
    return tmp/np.linalg.norm(tmp)

def process_special_degree_4(G: nx.Graph, boundary: Boundary, boundary_idx: int, img: np.ndarray):
    outer_points = boundary.outer_points(G)
    outer_nodes = boundary.outer_nodes()
    if len(outer_points) != 4:
        return False
    d = shapeutils.get_closed_polygon_distances(outer_points)
    idx = np.argsort(d)
    # print("successive distances", idx, d)
    #test if it's not roughly rectangle
    if (idx[0] + 1)%4 == idx[1] or (idx[1]+1)%4 == idx[0]:
        return False
    
    short_seg_a = outer_points[idx[0]], outer_points[(idx[0]+1)%4]
    short_seg_b = outer_points[idx[1]], outer_points[(idx[1]+1)%4]
    
    mid_segment = [shapeutils.lerp(*short_seg_a, 0.5), shapeutils.lerp(*short_seg_b, 0.5)]
    #planargraph_core.add_subgraph_with_pos(G, np.array(mid_segment), [(0, 1)])
    
    long_seg_a = outer_points[idx[2]], outer_points[(idx[2]+1)%4]
    long_seg_b = outer_points[idx[3]], outer_points[(idx[3]+1)%4]
    cross1 = outer_points[0], outer_points[2]
    cross2 = outer_points[1], outer_points[3]
    #planargraph_core.add_subgraph_with_pos(G, np.array(cross1), [(0, 1)])
    #planargraph_core.add_subgraph_with_pos(G, np.array(cross2), [(0, 1)])

    assert len(img.shape) == 2
    def pixel_avg(p0, p1):
        pixels = shapeutils.segment_pixels(p0, p1, approx=False)
        msk = np.zeros_like(img).astype(np.uint8)
        for q in pixels: msk[q[1], q[0]]=255 
        # cv2.imwrite(f"debug_files/msk_{p0.astype(int)}.png", msk)
        return sum(int(img[q[1], q[0]]) for q in pixels)/len(pixels), min(int(img[q[1], q[0]]) for q in pixels)
    
    a1, a1min = pixel_avg(*long_seg_a)
    a2, a2min = pixel_avg(*long_seg_b)
    #a_mid = pixel_avg(*mid_segment)
    c1, c1min = pixel_avg(*cross1)
    c2, c2min = pixel_avg(*cross2)

    # print(a1, a2, a_mid)
    
    if c1 + c2 < a1 + a2 and a1min > 10 and a2min > 10:
        print("deg 4 removed!")
        print(a1, a2, c1, c2, c1 + c2, a1 + a2)
        print(a1min, a2min, c1min, c2min)
        print(boundary.centroid)
        a = outer_nodes[idx[0]], outer_nodes[(idx[0]+1)%4]
        b = outer_nodes[idx[1]], outer_nodes[(idx[1]+1)%4]
        cycle = list(boundary.nodedict.keys())
        #print(a, getpos(G, a[0]), getpos(G, a[1]))
        #print(b, getpos(G, b[0]), getpos(G, b[1]))
        arc1 = get_circle_arc(cycle, *a)[1:-1]
        arc2 = get_circle_arc(cycle, *b)[1:-1]
        G.remove_nodes_from(arc1)
        G.remove_nodes_from(arc2)
        remove_inside_of_boundary(G, boundary)
        #remove_shortest_path(G, *a)
        #remove_shortest_path(G, *b)
        return True

    return False


def process_special_degree_3(G: nx.Graph, boundary: Boundary, boundary_idx: int, img: np.ndarray):
    outer_points = boundary.outer_points(G)
    outer_nodes = boundary.outer_nodes()
    if len(outer_points) != 3:
        return False
    
    distances = [(math.dist(outer_points[a], outer_points[b]), a, b) for a, b in [(0, 1), (1, 2), (2, 0)]]
    distances.sort(key=lambda x: x[0])
    #print("sorted distances", distances)
    if distances[0][0]*2 > distances[1][0] or distances[0][0]*2 > distances[2][0]:
        return False
    seg1 =  outer_nodes[distances[1][1]], outer_nodes[distances[1][2]]
    seg2 =  outer_nodes[distances[2][1]], outer_nodes[distances[2][2]]
    arc1 = get_circle_arc(list(boundary.nodedict.keys()), *seg1)
    arc2 = get_circle_arc(list(boundary.nodedict.keys()),  *seg2)
    
    pos = nx.get_node_attributes(G, "pos")
    line1 = np.array([pos[x] for x in arc1])
    line2 = np.array([pos[x] for x in arc2])
    _, d1 = shapeutils.furthest_point_signed(line1)
    _, d2 = shapeutils.furthest_point_signed(line2)
    # print(d1, d2)
    if abs(d1) > 1.99 or abs(d2)> 1.99:
        return False
    
    remove_all_except_paths(G, boundary.all_nodes(), [arc1, arc2])
    return True


def process_pairs_of_deg_3(G: nx.Graph, img):
    
    pairs = find_degree3_pairs_within_distance(G, 5)
    print("num pairs of degree 3 found:", len(pairs))
    pos = nx.get_node_attributes(G, "pos")
    for u, v, d in pairs:
        if not (G.has_node(u) and G.has_node(v)):
            continue
        pu, pv = np.array(pos[u]), np.array(pos[v])
        mid = shapeutils.lerp(pu, pv, 0.5)
        pts = np.array([pu, pv, mid])
        res = shapeutils.grid_sample(img, pts, mode="bilinear shifted")
        if res[2] < res[0] and res[2] < res[1]:
            print("removed!")
            remove_shortest_path(G, u, v)
        # else:
        #     path = nx.shortest_path(G, source=u, target=v)
        #     collapse_nodes(G, path, pos=tuple(mid))

def annotated_voronoi_graph(vor_vertices, vor_edges, poly_array):
    """
    1. Builds a graph from Voronoi data.
    2. Connects degree-1 nodes to the nearest boundary point.
    3. Marks nodes reachable from the boundary until a junction (degree > 2).
    """
    G = nx.Graph()
    
    # Add initial Voronoi nodes
    for i, pos in enumerate(vor_vertices):
        G.add_node(i, pos=tuple(pos), accessible=None, type='voronoi')
    G.add_edges_from(vor_edges)
    
    # 1. Identify degree-1 nodes and connect to boundary
    tree = cKDTree(poly_array)
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    
    # We will add boundary nodes starting from an index higher than vor_vertices
    next_node_idx = len(vor_vertices)
    boundary_node_map = {}

    for leaf in leaves:
        leaf_pos = G.nodes[leaf]['pos']
        dist, poly_idx = tree.query(leaf_pos)
        
        b_pos = tuple(poly_array[poly_idx])
        b_idx = next_node_idx
        
        G.add_node(b_idx, pos=b_pos, accessible=poly_idx, type='boundary')
        G.add_edge(b_idx, leaf)
        
        boundary_node_map[b_idx] = poly_idx
        next_node_idx += 1

    # 2. Mark accessible nodes starting from each boundary point individually
    for b_node, poly_idx in boundary_node_map.items():
        # Traverse from boundary node into the skeleton
        # Stop if we hit a node that is already part of a junction (degree > 2)
        curr = b_node
        visited = {curr}
        
        # Simple walk: since we start at a boundary (deg 1), 
        # we follow the path as long as degree is <= 2
        # We use a loop to traverse the 'hair' branch
        frontier = [b_node]
        while frontier:
            u = frontier.pop()            
            # Continue to neighbors only if this isn't a junction
            # (In a medial axis, hairs are simple chains)
            if G.degree(u) <= 2:
                G.nodes[u]['accessible'] = poly_idx
                for v in G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        frontier.append(v)
                        
    return G

def outbound_vectors(G: nx.Graph, boundary: Boundary):
    vecs = np.array([
        get_outbound_vector(u, list(boundary.outer_node_nbrs(u, G)),
            G) for u, is_outer in boundary.nodedict.items() if is_outer
        ])
    vecs =  vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

def validate_paths(G, boundary: Boundary, all_paths, n_intersection):
    # Number of paths is as expected?
    num_paths = len(boundary.outer_nodes())+ n_intersection-1
    if len(all_paths) < num_paths:
        print("problem, not enough paths:", len(all_paths))
        return False
    
    # All outer nodes reached?
    reached = set(x[-1] for x in all_paths)
    if not set(boundary.outer_nodes()).issubset(reached):
        print("problem, not all outer nodes reached")
        return False
        
    return True

def optimal_cluster_search(G: nx.Graph, boundary: Boundary, mask: np.ndarray, background_pixels: set[tuple[int, int]]):
    outer_nodes = boundary.outer_nodes()
    vecs = outbound_vectors(G, boundary)
    outer_points = boundary.outer_points(G)
    pos = nx.get_node_attributes(G, "pos")
    vecmap = {a: b for a, b in zip(boundary.outer_nodes(), vecs)}
    
    # goal: find the least amount of intersection points within the boundary that are the "best"
    
    # step 1: among any set of outer nodes paired with a valid inner node, 
    # if it exists (the intersection point),
    # take the intersection node with the best angle score
    
    #maps subsets of outer nodes -> (score, inner node id)
    node_subset_map = {}
    for k in boundary.inner_nodes:
        currvecs = []
        inner_node_o = []
        nodes = []
        for u, vec, p in zip(outer_nodes, vecs, outer_points):
            #if shapeutils.segment_cover(G.nodes[u]["pos"], G.nodes[k]["pos"], mask, approx=False)==0:
            if not shapeutils.segment_blocked(G.nodes[u]["pos"], G.nodes[k]["pos"], background_pixels):
                currvecs.append(vec)
                inner_node_o.append(p)
                nodes.append(u)
        if not currvecs:
            continue
        score = shapeutils.angle_metric(np.array(G.nodes[k]["pos"]), np.array(inner_node_o), np.array(currvecs))
        if score > node_subset_map.get(tuple(nodes), (-10, -1))[0]:
            node_subset_map[tuple(nodes)] = (score, k)
            
    # step 2: now we have a collection of sets where each set is a 
    # subset of outer nodes that can intersect. 
    # we need to find subcollections such that all outer nodes are covered. 
    # This is similar to the exact cover problem, except that we allow duplicates.
    
    #maps inner node id -> valid outbound nodes list
    subset_candidates = {tup[1]: list(nodes) for nodes, tup in node_subset_map.items()}
    #save inner node id -> score
    scores = {tup[1]: tup[0] for tup in node_subset_map.values()}
    
    # for k, v in subset_candidates.items():
    #     G.nodes[k]["color"] = "red"
    #     [G.add_edge(k, outer) for outer in v]
    
    print("solving subset cover...")    
    res = list(subset_cover(set(outer_nodes), subset_candidates, 2))
    # res = list(solve_exact_cover(set(outer_nodes), subset_candidates))
    if not res:
        print("no solution found!")
        return False
    print("got", len(res), "valid assignments")
    
    # step 3: among the solutions, take the assignment that has 
    # the best angle score, all the while assuring each outer node has only 1 branch


    out_to_in_best = {}
    mx = 0
    for assignment in res:
        out_to_in = {}
        tmpgraph = nx.Graph()
        for inner_node_id in assignment:
            inner_node_p = np.array(pos[inner_node_id])
            #print( inner_node_id, subset_candidates[inner_node_id])
            for out in subset_candidates[inner_node_id]:
                score = float(
                    shapeutils.angle_metric(
                    np.array(inner_node_p), np.array([pos[out]]), np.array([vecmap[out]]))
                              )
                if out not in out_to_in:
                    out_to_in[out] = (score, inner_node_id)
                elif score > out_to_in[out][0]:
                    out_to_in[out] = (score, inner_node_id)
        total = sum([x[0] for x in out_to_in.values()])
        if total > mx:
            out_to_in_best = out_to_in
            mx = total
    # print("best")
    # print(out_to_in_best)
    #invert it to get in -> out nodes
    source_to_targets = {}
    for out_node, (_, in_node) in out_to_in_best.items():
        source_to_targets.setdefault(in_node, []).append(out_node)
    
    # print(source_to_targets)
    
    # step 4: now that we have the intersection nodes, 
    # compute shortest disjoint paths from the intersection nodes to their outer nodes,
    # as well a path between the intersection nodes.
    cluster = boundary.all_nodes()


    #add keys indicating we want paths connecting the 2 intersections
    
    
    # we have to densify the graph,
    # to improve likelihood of disjoint paths being found
    for k in source_to_targets:
        for u in get_neighbors_of_neighbors(G, k):
            if u in cluster:
                dx, dy = np.array(G.nodes[u]["pos"]) - np.array(G.nodes[k]["pos"])
                if dx != 0 and dy!= 0 and abs(dy) != abs(dx):
                    G.add_edge(k, u)
    planargraph_core.add_grid_edges(G, cluster)
    newnodes = planargraph_core.add_intersection_nodes_fast(G, node_subset=cluster)
    cluster |= set(newnodes)
    subgraph = G.subgraph(cluster)

    #remove_collinear_edges(G)
    
    
    all_paths = []
    success = False
    try_one = False
    if len(source_to_targets) == 1:
        try_one = True
    else:
        a, b = source_to_targets.keys()
        d = math.dist(pos[a], pos[b])
        print("distance between intersections", d)
        try_one = d <=5 and boundary.unique_degree() > 4
    # if both intersections are close, try to see if a single intersection is enough.
    if try_one:
        U = max(source_to_targets.keys(), key=lambda x : len(source_to_targets[x]))
        print("node with most branches", U, printpos(G.nodes[U]["pos"]))
        # planargraph_core.split_opposite_edges(G, U)


        source_to_targets2 = {U: boundary.outer_nodes()}
        result = solve_multi_source_greedy(subgraph, source_to_targets2)
        all_paths = list(chain.from_iterable(result.values()))
        success = validate_paths(G, boundary, all_paths, 1)
       
    if not success:
        a, b = source_to_targets.keys()
        source_to_targets[a].append(b)
        result = solve_multi_source_greedy(subgraph, source_to_targets)
        all_paths = list(chain.from_iterable(result.values()))
        success = validate_paths(G, boundary, all_paths, 2)
    
    if not success:
        G.remove_nodes_from(newnodes)
        return False
    
    def weight(u, v, d):
        a = G.nodes[u]["pos"]
        b = G.nodes[v]["pos"]
        return np.linalg.norm(np.array(a)-np.array(b))
    
    planargraph_core.optimize_disjoint_paths(G, all_paths, weight=weight)
    
    for path in all_paths:
        for k in path[1:]: G.nodes[k]["color"] = "lawngreen"
        
    for in_node, outer in source_to_targets.items():
        G.nodes[in_node]["color"] = "red"
        # for v in outer:
        #     G.add_edge(v, in_node)

    # if not success:
    #     return False
   
    
    #step 5: only keep the paths in the subgraph
    remove_all_except_paths(G, cluster, all_paths)
    return True





def process_general(G: nx.Graph, boundary: Boundary, boundary_idx: int, mask, background_pixels):
    #print("centroid:", boundary.centroid, boundary.unique_degree())
    outer_points = boundary.outer_points(G)
    assert len(outer_points) >0

    #step 1: compute all outbound vectors of outer nodes
    vecs = outbound_vectors(G, boundary)
    assert vecs.shape == outer_points.shape

    #step 2: grid sample on polygon inside
    boundary_polygon = boundary.boundary_polygon(G)
   
    grid_points= shapeutils.generate_grid_in_aabb(shapeutils.compute_aabb(boundary_polygon), 0.25)
    valid_points = shapeutils.points_in_polygon(boundary_polygon, grid_points)
    points = np.concatenate([grid_points[valid_points], boundary_polygon])
    assert points.shape == (sum(valid_points) + len(boundary_polygon), 2)

    #step 3: filter out points that would lead to edges not on the line mask
    def cond(p):
        return not any(shapeutils.segment_blocked(p, q, background_pixels) for q in outer_points)
        #return all(shapeutils.segment_cover(p, q, mask, approx=False)==0 for q in outer_points)
    def not_eq(p):
        #we filter these out because they would lead to duplicate point positions
        return p not in outer_points
    #with timer("valid points"):
    points = np.array([p for p in points if not_eq(p) and cond(p)])
    
    if len(points):
        #step 4: find best max angle point
        res, _ = shapeutils.best_point_angle_optimized(points, outer_points, vecs)
        
        remove_inside_of_boundary(G, boundary)
        
        collapse_boundary_to_point(G, boundary, res)
        return True
    
    print("boundary at centroid", boundary.centroid, "No single point for collapsing found")
    #return False
    return optimal_cluster_search(G, boundary, mask, background_pixels)

def resolve_boundaries(func, G: nx.Graph, boundaries: dict[int, Boundary], *args):
    to_rm = []
    for idx, b in boundaries.items():
        if func(G, b, idx, *args):
            to_rm.append(idx)
    for i in to_rm:
        del boundaries[i]

def resolve_to_medial_axis(G: nx.Graph, boundaries: dict[int, Boundary]):
    pos = nx.get_node_attributes(G, "pos")

    for idx, boundary in boundaries.items():
        #print(boundary.centroid)
        boundary_polygon = np.array([np.array(pos[u]) for u in boundary.nodedict.keys()])

        points, ridges = shapeutils.get_polygon_voronoi(
            boundary_polygon, densification_distance=0.3)
        H = annotated_voronoi_graph(points, ridges, boundary_polygon)
        
        #remove branches that do not point to outer nodes
        to_rm_indices = [idx for idx, v in enumerate(boundary.nodedict.values()) if not v]
        #print(to_rm_indices)
        to_rm = [k for k in H.nodes for i in to_rm_indices if H.nodes[k]["accessible"] == i]
        H.remove_nodes_from(to_rm)
        
        #the outer nodes are duplicated in H. remove them and add an edge to G.
        to_rm_indices = {idx: k  for idx, (k, v) in enumerate(boundary.nodedict.items()) if v}
        to_rm =[]
        edges_to_add=[]
        for i, v in to_rm_indices.items():
            tmp = [x for x in H.nodes if H.degree(x) == 1 and H.nodes[x]["accessible"] == i]
            if tmp:
                #print("branch node: ", tmp)
                to_rm.append(tmp[0])
                #by construction it must have exactly 1 neighbor
                edges_to_add.append(
                    (v, next(H.neighbors(tmp[0])))
                )
            else:
                #this means there is no branch to this outer node. 
                # search for the closest point instead
                p = np.array([H.nodes[x]["pos"] for x in H.nodes])
                idx_to_node = {i:x for i, x in enumerate(H.nodes)}
                _, res, _ = shapeutils.find_closest_furthest_point(p, np.array(G.nodes[v]["pos"]))
                #print(len(p))
                #print(idx_to_node)
                #print(res)
                edges_to_add.append((v, idx_to_node[int(res)]))
            
        H.remove_nodes_from(to_rm)
                
        
        remove_inside_of_boundary(G, boundary)
        #remove boundary as well
        G.remove_nodes_from(k for k, v in boundary.nodedict.items() if not v )
        #at last, add the modified medial axis
        add_subgraph_to_G(G, H, G_to_H_edges=edges_to_add)
        

def remove_spurious_short_paths(G: nx.Graph, len_threshold = 2):
    nodes_to_remove = []
    for u in G.nodes:
        if G.degree[u] <=2:
            continue
        for v in G.neighbors(u):
            p = has_degree1_path(G, u, v)
            if p is None:
                continue
            if len(p) <= 2:
                nodes_to_remove.extend(p[1:])
    G.remove_nodes_from(nodes_to_remove)

def spurious_path_condition(path: list[int], G, img, len_thresh=5.8, img_thresh=50):
    def space_dist(path):
        p = np.array([G.nodes[u]["pos"] for u in path])
        p, i, d = shapeutils.find_closest_furthest_point(p, p[0], closest=False)
        return i, d
    
    v, d = space_dist(path)
    #print("distance", d)
    if d < len_thresh:
        spos = np.array([G.nodes[path[0]]["pos"]])
        tpos = np.array([G.nodes[path[-1]]["pos"]])
        sval = shapeutils.grid_sample(img, spos, mode="min shifted")[0]
        tval = shapeutils.grid_sample(img, tpos, mode="max shifted")[0]
        #print("source, target vals", sval, tval)
        #remove direct edges in any case
        if tval < img_thresh  or len(path) ==2:
            # print("removed path")
            return True
    return False
        
def remove_spurious_paths(G: nx.Graph, img, thresh = 50):
   

    nodes_to_remove = []
    for u in G.nodes:
        if G.degree[u] <=2:
            continue
        for v in G.neighbors(u):
            p = has_degree1_path(G, u, v)
            if p is None:
                continue
            if spurious_path_condition(p, G, img):
                nodes_to_remove.extend(p[1:])
    
    G.remove_nodes_from(nodes_to_remove)
                
        
def find_grid_quads_with_adjacency(G: nx.Graph):
    """
    Finds sets of four nodes (a, b, c, d) that are on an integer grid
    AND are connected by edges in the graph.
    """
    # 1. Map (x, y) -> node_index for all nodes at integer positions
    pos_to_node = {}
    for n, data in G.nodes(data=True):
        pos = data.get('pos')
        if pos:
            x, y = pos
            if float(x*2).is_integer() and float(y*2).is_integer():
                pos_to_node[(float(x), float(y))] = n

    quads = []

    # 2. Iterate through nodes, treating each as the "bottom-left" corner
    for (x, y), node_a in pos_to_node.items():
        # Target coordinates for the other corners
        quadpattern = [
                ((x + 1, y), (x + 1, y + 2), (x, y + 2)), 
                ((x + 2, y), (x + 2, y + 1), (x, y + 1)),
                
                ((x + 1, y), (x + 1, y + 2), (x, y + 1)), # \|
                ((x + 1, y+1),(x + 1, y + 2), (x, y + 2)), # |\
                ((x, y+2),(x -1, y + 2), (x-1, y + 1)),  # /|
                ((x, y+2),(x +1, y + 1), (x+1, y)),  #  |/
                ((x+1, y),(x +2, y + 1), (x, y+1)), # -\
                ((x+2, y),(x +1, y + 1), (x, y+1)), # -/
                ((x+2, y),(x +2, y + 1), (x+1, y+1)), # \-
                ((x+1, y),(x +1, y + 1), (x-1, y+1)), # /-
                ((x+1.5, y+0.5),(x +1.5, y + 1.5), (x+0.5, y+1.5)),
                ((x-1.5, y+0.5),(x -1.5, y + 1.5), (x-0.5, y+1.5)),
                      ]
        
        for p_b, p_c, p_d in quadpattern:
            # 3. Spatial Check: Do nodes exist at these grid positions?

            if p_b in pos_to_node and p_c in pos_to_node and p_d in pos_to_node:
                node_b = pos_to_node[p_b]
                node_c = pos_to_node[p_c]
                node_d = pos_to_node[p_d]
                
                # 4. Adjacency Check: Are they connected in a cycle?
                # We check the four edges of the square
                has_all_edges = (
                    G.has_edge(node_a, node_c) and
                    G.has_edge(node_b, node_d)
                )
                if has_all_edges:
                    quads.append((node_a, node_b, node_c, node_d))

    return quads

def prune_edges(G: nx.Graph):
    squares = find_grid_quads_with_adjacency(G)
    for a, b, c, d in squares:
        x1, y1 = G.nodes[a]["pos"]
        x2, y2 = G.nodes[b]["pos"]
        x3, y3 = G.nodes[c]["pos"]
        x4, y4 = G.nodes[d]["pos"]
        if G.has_edge(a, c) and abs(x1-x3) + abs(y1-y3) > 2: G.remove_edge(a, c)
        if G.has_edge(b, d) and abs(x2-x4) + abs(y2-y4) > 2: G.remove_edge(b, d)
    

def set_cluster_att(G: nx.Graph, clusters):
    for n in G.nodes:
        G.nodes[n]['cluster'] = []
    for idx, cluster in enumerate(clusters):
        for node in cluster['nodes']:
            G.nodes[node]['cluster'].append(idx+1)

def color_clusters(G: nx.Graph, clusters, color=None):
    """
    Assign a color attribute to all nodes in each cluster.
    
    Parameters:
    - G: NetworkX graph
    - clusters: list of sets of node indices
    - color: string or Matplotlib color code
    """
    colors = ['#FFA500', 'sienna', 'blue', 'green', 'orchid']
    for idx, cluster in enumerate(clusters):
        currcolor = colors[idx%len(colors)] if not color else color
        for node in cluster['nodes']:
            G.nodes[node]['color'] = currcolor

def improve_sharp_tips(G: nx.Graph):
    pos = nx.get_node_attributes(G, "pos")
    pairs = planargraph_core.find_degree1_degree3_pairs(G)
    nodes_to_remove = []
    visited = set()
    for d in pairs:
        path =  d["path"]
        if path[-1] in visited:
            continue
        if len(path) > 50:
            continue
        left, right = d["other_neighbors"]
        leftnext = [x for x in G.neighbors(left) if x != path[-1]]
        rightnext = [x for x in G.neighbors(right) if x != path[-1]]
        v1 = get_outbound_vector(left, leftnext, G)
        v2 = get_outbound_vector(right, rightnext, G)
        dot_val = np.dot(v1, v2)
        if dot_val <= 0.2:
            # print("continuing")
            continue
        p = np.array(pos[path[-1]])
        
                    
        
        
        
        # Compute geodesic length of path
        geodesic_length = sum(
            np.linalg.norm(np.array(pos[path[i+1]]) - np.array(pos[path[i]]))
            for i in range(len(path) - 1)
        )
        if dot_val <0.65 and geodesic_length > 10:
            # print("continuing")
            continue
        inner_v1 = np.array(pos[left]) - p
        inner_v2 = np.array(pos[right]) - p
        inner_v3 = np.array(pos[path[-2]]) - p
        if np.linalg.norm(inner_v1) > 2 or np.linalg.norm(inner_v2) > 2:
            inner_v2 /= np.linalg.norm(inner_v2)
            inner_v1 /= np.linalg.norm(inner_v1)
            inner_v3 /= np.linalg.norm(inner_v3)
            if np.dot(inner_v1, inner_v3) < -0.9 or np.dot(inner_v2, inner_v3) < -0.9:
                if abs(np.dot(inner_v1, inner_v2)) < 0.55:
                    continue
        pol = np.array([pos[x] for x in path])
        pol_smooth = shapeutils.smooth_polyline_constrained(pol, d=1)
        turnangle = shapeutils.total_unsigned_turning(pol_smooth)
        #print("center", pos[path[-1]])
        #print("angle", turnangle, math.degrees(turnangle))
        if turnangle > math.pi / 2:
            continue
        G.nodes[left]["info"] = f"dot={dot_val:.4f} geodesic={geodesic_length:.4f}"
        
        #path is ordered from tip to inside
        pathpol = pol_smooth[1:]
        #maxdist = math.dist(pos[left], pos[right])
        deltas = np.linspace(0.08, 1, len(pathpol)+1)[:-1]
        leftline, rightline = shapeutils.offset_polyline(pathpol, delta = deltas)
        complete = np.concatenate([leftline[::-1], [pos[path[0]]], rightline])
        r, _, _ = shapeutils.segment_intersection(pos[left], complete[0], pos[right], complete[-1])
        if r is None:
            outside_connect = [(left, 0), (right, len(complete)-1)]
        else:
            outside_connect = [(left, len(complete)-1), (right, 0)]
        oc_v1 = complete[outside_connect[0][1]] - pos[left]
        oc_v2 = complete[outside_connect[1][1]] - pos[right]
        oc_v1/=np.linalg.norm(oc_v1)
        oc_v2/=np.linalg.norm(oc_v2)
        print("\ncenter", pathpol[-1])
        print("dp1, len", dot_val, geodesic_length)
        print("dp2", np.dot(oc_v1, oc_v2))
        print("dpa", np.dot(inner_v1, inner_v2))
        print("dpb", np.dot(inner_v2, inner_v3))
        print("dpc", np.dot(inner_v3, inner_v1))
        if np.dot(oc_v1, oc_v2) <= 0:
            continue
        leftlinev = pol[0] - complete[outside_connect[0][1]]
        rightlinev = pol[0] - complete[outside_connect[1][1]]
        leftlinev/=np.linalg.norm(leftlinev)
        rightlinev/=np.linalg.norm(rightlinev)
        print("dp left", np.dot(oc_v1, leftlinev))
        print("dp right", np.dot(oc_v2, rightlinev))
        #60 deg
        if np.dot(oc_v1, leftlinev) <= 0.5 or np.dot(oc_v2, rightlinev) <= 0.5:
            continue
        planargraph_core.add_path_subgraph(G, complete, G_to_new_edges=outside_connect)
        nodes_to_remove.extend(path)
        visited.add(path[-1])
    G.remove_nodes_from(nodes_to_remove)