# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/graph_extraction/build_linegraph.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
#   - `a[*idx]` (Python 3.11 syntax) -> `a[tuple(idx)]`
#   - `plotutils` import dropped (debug_dir-only SVG plots)
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

import cv2
import numpy as np
from . import planargraph_core
plu = None  # upstream's debug SVG plotting is not vendored; only reached with a debug_dir
from . import graph_clusters
from . import shapeutils
from . import image
import networkx as nx
from skimage.graph import pixel_graph
from skimage.morphology import skeletonize
import math
from .. import _utils as utils

import logging as _logging
_LOG = _logging.getLogger("raster_parser.involution_stroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))



def primitive_graph(binary_img: np.ndarray) -> nx.Graph:
    #we use connectivity=2 for 8-neighborhood
    graph, nodes = pixel_graph(binary_img.astype(bool), connectivity=2)
    G = nx.from_scipy_sparse_array(graph)

    coords = np.column_stack(np.unravel_index(nodes, binary_img.shape))

    # Attach pixel coordinates as node attributes
    for i, coord in enumerate(coords):
        G.nodes[i]["pos"] = tuple(coord+0.5)[::-1]
    
    return G
    # graph_clusters.remove_spurious_short_paths(G, len_threshold=3)
    
    # deg3_nodes = {n for n, deg in G.degree() if deg != 2}
    
    # msk = np.zeros_like(binary_img, dtype=np.uint8)
    # for i in deg3_nodes:
    #     coord = G.nodes[i]["coord"] 
    #     msk[coord] = 255
    # return msk
        
    
        
    
def reconnect_gaps(binary):
    bin4_inv = 255-binary
    kernel = image.circ(1) 
    b4a, _ = image.detect_and_mark(bin4_inv, kernel, kernel)
    bin4_inv = cv2.subtract(bin4_inv, b4a)
    small = image.remove_components_area(bin4_inv, 5, keepsmall=True)
    return small

def cleanup(binary):
    h = image.detect_holes(binary, 4 , connectivity_8=True)
    bin_cleaned = cv2.bitwise_or(binary, h)
    h = image.detect_holes(bin_cleaned, 2, connectivity_8=False)
    bin_cleaned = cv2.bitwise_or(bin_cleaned, h)
    bin_cleaned = image.remove_spurious_spikes(bin_cleaned)
    return bin_cleaned

def cleanup2(binary):
    h = image.detect_holes(binary, 1, connectivity_8=False)
    bin_cleaned = cv2.bitwise_or(binary, h)
    bin_cleaned = image.remove_holes_by_kernel(bin_cleaned, image.circ(1), connectivity_8=False)
    bin_cleaned = image.remove_spurious_spikes(bin_cleaned)
    return bin_cleaned

def get_binary(img, debug=False):
    img= img.copy()
    img[img < 10] = 0
    binary = image.thresh_by_max_component_intensity(img, 50)
    if debug:
        cv2.imwrite("debug_files/binary_raw.png", binary)
    hole_img = image.detect_holes(binary, maxarea=2)
    binary = cv2.bitwise_or(binary, hole_img)
    hole_img2 = image.detect_holes(binary, maxarea=4, connectivity_8=True)
    binary = cv2.bitwise_or(binary, hole_img2)

    if debug:
        cv2.imwrite("debug_files/binary_1.png", binary)
    
    binary_new = image.remove_spurious_spikes(binary)
    if debug:
        cv2.imwrite("debug_files/binary_adjusted.png", binary_new)
    return binary_new


def get_binary_2(img: np.ndarray):
    img = img.copy()
    img[img<10] = 0
    binary = image.thresh_by_max_component_intensity(img, 50)
    adjusted = image.adaptive(img, size=9)
    adjusted = 255-adjusted
    binary2 = cv2.subtract(binary, adjusted)
    #remove insignificant regions (noise)
    binary3 = image.thresh_by_max_component_intensity(img, 50, binary2)
    
    # goal: find regions around intersections. otherwise, lines tend to be disconnected
    bin_dilated = cv2.dilate(binary3, image.circ(2))
    kernel = image.circ(5)
    r1, _ = image.detect_and_mark(bin_dilated, kernel, kernel)
    #and it with original binary to not add too much
    r2 = cv2.bitwise_and(binary, r1)
    #add it back
    binary4 = cv2.bitwise_or(binary3, r2)
    
    # Binary4 is already pretty good. It has the tendency to remove too much though.
    # Approach: see what we've removed from the original, and actually only remove
    # the large regions. 
    diff =cv2.subtract(binary, binary4)
    diff2 = image.remove_components_by_kernel(diff, np.ones((3, 3)))
    binary5 = cv2.bitwise_or(binary4, diff2)
    
    #clean up by removing small holes and spikes
    h = image.detect_holes(binary5, 4 , connectivity_8=True)
    bin_cleaned = cv2.bitwise_or(binary5, h)
    bin_cleaned = image.remove_spurious_spikes(bin_cleaned)
    return bin_cleaned


def get_binary_3(img: np.ndarray):
    img = img.copy()
    img[img<10] = 0
    binary = image.thresh_by_max_component_intensity(img, 50)
    adjusted = image.adaptive(img, size=9)
    adjusted = 255-adjusted
    binary2 = cv2.subtract(binary, adjusted)
    #remove insignificant regions (noise)
    binary3 = image.thresh_by_max_component_intensity(img, 50, binary2)
    
    holes_bin_3 = image.detect_holes(binary3, maxarea=50, connectivity_8=True)
    
    # goal: find regions around intersections. otherwise, lines tend to be disconnected
    bin_dilated = cv2.dilate(binary3, image.circ(2))
    kernel = image.circ(4)
    r1, detected = image.detect_and_mark(bin_dilated, kernel, kernel)
    
    detected2 = image.remove_components_by_kernel(detected, image.circ(1), inverse=True)
    r1_2 = cv2.dilate(detected2, kernel)

    #and it with binary to not add too much
    r2 = cv2.bitwise_and(binary, r1_2)
    #add it back
    binary4 = cv2.bitwise_or(binary3, r2)
    
    #subtract holes again
    binary4 = cv2.subtract(binary4, holes_bin_3)
    
    gaps = reconnect_gaps(binary4)
    gaps = cv2.bitwise_and(gaps, binary)
    binary5 = cv2.bitwise_or(binary4, gaps)
    
    return cleanup(binary5)

def get_binary_4(img: np.ndarray):
    T1, T2=10, 50
    T1, T2=25, 85
    img = img.copy()
    img[img<T1] = 0
    binary = image.thresh_by_max_component_intensity(img, T2)
    adjusted = image.adaptive(img, size=9)
    adjusted = 255-adjusted
    binary2 = cv2.subtract(binary, adjusted)
    #remove insignificant regions (noise)
    binary3 = image.thresh_by_max_component_intensity(img, T2, binary2)
    binary3 = image.remove_strictly_internal_components(binary3, binary)
    return cleanup2(binary3)

def to_mask(binary, coords):
    nodemsk = np.zeros_like(binary, dtype=np.uint8)
    for c in coords:
        coord_yx = np.array(c[::-1]).astype(np.int32)
        nodemsk[tuple(coord_yx)] = 255
    return nodemsk


def build_graph_from_binary(binary):
    
    #Detect image patches with kernels. They define graph points.
    # Hit-or-Miss kernel must contain: 1 (foreground), -1 (background), 0 (don't care)
    kernels = [
        # (np.array([[1, -1], [-1, 1]]), (0, 0), [(-0.5, -0.5), (0.5, 0.5)], False, None),
        # (np.array([[-1, 1], [1, -1]]), (0, 0), [(0.5, -0.5), (-0.5, 0.5)], False, None),
        # (np.array([[-1, 1, 1],
        #            [ 1, 1, 1],
        #            [ 1, 1,-1]], np.int8), (0, 0), [(0, 1), (1, 0)], True, None), 
        
        #  (np.array([[1, 1,-1],
        #             [1, 1, 1],
        #            [-1, 1, 1]], np.int8), (0, 0), [(0, 0), (1, 1)], True, None), 
        # (np.ones((2, 2), np.int8), (1, 1), [(0, 0)], True, None),
        # (np.ones((1, 2), np.int8), (0, 1), [(0, 0.5)], True, None),  
        # (np.ones((2, 1), np.int8), (1, 0), [(0.5, 0)], True, None),  
    ]

    binary_work = binary.copy()
    all_centers = []
    for i, (hm_kernel, shift, offsets, do_subtract, replace_kernel) in enumerate(kernels, start=1):

        mask, detected = image.detect_and_mark(binary_work, hm_kernel, replace_kernel)

        # collect centers (where the pattern matched)
        # The Hit-or-Miss result is 255 at the anchor point of the match
        ys, xs = np.where(detected == 255)
        centers = [(float(x) + off[0], float(y) + off[1]) for x, y in zip(xs, ys) for off in offsets]
        centers = [(x, y) for x, y in centers if x >0.5 and y>0.5 and x < binary.shape[0]-1 and y < binary.shape[1]-1]
        all_centers.extend(centers)
        if do_subtract:
            binary_work = cv2.subtract(binary_work, mask)
        # if debug_dir:
            # cv2.imwrite(f"{debug_dir}/mask{i}.png", mask)

    # add remaining single pixels
    ys, xs = np.where(binary_work == 255)
    centers = [(float(x)+0.5, float(y)+0.5) for x, y in zip(xs, ys)]
    all_centers.extend(centers)

    print("building graph...")
    all_centers_set = set(all_centers)
    g = graph_clusters.build_graph_fast(list(all_centers_set), binary)
    graph_clusters.remove_collinear_edges(g)
    graph_clusters.prune_edges(g)
    planargraph_core.add_intersection_nodes_fast(g)
    
    return g


def build_graph_from_binary2(binary: np.ndarray):
    
    g = primitive_graph(binary)
    planargraph_core.add_intersection_nodes_fast(g)
    return g
    

def get_graph_G1(img, binary, debug_dir="") -> nx.Graph:
    #g = build_graph_from_binary(binary)
    g :nx.Graph = primitive_graph(binary)
    graph_clusters.remove_spurious_short_paths(g, len_threshold=2)

    # Phase 3: get clusters and boundaries of clusters

    print("getting clusters")
    clusters = graph_clusters.find_triangle_clusters(g)
    
    # graph_clusters.clusters_check(g, clusters)
    # graph_clusters.set_cluster_att(g, clusters)
    # graph_clusters.remove_degree_2(g, clusters)
    # graph_clusters.color_clusters(g, clusters)
    
            
        
    #planargraph_core.add_subgraph_with_pos()
    if debug_dir:
        print("saving raw graph")
        plu.plot_img_graph(binary, g, f"{debug_dir}/raw_graph1.svg", max_gray="#9D9D9D")
    # exit()
    # boundaries = graph_clusters.get_boundaries(g, clusters)
    # graph_clusters.consistency_check(g, boundaries)
    # print("num of boundaries", len(boundaries))
    
    # graph_clusters.resolve_degree_1_boundaries(g, boundaries, img)
    # graph_clusters.resolve_degree_2_boundaries(g, boundaries, img)
    # graph_clusters.remove_spurious_paths(g, img)
    # clusters = graph_clusters.find_triangle_clusters(g)
    # graph_clusters.clusters_check(g, clusters)
    # graph_clusters.color_clusters(g, clusters)
    # graph_clusters.set_cluster_att(g, clusters)
    # boundaries = graph_clusters.get_boundaries(g, clusters)
    
    # # Phase 4: cluster resolution
    # graph_clusters.resolve_degree_2_boundaries(g, boundaries, img)

    return g

def get_better_binary(img, binary, G1, debug_dir = ""):
    
    nodes = {G1.nodes[v]["pos"] for v, deg in G1.degree() if deg > 2}
    nodemsk = to_mask(binary, nodes)
    
   
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/nodemask.png", nodemsk) 
    
    mskdilated = cv2.dilate(nodemsk, np.ones((5, 5), dtype=np.uint8))
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/nodemask2.png", mskdilated)
    
    simple_bin = image.simple_thresh(img, 10)
    tmp = cv2.bitwise_and(simple_bin, mskdilated)
    binarynxt = cv2.bitwise_or(binary, tmp)

    # term = {v: deg  for v, deg in G1.degree() if deg == 1}
    # coords = []
    # for v, deg in term.items():
    #     u = next(G1.neighbors(v))
    #     vec = graph_clusters.edge_vector(v, u, G1)
    #     p1 = np.array(G1.nodes[v]["pos"])
    #     p2 = p1 + -2 * vec
    #     planargraph_core.add_subgraph_with_pos(G1, np.array([p1, p2]), [(0, 1)])
    #     pixel_coords = shapeutils.segment_pixels(p1, p2, approx=True)
    #     coords.extend(p for p in pixel_coords if p[0] < simple_bin.shape[0] and p[1] < simple_bin.shape[1]  )
    # msk_term = np.zeros_like(simple_bin)
    # if len(coords):
    #     coordsnp = np.array(coords)
    #     msk_term[coordsnp[:, 1], coordsnp[:, 0]] =255
    
    # msk_dil = cv2.dilate(msk_term, np.ones((3, 3), dtype=np.uint8))
    # msk_toadd = cv2.bitwise_and(msk_dil, simple_bin)
    
    # if debug_dir:
    #     cv2.imwrite(f"{debug_dir}/simplebin.png", simple_bin)
    #     cv2.imwrite(f"{debug_dir}/nodemask3.png", tmp)
    #     cv2.imwrite(f"{debug_dir}/term_msk.png", msk_term)
    #     cv2.imwrite(f"{debug_dir}/term_toadd.png", msk_toadd)
        
    #binarynxt = cv2.bitwise_or(binarynxt, msk_toadd)
    holes_bin = image.detect_holes(binary, maxarea=50, connectivity_8=True)
    # if debug_dir:
    #      cv2.imwrite(f"{debug_dir}/holes.png", holes_bin)
    holes_bin2 = image.detect_holes(binary, maxarea=25, connectivity_8=False)
    # if debug_dir:
    #      cv2.imwrite(f"{debug_dir}/holes2.png", holes_bin2)
    binarynxt = cv2.subtract(binarynxt, holes_bin)
    binarynxt = cv2.subtract(binarynxt, holes_bin2)
    binarynxt = cleanup2(binarynxt)
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/binarynxt.png", binarynxt)
    
    return binarynxt

def process(img, invert=False, debug_dir="", sharp = False, deg3 = False, deg4 = True):
    
    assert len(img.shape) == 2
    if invert:
        img = 255-img

    #phase 1
    binary = get_binary_4(img)
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/binary.png", binary)
        # sk = image.skeleton(binary)
        # cv2.imwrite(f"{debug_dir}/medial.png", sk)

    #phase 2 build graph G1
    binary =  image.skeleton(binary)
    G1 = get_graph_G1(img, binary, debug_dir=debug_dir)
    binarynxt = get_better_binary(img, binary, G1, debug_dir=debug_dir)
    
    with utils.timer("graph construction"):
        G = build_graph_from_binary2(binarynxt)
    graph_clusters.shorten_endpoints(G, img)
    with utils.timer("cluster find"):
        clusters = graph_clusters.find_triangle_clusters(G)
    graph_clusters.clusters_check(G, clusters)
    graph_clusters.color_clusters(G, clusters)
    graph_clusters.set_cluster_att(G, clusters)
    with utils.timer("get boundaries"):
        boundaries = graph_clusters.get_boundaries(G, clusters, debug= True)
    print("there are", len(boundaries), "boundaries")
    if debug_dir:
        print("saving raw graph")
        # debug: draw each boundary's outbound vectors as short edges so the
        # directions are visible in raw_graph.svg
        VEC_LEN = 5
        for boundary in boundaries.values():
            if not boundary.outer_nodes():
                continue
            vecs = graph_clusters.outbound_vectors(G, boundary)
            for p, vec in zip(boundary.outer_points(G), vecs):
                planargraph_core.add_path_subgraph(G, np.array([p, p + VEC_LEN * vec]))
        #plu.plot_img_graph(255-binarynxt, G, f"{debug_dir}/raw_graph.svg", color="black", max_gray="white", min_black="#BFBFBF")
        plu.plot_img_graph(binarynxt, G, f"{debug_dir}/raw_graph.svg", color="white", max_gray="#9D9D9D")
    graph_clusters.resolve_boundaries(graph_clusters.remove_degree_0_boundary, G, boundaries)
    print("process degree 2")
    graph_clusters.resolve_degree_2_boundaries(G, boundaries, img)
    print("process degree 1")
    graph_clusters.resolve_degree_1_boundaries(G, boundaries, img)
    print("process degree 3 and 4")
    if deg4:
        graph_clusters.resolve_boundaries(graph_clusters.process_special_degree_4, G, boundaries, img)
    if deg3:
        graph_clusters.resolve_boundaries(graph_clusters.process_special_degree_3, G, boundaries, img)
    
    background_pixels = set(zip(*np.where(binarynxt== 0)))
    #with utils.timer("general case"):
    graph_clusters.resolve_boundaries(graph_clusters.process_general, G, boundaries, binarynxt, background_pixels)
     # #all remaining boundaries are replaced by the medial axis
    graph_clusters.resolve_to_medial_axis(G, boundaries)
    
    graph_clusters.remove_spurious_paths(G, img)
    #graph_clusters.process_pairs_of_deg_3(G, img)
    if sharp:
        graph_clusters.improve_sharp_tips(G)
    if debug_dir:
        print("saving result graph")
        plu.plot_img_graph(img, G, f"{debug_dir}/result_overlay.svg", edge_width=0.05, color="white")
        #plu.plot_img_graph(255-img, G, f"{debug_dir}/result_overlay.svg", edge_width=0.05, color="black", max_gray="white", max_black="#BFBFBF")
    return G, True







def compute_polylines(g: nx.Graph) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    
    paths: list[list[int]] = planargraph_core.decompose_into_paths(g)
    
    pos = nx.get_node_attributes(g, "pos")
    polylines = []
    connections =[]
    for p in paths:
        tup = p[0], p[-1]
        polylines.append(np.array([pos[x] for x in p]))
        connections.append(tup)
    
    return polylines, connections


def get_smoothed_lines(lines: list[np.ndarray], d):
    return [shapeutils.smooth_polyline_constrained(pl, d) for pl in lines]

def contracted_graph(g: nx.Graph):
    H = g.copy()
    to_rm = [v for v, d in H.degree() if d ==2 ]
    H.remove_nodes_from(to_rm)
    return H

    