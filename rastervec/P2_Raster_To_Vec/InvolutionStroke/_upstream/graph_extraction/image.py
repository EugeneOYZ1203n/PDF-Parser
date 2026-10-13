# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/graph_extraction/image.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
#   - `np.bool` -> `np.bool_` (numpy 1.x)
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

import cv2
import numpy as np
from skimage.morphology import skeletonize

import logging as _logging
_LOG = _logging.getLogger("rastervec.P2.InvolutionStroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))



def simple_thresh(img, t):
    _, thresh = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    return thresh

def detect_and_mark(binary: np.ndarray, hm_kernel, replace_kernel = None):
    
    detected = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, hm_kernel)
    
    shift = hm_kernel.shape[0]%2==0, hm_kernel.shape[1]%2==0
    shifted = np.zeros_like(detected)
    if shift is not None:
        dy, dx = shift
        # Safety check for array slicing
        h, w = detected.shape
        shifted[:max(0, h-dy), :max(0, w-dx)] = detected[max(0, dy):, max(0, dx):]
    else:
        shifted = detected.copy()
        
    if replace_kernel is None:
        replace_kernel = np.where(hm_kernel == 1, 1, 0)
    replace_kernel = replace_kernel.astype(np.uint8)
    mask = cv2.dilate(shifted, np.flip(replace_kernel))
    return mask, detected

def remove_components_area(binary_img, K, keepsmall = False):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]

    keep = areas <= K if keepsmall else areas>=K
    keep[0] = False  # background

    mask = keep[labels]
    return (mask.astype(np.uint8) * 255)

def circ(radius: int):
    size=2*radius+1
    kernel = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel

def remove_components_by_kernel(binary: np.ndarray, kernel, inverse = False, connectivity_8=True):
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8 if connectivity_8 else 4)
    _, detected = detect_and_mark(binary, kernel)
    pixels = np.where(detected>0)
    components = np.unique([labels[pixels]])
    msk = np.zeros_like(labels, dtype=np.bool_)
    for k in components:
        msk |= labels == k
    cp = binary.copy()
    cp[msk]=0
    if inverse:
        return binary - cp
    return cp

def remove_holes_by_kernel(binary: np.ndarray, kernel, keep_marked=True, connectivity_8=True):
    
    return 255- remove_components_by_kernel(
        255-binary, kernel, inverse=keep_marked, connectivity_8=connectivity_8)


def detect_holes(img, maxarea=2, connectivity_8=False):
    # 1. Find connected components on the inverted image
    # Note: Using 255 - img is faster than bitwise_not for binary images
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - img, 
        connectivity=8 if connectivity_8 else 4
    )

    # If only background is found, return empty
    if num_labels <= 1:
        return np.zeros_like(img)

    # 2. Extract stats into arrays (vectorized)
    # stats indices: 0:x, 1:y, 2:width, 3:height, 4:area
    areas = stats[:, cv2.CC_STAT_AREA]
    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]

    # 3. Create a boolean mask of "valid" labels
    # A label is a hole if:
    # - It is not label 0 (background)
    # - It does not touch the image borders
    # - Its area is <= maxarea
    h_img, w_img = img.shape
    keep = (areas <= maxarea) & \
           (x > 0) & \
           (y > 0) & \
           ((x + w) < w_img) & \
           ((y + h) < h_img)
    
    # Label 0 is always the background of the inverted image, so we ignore it
    keep[0] = False

    # 4. Apply the filter using a Look-Up Table (LUT)
    # We create an array where the index is the label and the value is the output color
    lut = np.zeros(num_labels, dtype=np.uint8)
    lut[keep] = 255

    return lut[labels]


def adaptive(img, size=3):
    adjusted = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, 1)
    return adjusted



def remove_strictly_internal_components(mask1, mask2):
    """
    Removes components from mask1 that are strictly within a component of mask2 
    (i.e., fully contained and not touching the mask2 border).
    
    Args:
        mask1: Binary mask (uint8) containing components to filter.
        mask2: Binary mask (uint8) acting as the container.
        
    Returns:
        filtered_mask: A copy of mask1 with internal components removed.
    """
    # 1. Prepare the container mask
    # We erode mask2. If a mask1 component fits entirely inside the eroded mask, 
    # it means it was strictly inside the original mask2 without touching borders.
    kernel = np.ones((3, 3), np.uint8)
    mask2_eroded = cv2.erode(mask2, kernel, iterations=1)

    # 2. Find connected components in mask1
    # stats returns [x, y, width, height, area] for each label
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1, connectivity=8)

    # Create a copy of mask1 to modify
    output_mask = mask1.copy()

    # 3. Iterate through each component (skip label 0, which is background)
    for i in range(1, num_labels):
        # Extract the specific component's bounding box to speed up processing
        x, y, w, h, area = stats[i]
        
        # Isolate the component in the labels map (within the bounding box)
        component_slice = labels[y:y+h, x:x+w]
        
        # Create a binary mask for just this component
        comp_mask = (component_slice == i).astype(np.uint8) * 255
        
        # Extract the corresponding area from the eroded mask2
        eroded_slice = mask2_eroded[y:y+h, x:x+w]
        
        # Calculate intersection
        # We use bitwise_and to find where the component overlaps with the SAFE ZONE (eroded mask)
        intersection = cv2.bitwise_and(comp_mask, eroded_slice)
        
        # 4. Check condition
        # If the area of intersection equals the area of the component, 
        # the component is fully inside the eroded region.
        intersection_area = cv2.countNonZero(intersection)
        component_area = cv2.countNonZero(comp_mask)

        if intersection_area == component_area:
            # It is strictly inside, so we remove it from the output
            # We set the pixels corresponding to this component to 0
            output_mask[y:y+h, x:x+w][comp_mask == 255] = 0

    return output_mask



def thresh_by_max_component_intensity(img, T, binary = None):
    """
    Finds connected components in a grayscale image and removes those where 
    the maximum original pixel value within the component is less than T.
    
    Args:
        img (np.ndarray): Grayscale input image.
        T (int/float): Intensity threshold.
        
    Returns:
        np.ndarray: Binary image (0 or 255) containing only the valid components.
    """
    if binary is None:
        # 1. Threshold: sets everything > 0 to 255
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        
    # 2. Find connected components
    # num_labels is the number of components (including background)
    # labels is an image where each pixel has the ID of its component
    num_labels, labels = cv2.connectedComponents(binary)
    
    # 3. Prepare an empty output image
    output = np.zeros_like(img, dtype=np.uint8)
    
    # 4. Iterate through components (label 0 is the background, so we skip it)
    for i in range(1, num_labels):
        # Create a mask for the current component
        component_mask = (labels == i)
        
        # Find the maximum value in the original image within this component
        max_val = np.max(img[component_mask])
        
        # 5. If the max value meets the threshold, draw it on the output
        if max_val >= T:
            output[component_mask] = 255
            
    return output


def remove_components_with_holes(binary_img):
    """
    binary_img: 0/255 uint8 image (foreground = 255)
    returns: cleaned binary image
    """

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        binary_img,
        cv2.RETR_CCOMP,   # Gives two-level hierarchy (outer + holes)
        cv2.CHAIN_APPROX_SIMPLE,
        
    )

    if hierarchy is None:
        return binary_img.copy()

    hierarchy = hierarchy[0]

    output = np.zeros_like(binary_img)

    for i, contour in enumerate(contours):

        parent = hierarchy[i][3]
        child = hierarchy[i][2]

        # We only want outer contours
        if parent == -1:
            # If outer contour has NO child -> no holes
            if child == -1:
                cv2.drawContours(output, contours, i, 255, thickness=-1)

    return output

def remove_spurious_spikes(binary):

    assert len(binary.shape) == 2
    
    kernel = np.array([[-1, -1, -1],
                       [-1, 1, -1],
                        [1, 1,  1]])
    
    detected = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
    for i in range(0, 3):
        kernel = np.rot90(kernel)
        detected_curr = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
        detected = cv2.bitwise_or(detected, detected_curr)
   
    #cv2.imwrite("debug_files/detect_spurious.png", detected)
    return binary-detected


def skeleton(img_binary):
    img_skeleton = skeletonize(img_binary.astype(bool))

    # Convert skeleton back to uint8 for OpenCV operations
    img_skeleton_uint8 = (img_skeleton.astype(np.uint8)) * 255
    return img_skeleton_uint8




def get_skeleton_nodes(binary_img):
    """
    Takes a binary image, skeletonizes it, and returns the coordinates
    of nodes where the degree is not equal to 2 (endpoints and intersections).
    
    Parameters:
        binary_img (numpy.ndarray): Input binary image (0 and 255/1).
        
    Returns:
        numpy.ndarray: An (N, 2) array of [y, x] coordinates.
    """
    # 1. Ensure image is Boolean for skimage
    # (Works with 0/255 or 0/1 inputs)
    
    # 2. Compute the skeleton
    # This returns a boolean array where True is the skeleton
    img_skeleton = skeletonize(binary_img.astype(bool))

    # Convert skeleton back to uint8 for OpenCV operations
    skel_int = img_skeleton.astype(np.uint8)
    # 4. Calculate Neighbors (Degree)
    # We use a 3x3 kernel to count the 8-connected neighbors.
    # The center is 0 because we only want to count neighbors, not the pixel itself.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve to get the neighbor count for every pixel
    # borderType=cv2.BORDER_CONSTANT assumes 0 outside the image
    degrees = cv2.filter2D(skel_int, cv2.CV_16U, np.ones((3,3), np.uint8))
    degrees -= skel_int
    #return degrees
    # 5. Filter Nodes
    # We want pixels that:
    #   a) Are part of the skeleton (skeleton_uint == 1)
    #   b) Do NOT have exactly 2 neighbors (degrees != 2)
    #
    # This captures:
    #   - Endpoints (degree 1)
    #   - Intersections (degree 3, 4...)
    #   - Isolated pixels (degree 0)
    #nodes_mask = (skeleton_uint == 1) & (degrees != 2)
    nodes_mask = (skel_int == 1) & (degrees !=2)
    nodes_mask_uint8 = (nodes_mask.astype(np.uint8)) * 255

    return nodes_mask_uint8
    # 6. Extract coordinates
    # np.argwhere returns coordinates in (row, col) -> (y, x) format
    # coords = np.argwhere(nodes_mask)
    
    # return coords
    




def component_pixel_polygon(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Given a binary mask (uint8, single component, non-zero = foreground),
    returns the exact outer polygon as a list of (x, y) corner coordinates
    tracing the pixel boundary — not pixel centers.

    The polygon is formed by following the border between foreground and
    background pixels, returning integer corner coordinates.

    For a 1x1 pixel at (col=0, row=0), returns:
        [(0,0), (1,0), (1,1), (0,1)]  — a unit square.

    For a cross shape, returns exactly the 12 corner coordinates.
    """
    # Find all foreground pixel coordinates (row, col)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    # --- Direction encoding ---
    # We trace edges between pixels using a clockwise winding.
    # Each "edge" is a directed segment along the boundary.
    # Directions: 0=right, 1=down, 2=left, 3=up
    # For each foreground pixel and each of its 4 sides,
    # if the neighbor on that side is background → that side is a boundary edge.

    # Edge direction → movement along that edge (dx, dy) for the polygon vertex walk
    # When walking clockwise around the exterior:
    #   right edge (top side of pixel, y=row):   from (col, row)   → (col+1, row)
    #   down  edge (right side of pixel, x=col+1): from (col+1, row) → (col+1, row+1)
    #   left  edge (bottom side of pixel, y=row+1): from (col+1, row+1) → (col, row+1)
    #   up    edge (left side of pixel, x=col): from (col, row+1) → (col, row)

    # For each boundary edge, store as: (start_corner, end_corner)
    # We'll build a graph of corner → corner and trace cycles.

    edge_graph = {}  # start_corner -> end_corner

    neighbor_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
    # Clockwise exterior edge for each direction the neighbor is absent:
    # If neighbor is above (dy=-1): top edge → goes right: (x, y) → (x+1, y)
    # If neighbor is right (dx=+1): right edge → goes down: (x+1, y) → (x+1, y+1)
    # If neighbor is below (dy=+1): bottom edge → goes left: (x+1, y+1) → (x, y+1)
    # If neighbor is left  (dx=-1): left edge → goes up: (x, y+1) → (x, y)
    edge_for_missing_neighbor = {
        (0, -1):  lambda x, y: ((x,   y),   (x+1, y)),    # top
        (1,  0):  lambda x, y: ((x+1, y),   (x+1, y+1)),  # right
        (0,  1):  lambda x, y: ((x+1, y+1), (x,   y+1)),  # bottom
        (-1, 0):  lambda x, y: ((x,   y+1), (x,   y)),    # left
    }

    for (x, y) in pixel_set:
        for (dx, dy), edge_fn in edge_for_missing_neighbor.items():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in pixel_set:
                start, end = edge_fn(x, y)
                edge_graph[start] = end

    # Trace all closed polygons by following the edge graph
    # (There should be exactly one outer polygon for a single connected component,
    #  plus possible inner holes — we return all as separate polygons.)
    visited = set()
    polygons = []

    for start_node in edge_graph:
        if start_node in visited:
            continue
        poly = []
        node = start_node
        while node not in visited:
            visited.add(node)
            poly.append(node)
            node = edge_graph[node]
        polygons.append(poly)

    # For a simple component, return the longest polygon (outer boundary)
    polygons.sort(key=len, reverse=True)
    return polygons[0] if polygons else []


def components_pixel_polygons(binary_image: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Run connected components on a binary image and return the pixel polygon
    for each component.

    Returns a list of polygons (one per component, excluding background).
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    polygons = []
    for label in range(1, labels.max() + 1):  # 0 is background
        component_mask = (labels == label).astype(np.uint8)
        poly = np.array(component_pixel_polygon(component_mask))
        polygons.append(poly)
    return polygons


