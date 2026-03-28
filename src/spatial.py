import math

def calculate_center(box: list, anchor: str = "center") -> tuple:
    """
    Calculate the anchor point of a bounding box [x1, y1, x2, y2].
    For flames on a stove, anchor="bottom" targets the base of the flame.
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    
    if anchor == "bottom":
        # Target the lower 15% of the bounding box as the base
        # This handles tall flames expanding upwards from the stove
        center_y = int(y1 + 0.85 * (y2 - y1))
    else:
        center_y = int((y1 + y2) / 2)
        
    return (center_x, center_y)

def is_point_in_zones(point: tuple, zones: list) -> bool:
    """
    Check if a given point is within the radius of any point in 'zones'.
    'zones' should be a list of (x, y, radius) tuples.
    """
    px, py = point
    for zx, zy, r in zones:
        distance = math.sqrt((px - zx)**2 + (py - zy)**2)
        if distance <= r:
            return True
    return False
