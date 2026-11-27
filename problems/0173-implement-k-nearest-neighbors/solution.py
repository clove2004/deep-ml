import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    lst = []
    for i in range (len(points)):
        d = np.sqrt((query_point[0] - points[i][0])**2 + (query_point[1] - points[i][1])**2)
        lst.append((points[i], d, i))
    lst.sort(key=lambda x : (x[1], x[2]))
    ans = [lst[i][0] for i in range(k)]
    return ans
