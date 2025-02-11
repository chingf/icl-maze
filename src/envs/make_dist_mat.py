import numpy as np
import pickle

def calculate_geodesic_distance(node1, node2):
    # Calculate the path from the root to each node
    path1 = []
    path2 = []
    
    # Find the path to the root for node1
    while node1 > 0:
        path1.append(node1)
        node1 = (node1 - 1) // 2
    
    # Find the path to the root for node2
    while node2 > 0:
        path2.append(node2)
        node2 = (node2 - 1) // 2
    
    # Add the root node
    path1.append(0)
    path2.append(0)
    
    # Reverse the paths to start from the root
    path1.reverse()
    path2.reverse()
    
    # Find the lowest common ancestor
    i = 0
    while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
        i += 1
    
    # Calculate the geodesic distance
    distance = (len(path1) - i) + (len(path2) - i)
    return distance

def create_geodesic_distance_matrix(depth):
    n_nodes = 2**depth - 1
    distance_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = calculate_geodesic_distance(i, j)
    
    return distance_matrix

def save_matrix_to_pickle(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)

def main():
    depth = int(input("Enter the depth of the full binary tree: "))
    distance_matrix = create_geodesic_distance_matrix(
        depth)
    save_matrix_to_pickle(distance_matrix, f'depth{depth}_distance_matrix.pkl')
    print(f"Geodesic distance matrix saved.")
    return distance_matrix

if __name__ == "__main__":
    d = main()
    import pdb; pdb.set_trace()