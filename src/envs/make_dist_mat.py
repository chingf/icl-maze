import numpy as np
import pickle

def calculate_tree_geodesic_distance(node1, node2):
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

def create_tree_geodesic_distance_matrix(depth):
    n_nodes = 2**depth - 1
    distance_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = calculate_tree_geodesic_distance(i, j)
    
    return distance_matrix

def create_darkroom_distance_matrix(maze_dim, distance_type):
    n_nodes = maze_dim * maze_dim
    distance_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    for i_node_idx in range(n_nodes):
        for j_node_idx in range(i_node_idx+1, n_nodes):
            i_x = i_node_idx // maze_dim
            i_y = i_node_idx % maze_dim
            j_x = j_node_idx // maze_dim
            j_y = j_node_idx % maze_dim
            if distance_type == 'geodesic':
                dist = abs(i_x - j_x) + abs(i_y - j_y)
            elif distance_type == 'euclidean':
                dist = np.sqrt((i_x - j_x)**2 + (i_y - j_y)**2)
            distance_matrix[i_node_idx][j_node_idx] = dist
            distance_matrix[j_node_idx][i_node_idx] = dist
    return distance_matrix

def save_matrix_to_pickle(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)

def main():
    maze_type = input("Enter the maze type (tree or darkroom): ")
    if maze_type == "tree":
        depth = int(input("Enter the depth of the full binary tree: "))
        distance_matrix = create_tree_geodesic_distance_matrix(
            depth)
        save_matrix_to_pickle(distance_matrix, f'tree_geodesic_dist_matrix_depth{depth}.pkl')
        print(f"Geodesic distance matrix saved.")
    elif maze_type == "darkroom":
        maze_dim = int(input("Enter the dimension of the maze: "))
        distance_matrix = create_darkroom_distance_matrix(maze_dim, 'geodesic')
        save_matrix_to_pickle(distance_matrix, f'maze_geodesic_dist_matrix_dim{maze_dim}.pkl')
        distance_matrix = create_darkroom_distance_matrix(maze_dim, 'euclidean')
        save_matrix_to_pickle(distance_matrix, f'maze_euclidean_dist_matrix_dim{maze_dim}.pkl')
        print(f"Darkroom distance matrix saved.")
    return distance_matrix

if __name__ == "__main__":
    d = main()