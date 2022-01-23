def generate_similarity_graph(sub_sub_sim_mat, n):
    similarity_graph = {}
    for cur_node, edge_dict in sub_sub_sim_mat.items():
        similarity_graph[cur_node] = []
        sorted_edges = sorted(edge_dict.items(), key=lambda item: item[1])
        count = 0
        i = 0
        while count < n:
            if sorted_edges[i][1] != 0:
                similarity_graph[cur_node].append(list(sorted_edges[i]))
                count += 1
            i += 1
    return similarity_graph


def generate_sim_graph(sub_sub_sim_mat, n):
    simGraph = {}
    for curNode, edgeDict in sub_sub_sim_mat.items():
        simGraph[curNode] = []
        sorted_edges = sorted(edgeDict.items(), key=lambda item: item[1])
        count = 0
        i = 0
        while count < n:
            if sorted_edges[i][1] != 0:
                simGraph[curNode].append(sorted_edges[i])
                count += 1
            i += 1
    return simGraph
