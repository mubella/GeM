""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

#data_graph:u l t x y

def build_global_POI_checkin_graph(POI,traj_ultxy, exclude_user=None):
    G = nx.DiGraph()
    print('len(POI)',len(POI))
    for i in range(1,len(POI)+1):
        node = i
        if node not in G.nodes():
            G.add_node(i,
                       checkin_cnt=0,
                       poi_deltat=0,  # 此节点和下一个poi的时间间隔的sum
                       poi_index=i,
                       latitude=POI[i-1][1],
                       longitude=POI[i-1][2])

    for (i,item)in enumerate(tqdm(traj_ultxy)):
        # Add node (POI)
        for(j,feature)in enumerate(item):
            #print(feature[0])
            if feature[0]!=0:
                node = feature[1]
                if node not in G.nodes():
                    G.add_node(feature[1],
                           checkin_cnt=1,
                           poi_deltat=0, #此节点和下一个poi的时间间隔的sum
                           poi_index=feature[1],
                           latitude=feature[3],
                           longitude=feature[4])
                else:
                    G.nodes[node]['checkin_cnt'] += 1


        # Add edges (Check-in seq)
        for (j, feature) in enumerate(item):
            if j<item.shape[0]-1:
                if feature[0] != 0 and item[j+1][0]!=0 :
                    if G.has_edge(feature[1], item[j+1][1]):
                        G.edges[feature[1], item[j+1][1]]['weight'] += 1
                        G.nodes[feature[1]]['poi_deltat']=G.nodes[feature[1]]['poi_deltat']+item[j+1][2]-feature[2]   #此节点和下一个poi的时间间隔的sum
                    else:  # Add new edge
                        G.add_edge(feature[1], item[j+1][1], weight=1)
                        G.nodes[feature[1]]['poi_deltat'] = G.nodes[feature[1]]['poi_deltat']+item[j + 1][2] - feature[2]  # 此节点和下一个poi的时间间隔的sum
        #对于delta_t进行平均修正
    for node in G.nodes():
        if G.nodes[node]['poi_deltat'] != 0 and G.nodes[node]['checkin_cnt'] != 0 :
            G.nodes[node]['poi_deltat']=G.nodes[node]['poi_deltat']/G.nodes[node]['checkin_cnt']
    return  G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    print('A.shpae',A.shape)
    #print(A.todense()[230].shape)
    np.save(os.path.join(dst_dir, 'graph_A.npy'), A.todense())
    #np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_deltat,poi_index,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_deltat = each[1]['poi_deltat']
            poi_index = each[1]['poi_index']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']

            print(f'{node_name},{checkin_cnt},'
                  f'{poi_deltat},{poi_index},'
                  f'{latitude},{longitude}', file=f)
    print('save graph_A.csv/graph_X.csv')

def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))
    print('save graph.pkl')

'''
def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)
'''



def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_index',
                             feature3='poi_deltat',
                             feature4='latitude', feature5='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4,feature5]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    #dname = 'NYC'
    #dname='TKY'
    # dname='Gowalla33'
    dname='CA'
    print(dname+" for train")
    dst_dir = r'data_graph/'+dname+'_graph'
    # Build POI checkin trajectory graph
    file = open('./data_graph/' + dname + '_ultxy_data.pkl', 'rb')
    traj_ultxy = joblib.load(file)#  (N,M,5)
    print(traj_ultxy.shape)
    traj_ultxy_98=traj_ultxy[:,:98,:]
    print(traj_ultxy_98[0].shape,traj_ultxy_98[1].shape)

    file = open('./data/' + dname + '_POI.npy', 'rb')
    POI_data = np.load(file,allow_pickle=True)
    print(POI_data)

    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(POI_data,traj_ultxy_98)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    #save_graph_edgelist(G, dst_dir=dst_dir)
