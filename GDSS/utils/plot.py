import math
import networkx as nx
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


options = {
    'node_size': 96,
    'edge_color' : 'black',
    'linewidths': 1.5,
    'width': 1,
    "edgecolors": "k"
}

plt.rcParams['font.size'] = 16

def plot_graphs_list(graphs, title='title', rows=4, cols=2, save_dir=None, N=0, 
                     pos_list=None, rel_x_err=None):
    batch_size = len(graphs)
    max_num = rows * cols
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure(figsize=(8,16))
    use_existing_pos = True
    if pos_list is None:
        pos_list = []
        use_existing_pos = False

    for i in range(max_num):
        # idx = i * (batch_size // max_num)
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        #G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(rows, cols, i + 1)
        title_str = f'e={e - l}, n={v}'
        # if 'lobster' in save_dir.split('/')[0]:
        #     if is_lobster_graph(graphs[idx]):
        #         title_str += f' [L]'
        if not use_existing_pos:
            pos = nx.spring_layout(G)
            pos_list.append(pos)
        else:
            pos = pos_list[idx]
        if rel_x_err is None:
            nx.draw(G, pos, with_labels=False, **options,
                    node_color=np.zeros(v), cmap='cool', vmin=0, vmax=1)
        else:
            nx.draw(G, pos, with_labels=False, **options,
                    node_color=rel_x_err[idx][:v], cmap='cool', vmin=0, vmax=1)
        ax.title.set_text(title_str)
    #figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)
    return pos_list


def save_fig(save_dir=None, title='fig', dpi=300):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*['samples', 'fig', save_dir])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=False)
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):

    if not(os.path.isdir('./samples/pkl/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./samples/pkl/{}'.format(log_folder_name)))
    with open('./samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name), 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = './samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name)
    return save_dir
