import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class Aggregator(nn.Module):
    """
    Local Weighted Smoothing aggregation scheme
    """
    def __init__(self, n_users, n_virtual, n_iter):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.n_iter = n_iter
        self.w = torch.nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.3]), requires_grad=True)

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, adj_mat):

        device = torch.device("cuda:0")
        n_entities = entity_emb.shape[0]
        n_users = self.n_users

        edge_type_uni = torch.unique(edge_type)
        entity_emb_list = []

        user_index, item_index = adj_mat.nonzero()
        user_index = torch.tensor(user_index).type(torch.long).to(device)
        item_index = torch.tensor(item_index).type(torch.long)
        """LWS for item representation on VRKGs"""
        for i in edge_type_uni:
            index = torch.where(edge_type == i)
            index = index[0]
            head, tail = edge_index
            head = head[index]
            tail = tail[index]
            u = None
            neigh_emb = entity_emb[tail]
            for clus_iter in range(self.n_iter):
                if u is None:
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)
                else:
                    center_emb = u[head]
                    sim = torch.sum(center_emb * neigh_emb, dim=1)
                    n, d = neigh_emb.size()
                    sim = torch.unsqueeze(sim, dim=1)
                    sim.expand(n, d)
                    neigh_emb = sim * neigh_emb
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)

                if clus_iter < self.n_iter - 1:
                    squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                    u = squash.unsqueeze(1) * F.normalize(u, dim=1)
                u += entity_emb
            entity_emb_list.append(u)
        entity_emb_list = torch.stack(entity_emb_list, dim=0)
        item_0 = entity_emb_list[0]
        item_1 = entity_emb_list[1]
        item_2 = entity_emb_list[2]
        w0 = self.w[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w1 = self.w[1].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w2 = self.w[2].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2)

        """LWS for user representation learning"""
        u = None
        for clus_iter in range(self.n_iter):
            neigh_emb = entity_emb[item_index]
            if u is None:
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=n_users, dim=0)
            else:
                center_emb = u[user_index]
                sim = torch.sum(center_emb * neigh_emb, dim=1)
                n, d = neigh_emb.size()
                sim = torch.unsqueeze(sim, dim=1)
                sim.expand(n, d)
                neigh_emb = sim * neigh_emb
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=n_users, dim=0)

            if clus_iter < self.n_iter - 1:
                squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                u = squash.unsqueeze(1) * F.normalize(u, dim=1)
            u += user_emb
        user_agg = u

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_iter, n_users,
                 n_virtual, n_relations, adj_mat, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.adj_mat = adj_mat
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_virtual=n_virtual, n_iter=n_iter))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                adj_mat, interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = 0
        weight = self.weight
        relation_ = torch.mm(weight, latent_emb.t())
        relation_remap = torch.argmax(relation_, dim=1)
        edge_type = relation_remap[edge_type - 1]

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, adj_mat)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.kg_l2loss_lambda = args_config.kg_l2loss_lambda
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_iter = args_config.n_iter
        self.n_virtual = args_config.n_virtual  # The number of virtual relation types
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_virtual, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_iter=self.n_iter,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_virtual=self.n_virtual,
                         adj_mat=self.adj_mat,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, cf_batch):
        user = cf_batch['users']
        pos_item = cf_batch['pos_items']
        neg_item = cf_batch['neg_items']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]

        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.adj_mat,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.adj_mat,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss + emb_loss, emb_loss

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)