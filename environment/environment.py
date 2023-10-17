from utils.functions import pairwise_iteration
from utils.functions import Get_List_Max_Index
from utils.functions import remove_equal
from utils.functions import remove_repe
from utils.functions import fix_list
from pulp import *
import networkx as nx
import numpy as np
import copy
import os
import gin.tf
import time
import heapq
# from ortools.linear_solver import pywraplp

DEFAULT_EDGE_ATTRIBUTES = {
    'increments': 1,
    'reductions': 1,
    'weight': 0.0,
    'traffic': 0.0,
    'pre_traffic': 0.0
}


@gin.configurable
class Environment(object):

    def __init__(self,
                 env_type='NSFNet',
                 traffic_profile='uniform',
                 # routing='ecmp',
                 routing='sp',
                 init_sample=0,
                 seed_init_weights=1,
                 min_weight=1.0,
                 max_weight=4.0,
                 weight_change=1.0,
                 weight_update='sum',
                 weigths_to_states=True,
                 link_traffic_to_states=True,
                 probs_to_states=False,
                 reward_magnitude='link_traffic',
                 base_reward='min_max',
                 reward_computation='change',
                 base_dir='datasets'):
        
        env_type = [env for env in env_type.split('+')]
        self.env_type = env_type 
        self.traffic_profile = traffic_profile
        self.routing = routing

        self.num_sample = init_sample-1
        self.seed_init_weights = seed_init_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_change = weight_change
        self.weight_update = weight_update

        num_features = 0
        self.weigths_to_states = weigths_to_states
        if self.weigths_to_states: num_features += 1
        self.link_traffic_to_states = link_traffic_to_states
        if self.link_traffic_to_states: num_features += 1
        self.probs_to_states = probs_to_states
        if self.probs_to_states: num_features += 2
        self.num_features = num_features
        self.reward_magnitude = reward_magnitude
        self.base_reward = base_reward
        self.reward_computation = reward_computation

        self.base_dir = base_dir
        self.dataset_dirs = []
        for env in env_type:
            self.dataset_dirs.append(os.path.join(base_dir, env, traffic_profile))
        
        self.initialize_environment()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        self.reward_measure = self.compute_reward_measure()
        self.set_target_measure()




    def load_topology_object(self):
        try:
            nx_file = os.path.join(self.base_dir, self.network, 'graph_attr.txt')
            self.topology_object = nx.DiGraph(nx.read_gml(nx_file, destringizer=int))
        except: 
            self.topology_object = nx.DiGraph()
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph.txt')
            with open(capacity_file) as fd:
                '''系统内核(kernel)使用文件描述符(file descriptor，简称fd)来访问文件，也就是说，实际上在使用open()函数打开现存文件时，
                内核返回的是一个文件描述符。读写文件时也需要使用文件描述符来指定要读写的文件。文件描述符在形式上是一个非负整数，实则是一个索引值。'''
                for line in fd:
                    if 'Link_' in line:
                        camps = line.split(" ")
                        self.topology_object.add_edge(int(camps[1]),int(camps[2]))
                        self.topology_object[int(camps[1])][int(camps[2])]['bandwidth'] = int(camps[4])

    def load_capacities(self):
        if self.traffic_profile == 'gravity_full':
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph-TM-'+str(self.num_sample)+'.txt')
        else:
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph.txt')
        with open(capacity_file) as fd:
            for line in fd:
                if 'Link_' in line:
                    camps = line.split(" ")
                    self.G[int(camps[1])][int(camps[2])]['capacity'] = int(camps[4])
                    # self.G[int(camps[1])][int(camps[2])]['weight'] = int(camps[3])

    def load_traffic_matrix(self):
        tm_file = os.path.join(self.dataset_dir, 'TM', 'TM-'+str(self.num_sample))
        self.traffic_demand = np.zeros((self.n_nodes,self.n_nodes))
        with open(tm_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                self.traffic_demand[int(camps[1]),int(camps[2])] = float(camps[3])
        self.get_link_probs()

    def initialize_environment(self, num_sample=None, random_env=True):
        if num_sample is not None:
            self.num_sample = num_sample
        else:
            self.num_sample += 1
        if random_env:
            num_env = np.random.randint(0,len(self.env_type))
        else:
            num_env = self.num_sample % len(self.env_type)
        self.network = self.env_type[num_env]
        self.dataset_dir = self.dataset_dirs[num_env]

        self.load_topology_object()
        self.generate_graph()
        self.load_capacities()
        self.load_traffic_matrix()

    def next_sample(self):
        if len(self.env_type) > 1:
            self.initialize_environment()
        else:
            self.num_sample += 1
            self._reset_edge_attributes()
            self.load_capacities()
            self.load_traffic_matrix()

    def define_num_sample(self, num_sample):
        self.num_sample = num_sample - 1

    def reset(self, change_sample=False):
        if change_sample:
            self.next_sample()
        else:
            if self.seed_init_weights is None:self._define_init_weights()
            self._reset_edge_attributes()

        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        self.reward_measure = self.compute_reward_measure()
        self.set_target_measure()


    def generate_graph(self):
        G  = copy.deepcopy(self.topology_object)
        self.n_nodes = G.number_of_nodes()
        self.n_links = G.number_of_edges()
        self._define_init_weights()
        idx = 0
        link_ids_dict = {}
        for node in G.nodes():
            G.nodes[node]['is_SR'] = 0
        for (i,j) in G.edges():
            G[i][j]['id'] = idx
            G[i][j]['increments'] = 1
            G[i][j]['reductions'] = 1
            G[i][j]['weight'] = copy.deepcopy(self.init_weights[idx])
            link_ids_dict[idx] = (i,j)
            G[i][j]['capacity'] = G[i][j]['bandwidth']
            G[i][j]['traffic'] = 0.0
            G[i][j]['pre_traffic'] = 0.0
            idx += 1
        self.G = G
        incoming_links, outcoming_links = self._generate_link_indices_and_adjacencies()
        self.G.add_node('graph_data', link_ids_dict=link_ids_dict, incoming_links=incoming_links, outcoming_links=outcoming_links)

    def nodes_degree(self, graph):
        degree_list = []
        for i in graph.nodes():
            degree = graph.degree(i)
            degree_list.append(degree)
        xmin = min(degree_list)
        xmax = max(degree_list)
        for i, x in enumerate(degree_list):
            degree_list[i] = (x - xmin) / (xmax - xmin)
            # graph.nodes[i]['degree'] = degree_list[i]
        # print('Normalized List:', list)
        return degree_list

    def nodes_betweenness(self, graph):
        betweenness = nx.betweenness_centrality(graph, weight='weight')
        return betweenness

    def most_loaded_link(self, nodes):
        neibor_link_traffic = []
        for i in self.G.neighbors(nodes):
            if (nodes, i) in self.G.edges():
                link_traffic = self.G[nodes][i]['pre_traffic']
                link_capacity = link_traffic / self.G[nodes][i]['capacity']
                neibor_link_traffic.append(link_capacity)
            else:
                link_traffic = self.G[i][nodes]['pre_traffic']
                link_capacity = link_traffic / self.G[i][nodes]['capacity']
                neibor_link_traffic.append(link_capacity)
        most_load = max(neibor_link_traffic, default=0)
        # print(neibor_link_traffic)
        return most_load

    def SR_nodes_choose(self):
        para_1 = 1 / 3
        para_2 = 1 / 3
        para_3 = 1 / 3
        betweenness = self.nodes_betweenness(self.G)
        degree = self.nodes_degree(self.G)
        most_load = []
        sr_value_list = []
        for i in self.G.nodes():
            most_load.append(self.most_loaded_link(i))
            # print(i,most_load)
        for i in range(self.n_nodes):
            sr_value = para_1 * degree[i] + para_3 * most_load[i] + para_2 * betweenness[i]
            sr_value_list.append(sr_value)
        split_SR = 10
        # if self.n_nodes * 0.5 <= 2:
        #     split_SR = 2
        # else:
        #     split_SR = int(self.n_nodes * 0.1)
        final_sr_value = heapq.nlargest(split_SR, sr_value_list)
        final_sr_value_index = Get_List_Max_Index(sr_value_list, split_SR)
        for i in final_sr_value_index:
            self.G.nodes[i]['is_SR'] = 1
        # print(sr_value_list)
        return sr_value_list, final_sr_value, final_sr_value_index

    def subpath_1(self):
        routing = self.sp_routing
        # print(routing)
        sr_middle = []
        subpath_dict_1 = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
                          range(self.G.number_of_nodes() - 1)}
        self.SR_nodes_choose()
        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                if routing.get(i, {}).get(j, []):
                    traffic = self.traffic_demand[i][j]
                    for n in (routing[i][j][0:-1]):
                        if self.G.nodes[n]['is_SR'] == 1:
                            sr_middle.append(n)
                    if len(sr_middle) == 1:
                        self.all_path_sets[i][j].append(routing[i][j])
                    elif len(sr_middle) == 0:
                        self.all_path_sets[i][j].append(routing[i][j])
                    else:
                        lens = len(sr_middle)
                        for n in range(lens):
                            sr_index = routing[i][j].index(sr_middle[n])
                            subpath_dict_1[i][j].append(routing[i][j][0:sr_index + 1])
                    sr_middle.clear()
        self.subpath_dict_1 = subpath_dict_1
        return dict(subpath_dict_1)

    def subpath_2(self):
        subpath_list_2 = []
        _, _, sr_list = self.SR_nodes_choose()
        for i in sr_list:
            for j in sr_list:
                if i == j: continue
                path = nx.shortest_path(self.G, i, j, weight='weight')
                subpath_list_2.append(path)
        self.subpath_list_2 = subpath_list_2
        return subpath_list_2

    def subpath_3(self):
        subpath_dict_3 = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
                          range(self.G.number_of_nodes() - 1)}
        _, _, sr_list = self.SR_nodes_choose()
        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                for n in sr_list:
                    if n == j or n == i: continue
                    subpath_dict_3[i][j].append(nx.shortest_path(self.G, n, j, weight='weight'))
        self.subpath_dict_3 = subpath_dict_3
        return dict(subpath_dict_3)

    def all_path(self):
        path_sets = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
                     range(self.G.number_of_nodes() - 1)}
        all_path_sets = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
                         range(self.G.number_of_nodes() - 1)}
        self.all_path_sets = all_path_sets

        self.subpath_1()
        self.subpath_2()
        self.subpath_3()
        # self.path_sets = path_sets
        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                if len(self.subpath_dict_1[i][j]) == 0: continue
                # if not self.all_path_sets[i][j]: continue
                fix_list(self.subpath_dict_1[i][j], self.subpath_list_2, path_sets[i][j])
                fix_list(path_sets[i][j], self.subpath_dict_3[i][j], self.all_path_sets[i][j])
                self.all_path_sets[i][j].append(nx.shortest_path(self.G, i, j, weight='weight'))
                remove_repe(self.all_path_sets[i][j])
                remove_equal(self.all_path_sets[i][j])
        return dict(self.all_path_sets)

    # def try_all_path_sets(self):
    #     flow_ratio = {i: {j: {p: list() for p in range(len(self.all_path_sets[i][j]))}
    #     for j in range(self.G.number_of_nodes() - 1) if j != i} for i in range(self.G.number_of_nodes() - 1)}
    #     print("\n",'初始化',flow_ratio)
    #     flow_ratio[0][6][0]=1
    #     a=flow_ratio[0][6][0]
    #     print('a的值',a)
    #     flow_ratio[0][6][1] = 1
    #     a=a+flow_ratio[0][6][1]
    #     print('a的值',a)
    #     print(flow_ratio)

        # flow_ratio[0][1].append(1)
        # flow_ratio[0][1].append(3)
    #
    #     # print('all_path_sets',self.all_path_sets)
    #     flow_ratio = self.all_path_sets
    #     print('flow_ratio',flow_ratio)
    #     print(flow_ratio[0][6][1])
    #     # print('all_path_sets', self.all_path_sets[0][6][1])
    #     # print('len all_path_sets',len(self.all_path_sets[0][6]))

    def Umax(self, measure = None):
        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                traffic = self.traffic_demand[i][j]
                number = range(len(self.all_path_sets[i][j]))
                if traffic !=0 :
                    for p in number:
                        actual_traffic = traffic * self.flow_ratio[i][j][p]
                        actual_routing = self.all_path_sets[i][j][p]
                        for u, v in pairwise_iteration(actual_routing):
                            self.G[u][v]['traffic'] += actual_traffic
                else:
                    for p in number:
                        actual_routing = self.all_path_sets[i][j][p]
                        for u, v in pairwise_iteration(actual_routing):
                            self.G[u][v]['traffic'] += 0

        for (i,j) in self.G.edges():
            self.G[i][j]['traffic'] /= self.G[i][j]['capacity']

        link_traffic = [0] * self.n_links
        for i, j in self.G.edges():
            link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
        self.link_traffic = link_traffic
        self.mean_traffic = np.mean(link_traffic)
        self.get_weights()

        if measure is None:
            if self.reward_magnitude == 'link_traffic':
                measure = self.link_traffic
            elif self.reward_magnitude == 'weights':
                measure = self.raw_weights


        return measure

    # def split_ratio(self):
    #     self.all_path()
    #     sum_ratio = {i: {j: set() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
    #                  range(self.G.number_of_nodes() - 1)}
    #     edge_ratio = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
    #                   range(self.G.number_of_nodes() - 1)}
    #     typeChange_traffic_demand = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
    #                                  range(self.G.number_of_nodes() - 1)}
    #     typechange_capacity = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
    #                            range(self.G.number_of_nodes() - 1)}
    #
    #
    #     for i in self.G.nodes():
    #         if i == 'graph_data': continue
    #         for j in self.G.nodes():
    #             if j == 'graph_data' or i == j: continue
    #             sum_ratio[i][j] = 0.00
    #             edge_ratio[i][j] = 0.00
    #             # Umax[i][j] = 0.00
    #             typeChange_traffic_demand[i][j] = 0.00
    #             typechange_capacity[i][j] = 0.00
    #
    #     solver = pywraplp.Solver.CreateSolver('GLOP_GPU')
    #
    #     # # Linear Programming
    #     # LP_Problem = LpProblem("LP_Problem", LpMinimize)
    #
    #     # LpVariable
    #     max_measure = solver.NumVar(0, 1, 'max_measure')
    #
    #     flow_ratio = {}
    #     for i in range(self.G.number_of_nodes() - 1):
    #         if i not in flow_ratio:
    #             flow_ratio[i] = {}
    #         for j in range(self.G.number_of_nodes() - 1):
    #             if j != i:
    #                 if j not in flow_ratio[i]:
    #                     flow_ratio[i][j] = {}
    #                 if self.traffic_demand[i][j] == 0: continue
    #                 for p in range(len(self.all_path_sets[i][j])):
    #                     flow_ratio[i][j][p] = solver.NumVar(0, 1,'flow_ratio' + str(i) + ',' + str(j) + ',' + str(p))
    #
    #     self.flow_ratio = flow_ratio
    #
    #     # objective function
    #     # LP_Problem += self.Umax()
    #     constraint = {}
    #     measure = self.Umax()
    #     # LP_Problem += measure[i] <= max_measure
    #     for i in range(len(measure)):
    #         constraint[i] = solver.Constraint(-solver.infinity(), 0)
    #         constraint[i].SetCoefficient(measure[i], 1)
    #         constraint[i].SetCoefficient(max_measure, -1)
    #
    #     #  LP_Problem += sum_ratio[i][j] >= 1
    #     item = len(measure)-1
    #     for i in self.G.nodes():
    #         if i == 'graph_data': continue
    #         for j in self.G.nodes():
    #             if j == 'graph_data' or i == j: continue
    #             ratio = 0
    #             number = range(len(self.all_path_sets[i][j]))
    #             if self.traffic_demand[i][j] == 0: continue
    #             for p in number:
    #                 ratio = ratio + flow_ratio[i][j][p]
    #             sum_ratio[i][j] = ratio
    #             item += 1
    #             constraint[item] = solver.Constraint(1, solver.infinity())
    #             constraint[item].SetCoefficient(sum_ratio[i][j], 1)
    #
    #     traffic_demand = self.traffic_demand
    #     type_change = traffic_demand.tolist()
    #     # print('type_change',type_change)
    #
    #     for i in self.G.nodes():
    #         if i == 'graph_data': continue
    #         for j in self.G.nodes():
    #             if j == 'graph_data' or i == j: continue
    #             typeChange_traffic_demand[i][j] = type_change[i][j]
    #     # print('typeChange_traffic_demand',typeChange_traffic_demand) #格式转变为list
    #
    #     for i in self.G.nodes():
    #         if i == 'graph_data': continue
    #         for j in self.G.nodes():
    #             if j == 'graph_data' or i == j: continue
    #             number = range(len(self.all_path_sets[i][j]))
    #             if self.traffic_demand[i][j] == 0: continue
    #             for p in number:
    #                 length = range(len(self.all_path_sets[i][j][p]) - 1)
    #                 for edge in length:
    #                     edge_list = self.all_path_sets[i][j][p]
    #                     e_in = edge_list[edge]
    #                     e_out = edge_list[edge + 1]
    #                     edge_ratio[e_in][e_out] = edge_ratio[e_in][e_out] + flow_ratio[i][j][p] * typeChange_traffic_demand[i][j]
    #
    #     # LP_Problem += edge_ratio[i][j] <= typechange_capacity[i][j] * max_measure
    #     for (i, j) in self.G.edges:
    #         typechange_capacity[i][j] = self.G[i][j]['capacity']
    #         constraint[item] = solver.Constraint(-solver.infinity(), 0)
    #         constraint[item].SetCoefficient(edge_ratio[i][j], 1)
    #         constraint[item].SetCoefficient(typechange_capacity[i][j] * max_measure, -1)
    #
    #         # Umax[i][j] = edge_ratio[i][j]/typechange_capacity[i][j]
    #
    #     objective = solver.Objective()
    #     objective.SetCoefficient(max_measure, 1)
    #     objective.SetMinimization()
    #
    #     # LP_Problem.writeLP("LP_Problem.lp")
    #     # LP_Problem.solve()
    #
    #     for i in flow_ratio:
    #         for j in flow_ratio[i]:
    #             if self.traffic_demand[i][j] == 0: continue
    #             for p in flow_ratio[i][j]:
    #                 flow_ratio[i][j][p] = flow_ratio[i][j][p].solution_value()
    #
    #     self.flow_ratio = flow_ratio
    #     # print(flow_ratio)
    #     measure = self.Umax()
    #     min_Umax =max(measure)
    #     self.max_measure = min_Umax
    #
    #     return self.flow_ratio, self.max_measure


    def split_ratio(self):
        self.all_path()
        # print('path',self.all_path_sets)
        #Initialize all parameters
        # flow_ratio = {i: {j: {p: list() for p in range(len(self.all_path_sets[i][j]))} for j in range(self.G.number_of_nodes() - 1) if j != i} for i in range(self.G.number_of_nodes() - 1)}
        sum_ratio = {i: {j: set() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in range(self.G.number_of_nodes() - 1)}
        edge_ratio = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in range(self.G.number_of_nodes() - 1)}
        typeChange_traffic_demand = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
                                     range(self.G.number_of_nodes() - 1)}
        typechange_capacity = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
                               range(self.G.number_of_nodes() - 1)}
        # Umax = {i: {j: list() for j in range(self.G.number_of_nodes() - 1) if j != i} for i in
        #               range(self.G.number_of_nodes() - 1)}

        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                sum_ratio[i][j] = 0.00
                edge_ratio[i][j] = 0.00
                # Umax[i][j] = 0.00
                typeChange_traffic_demand[i][j] = 0.00
                typechange_capacity[i][j] = 0.00
        # for i in self.G.nodes():
        #     if i == 'graph_data': continue
        #     for j in self.G.nodes():
        #         if j == 'graph_data' or i == j: continue
        #         for p in range(len(self.all_path_sets[i][j])):
        #             flow_ratio[i][j][p] = 0.0
        # self.flow_ratio = flow_ratio

        # Linear Programming
        LP_Problem = LpProblem("LP_Problem",LpMinimize)

        # LpVariable
        max_measure = LpVariable('max_measure', lowBound=0, upBound=1, cat=LpContinuous)
        # flow_ratio = {
        #     i: {
        #         j: {
        #             p: LpVariable('flow_ratio' + str(i) + ',' + str(j) + ',' + str(p), 0, 1, LpContinuous)
        #             for p in range(len(self.all_path_sets[i][j]))
        #         }
        #         for j in range(self.G.number_of_nodes() - 1) if j != i
        #     }
        #     for i in range(self.G.number_of_nodes() - 1)
        # }
        flow_ratio = {}
        for i in range(self.G.number_of_nodes() - 1):
            if i not in flow_ratio:
                flow_ratio[i] = {}
            for j in range(self.G.number_of_nodes() - 1):
                if j != i:
                    if j not in flow_ratio[i]:
                        flow_ratio[i][j] = {}
                    if self.traffic_demand[i][j] ==0: continue
                    for p in range(len(self.all_path_sets[i][j])):
                        flow_ratio[i][j][p] = LpVariable('flow_ratio' + str(i) + ',' + str(j) + ',' + str(p), 0, 1, LpContinuous)

        self.flow_ratio = flow_ratio

        # objective function
        # LP_Problem += self.Umax()
        measure = self.Umax()
        for i in range(len(measure)):
            LP_Problem += measure[i] <= max_measure

        LP_Problem += max_measure

        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                ratio = 0
                number = range(len(self.all_path_sets[i][j]))
                if self.traffic_demand[i][j] == 0: continue
                for p in number:
                    ratio = ratio + flow_ratio[i][j][p]
                sum_ratio[i][j] = ratio
                LP_Problem += sum_ratio[i][j] >= 1
                LP_Problem += sum_ratio[i][j] <= 1


        # print('sum_ratio',sum_ratio)
        # LP_Problem += self.compute_reward_measure()

        traffic_demand = self.traffic_demand
        type_change = traffic_demand.tolist()
        # print('type_change',type_change)

        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                typeChange_traffic_demand[i][j] = type_change[i][j]
        # print('typeChange_traffic_demand',typeChange_traffic_demand) #格式转变为list

        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                number = range(len(self.all_path_sets[i][j]))
                if self.traffic_demand[i][j] == 0: continue
                for p in number:
                    length = range(len(self.all_path_sets[i][j][p])-1)
                    for edge in length:
                        edge_list = self.all_path_sets[i][j][p]
                        e_in = edge_list[edge]
                        e_out = edge_list[edge + 1]
                        edge_ratio[e_in][e_out] = edge_ratio[e_in][e_out] + flow_ratio[i][j][p] * typeChange_traffic_demand[i][j]


        for (i, j) in self.G.edges:
            typechange_capacity[i][j] = self.G[i][j]['capacity']
            LP_Problem += edge_ratio[i][j] <= typechange_capacity[i][j] * max_measure
            # Umax[i][j] = edge_ratio[i][j]/typechange_capacity[i][j]

        LP_Problem.writeLP("LP_Problem.lp")
        LP_Problem.solve(PULP_CBC_CMD(msg=False))
        # print(LpStatus[LP_Problem.status],"\n")
        # for v in LP_Problem.variables():
        #     print(v.name, "=", v.varValue, "\n")
        # print('value:',pulp.value(LP_Problem.objective),"\n")

        # for v in LP_Problem.variables():
        #     for i in self.G.nodes():
        #         if i == 'graph_data': continue
        #         for j in self.G.nodes():
        #             if j == 'graph_data' or i == j: continue
        #             number = range(len(self.all_path_sets[i][j]))
        #             for p in number:
        #                 if v.name == 'flow_ratio'+str(i)+','+str(j)+','+str(p):
        #                     flow_ratio[i][j][p] = v.varValue
        #                     print("flow_ratio",i,j,p,":",flow_ratio[i][j][p])
        for i in flow_ratio:
            for j in flow_ratio[i]:
                if self.traffic_demand[i][j] == 0: continue
                for p in flow_ratio[i][j]:
                    flow_ratio[i][j][p] = flow_ratio[i][j][p].value()

        self.flow_ratio = flow_ratio
        # print(flow_ratio)
        min_Umax = pulp.value(LP_Problem.objective)
        self.max_measure = min_Umax

        return self.flow_ratio, self.max_measure




    def set_target_measure(self):
        self.target_sp_routing = copy.deepcopy(self.sp_routing)
        self.target_reward_measure = copy.deepcopy(self.reward_measure)
        self.target_link_traffic = copy.deepcopy(self.link_traffic)
        self.get_weights()
        self.target_weights = copy.deepcopy(self.raw_weights)


    def get_weights(self, normalize=True):
        weights = [0.0]*self.n_links
        for i,j in self.G.edges():
            weights[self.G[i][j]['id']] = copy.deepcopy(self.G[i][j]['weight'])
        self.raw_weights = weights
        max_weight = self.max_weight*3
        self.weights = [weight/max_weight for weight in weights]

    def get_state(self):
        state = []
        link_traffic = copy.deepcopy(self.link_traffic)
        weights = copy.deepcopy(self.weights)
        if self.link_traffic: 
            state += link_traffic
        if self.weigths_to_states: 
            state += weights
        if self.probs_to_states:
            state += self.p_in
            state += self.p_out
        return np.array(state, dtype=np.float32)

    def define_weight(self, link, weight):
        i, j = link
        self.G[i][j]['weight'] = weight
        self._generate_routing()
        self._get_link_traffic()
        
    def update_weights(self, link, action_value, step_back=False):
        i, j = link
        if self.weight_update == 'min_max':
            if action_value == 0:
                self.G[i][j]['weight'] = max(self.G[i][j]['weight']-self.weight_change, self.min_weight)
            elif action_value == 1:
                self.G[i][j]['weight'] = min(self.G[i][j]['weight']+self.weight_change, self.max_weight)
        else: 
            if self.weight_update == 'increment_reduction':
                if action_value == 0:
                    self.G[i][j]['reductions'] += 1
                elif action_value == 1:
                    self.G[i][j]['increments'] += 1
                self.G[i][j]['weight'] = self.G[i][j]['increments'] / self.G[i][j]['reductions']
            elif self.weight_update == 'sum':
                if step_back:
                    self.G[i][j]['weight'] -= self.weight_change
                else:    
                    self.G[i][j]['weight'] += self.weight_change
            
    def reinitialize_weights(self, seed_init_weights=-1, min_weight=None, max_weight=None):
        if seed_init_weights != -1: 
            self.seed_init_weights = seed_init_weights
        if min_weight: self.min_weight = min_weight
        if max_weight: self.max_weight = max_weight

        self.generate_graph()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()

    def reinitialize_routing(self, routing):
        self.routing = routing
        self._get_link_traffic()

    def step(self, action, step_back=False):
        #link_id, action_value = action
        link = self.G.nodes()['graph_data']['link_ids_dict'][action]
        #self.update_weights(link, action_value, step_back)
        self.update_weights(link, 0, step_back)
        self.get_weights()
        self._generate_routing()
        self._distribute_pre_link_traffic()
        pre_routing = self.routing
        state, reward = self._distribute_link_traffic(routing="flow_split")
        self.routing = pre_routing
        # self._get_link_traffic(routing="flow_split")
        # state = self.get_state()
        # reward = self._compute_reward()
        return state, reward

    def step_max_LU(self, action, step_back=False):
        # link_id, action_value = action
        link = self.G.nodes()['graph_data']['link_ids_dict'][action]
        # self.update_weights(link, action_value, step_back)
        self.update_weights(link, 0, step_back)
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        LU_max = self.compute_reward_measure()
        return LU_max

    def step_back(self, action):
        state, reward = self.step(action, step_back=True)
        return state, reward


    # in the q_function we want to use info on the complete path (src_node, next_hop, n3, n4, ..., dst_node)
    # this function returns the indices of links in the path
    def get_complete_link_path(self, node_path):
        link_path = []
        for i, j in pairwise_iteration(node_path):
            link_path.append(self.G[i][j]['id'])
        # pad the path until "max_length" (implementation is easier if all paths have same size)
        link_path = link_path + ([-1] * (self.n_links-len(link_path)))
        return link_path


 
    """
    ****************************************************************************
                 PRIVATE FUNCTIONS OF THE ENVIRONMENT CLASS
    ****************************************************************************
    """
    
    def _define_init_weights(self):
        np.random.seed(seed=self.seed_init_weights)
        self.init_weights = np.random.randint(self.min_weight, self.max_weight+1, self.n_links)
        # self.init_weights = [1,1,2,1,1,1,1,3,1,1,1,2,2,3,3,2,1,3]
        np.random.seed(seed=None)
        

    # generates indices for links in the network
    def _generate_link_indices_and_adjacencies(self):
        # for the q_function, we want to have info on link-link connection points
        # there is a link-link connection between link A and link B if link A
        # is an incoming link of node C and link B is an outcoming node of node C.
        # For connection "i", the incoming link is incoming_links[i] and the
        # outcoming link is outcoming_links[i]
        incoming_links = []
        outcoming_links = []
        # iterate through all links
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                incoming_link_id = self.G[i][j]['id']
                # for each link, search its outcoming links
                for k in self.G.neighbors(j):
                    outcoming_link_id = self.G[j][k]['id']
                    incoming_links.append(incoming_link_id)
                    outcoming_links.append(outcoming_link_id)

        return incoming_links, outcoming_links

    def _reset_edge_attributes(self, attributes=None):
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list:
            attributes = [attributes]
        for (i,j) in self.G.edges():
            for attribute in attributes:
                if attribute == 'weight':
                    self.G[i][j][attribute] = copy.deepcopy(self.init_weights[self.G[i][j]['id']])
                else:
                    self.G[i][j][attribute] = copy.deepcopy(DEFAULT_EDGE_ATTRIBUTES[attribute])

    def _normalize_traffic(self):
        for (i,j) in self.G.edges():
            self.G[i][j]['traffic'] /= self.G[i][j]['capacity']

    def _generate_routing(self, next_hop=None):
        self.sp_routing = dict(nx.all_pairs_dijkstra_path(self.G))
        #self.path_lengths = dict(nx.all_pairs_dijkstra_path_length(self.G))


    def successive_equal_cost_multipaths(self, src, dst, traffic):
        new_srcs = self.next_hop_dict[src][dst]
        traffic /= len(new_srcs)
        for new_src in new_srcs:
            self.G[src][new_src]['traffic'] += traffic
            if new_src != dst:
                self.successive_equal_cost_multipaths(new_src, dst, traffic)

    # returns a list of traffic volumes of each link
    def _distribute_link_traffic(self, routing=None):
        self._reset_edge_attributes('traffic')
        if routing != None:
            self.routing = routing
        if self.routing == 'sp':
            # start_time = time.time()
            if routing is None: routing = self.sp_routing
            for i in self.G.nodes():
                if i=='graph_data': continue
                for j in self.G.nodes():
                    if j=='graph_data' or i == j: continue
                    traffic = self.traffic_demand[i][j]
                    if traffic == 0:continue
                    for u,v in pairwise_iteration(routing[i][j]):
                        self.G[u][v]['traffic'] += traffic
            self._normalize_traffic()
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            #
            # print('sp_time',f"The code took {elapsed_time} seconds to run.")
            return
        elif self.routing == 'ecmp':
            visited_pairs = set()
            self.next_hop_dict = {i : {j : set() for j in range(self.G.number_of_nodes()-1) if j != i} for i in range(self.G.number_of_nodes()-1)}
            for src in range(self.G.number_of_nodes()-1):
                for dst in range(self.G.number_of_nodes()-1):
                    if src == dst: continue
                    if (src,dst) not in visited_pairs:
                        routings = set([item for sublist in [[(routing[i],routing[i+1]) for i in range(len(routing)-1)] for routing in nx.all_shortest_paths(self.G, src, dst, 'weight')] for item in sublist])
                        for (new_src,next_hop) in routings:
                            self.next_hop_dict[new_src][dst].add(next_hop)
                            visited_pairs.add((new_src,dst))
                    traffic = self.traffic_demand[src][dst]
                    self.successive_equal_cost_multipaths(src, dst, traffic)
            self._normalize_traffic()
            return
        elif self.routing == 'flow_split':
            # start_time = time.time()
            link_traffic = [0] * self.n_links
            for i, j in self.G.edges():
                link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
            self.link_traffic = link_traffic
            # if routing is None: routing = self.all_path()
            path_ratio, min_Umax = self.split_ratio()
            self.reward_measure = min_Umax
            for i in self.G.nodes():
                if i == 'graph_data': continue
                for j in self.G.nodes():
                    if j == 'graph_data' or i == j: continue
                    for i, j in self.G.edges():
                        self.G[i][j]['traffic'] = 0.0
            for i in self.G.nodes():
                if i == 'graph_data': continue
                for j in self.G.nodes():
                    if j == 'graph_data' or i == j: continue
                    traffic = self.traffic_demand[i][j]
                    number = range(len(self.all_path_sets[i][j]))
                    if traffic != 0:
                        for p in number:
                            actual_traffic = traffic * self.flow_ratio[i][j][p]
                            actual_routing = self.all_path_sets[i][j][p]
                            for u, v in pairwise_iteration(actual_routing):
                                self.G[u][v]['traffic'] += actual_traffic
                    else:
                        for p in number:
                            actual_routing = self.all_path_sets[i][j][p]
                            for u, v in pairwise_iteration(actual_routing):
                                self.G[u][v]['traffic'] += 0
            self._normalize_traffic()
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            #
            # print('flow_split_time',f"The code took {elapsed_time} seconds to run.")
            link_traffic = [0] * self.n_links
            for i, j in self.G.edges():
                link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
            self.link_traffic = link_traffic
            self.mean_traffic = np.mean(link_traffic)
            self.get_weights()
            state = self.get_state()
            reward = self._compute_reward(current_reward_measure=min_Umax)
            return state, reward



    def _distribute_pre_link_traffic(self, routing=None):
        self._reset_edge_attributes('pre_traffic')
        if routing is None:
            routing = self.sp_routing
        for i in self.G.nodes():
            if i == 'graph_data': continue
            for j in self.G.nodes():
                if j == 'graph_data' or i == j: continue
                traffic = self.traffic_demand[i][j]
                if traffic == 0: continue
                for u, v in pairwise_iteration(routing[i][j]):
                    self.G[u][v]['pre_traffic'] += traffic
        for (i,j) in self.G.edges():
            self.G[i][j]['pre_traffic'] /= self.G[i][j]['capacity']
        # self._normalize_traffic()

    def _get_link_traffic(self, routing=None):
        self._distribute_pre_link_traffic(routing)
        self._distribute_link_traffic(routing)
        if self.routing != "flow_split":
            link_traffic = [0]*self.n_links
            for i,j in self.G.edges():
                link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
            self.link_traffic = link_traffic
            self.mean_traffic = np.mean(link_traffic)
            self.get_weights()

    def get_link_traffic(self):
        link_traffic = [0]*self.n_links
        for i,j in self.G.edges():
            link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
        return link_traffic

    def get_link_probs(self):
        traffic_in = np.sum(self.traffic_demand, axis=0)
        traffic_out = np.sum(self.traffic_demand, axis=1)
        node_p_in = list(traffic_in / np.sum(traffic_in))
        node_p_out = list(traffic_out / np.sum(traffic_out))
        self.p_in = [0]*self.n_links
        self.p_out = [0]*self.n_links
        for i,j in self.G.edges():
            self.p_in[self.G[i][j]['id']] = node_p_out[i]
            self.p_out[self.G[i][j]['id']] = node_p_in[j]

    # reward function is currently quite simple
    def compute_reward_measure(self, measure=None):
        if measure is None:
            if self.reward_magnitude == 'link_traffic':
                measure = self.link_traffic
            elif self.reward_magnitude == 'weights':
                measure = self.raw_weights
        
        if self.base_reward == 'mean_times_std':
            return np.mean(measure) * np.std(measure)
        elif self.base_reward == 'mean':
            return np.mean(measure)
        elif self.base_reward == 'std':
            return np.std(measure)
        elif self.base_reward == 'diff_min_max':
            return np.max(measure) - np.min(measure)
        elif self.base_reward == 'min_max':
            # print('最大利用率',np.max(measure))
            return np.max(measure)


    def _compute_reward(self, current_reward_measure=None):
        if current_reward_measure is None:
            # current_reward_measure = self.max_measure
            current_reward_measure = self.compute_reward_measure()

        if self.reward_computation == 'value':
            reward = - current_reward_measure
        elif self.reward_computation == 'change':
            reward = self.reward_measure - current_reward_measure

        self.reward_measure = current_reward_measure
        
        return reward

# if __name__ == '__main__':
#     env = Environment()
#     # env.initialize_environment()
#
#     # env._get_link_traffic()
#
#
#     # print(env.env_type)
#     # print(env.link_traffic)
#
#     # print(env.most_loaded_link(4))
#     # print(env.SR_nodes_choose())
#     # typechange_capacity = {i: {j: list() for j in range(env.G.number_of_nodes() - 1) if j != i} for i in
#     #               range(env.G.number_of_nodes() - 1)}
#     # for (i,j) in env.G.edges:
#     #     typechange_capacity[i][j]=env.G[i][j]['capacity']
#     # print(typechange_capacity)
#     # a = typechange_capacity[0][1] * 0.1
#     # print(a)
#
#     # print("\n")
#     #
#     # print(env.G.nodes)
#     #
#     # print(env.most_loaded_link(4))
#     # print(env.nodes_degree(env.G))
#     # print(env.subpath_1())
#     # print(env.subpath_2())
#     # print(env.subpath_3())
#     # path = env.all_path_sets()
#     # print('path',path)
#     # print(env.try_all_path_sets())
#     # print(env.all_path_sets)
#
#     # for node in env.G.nodes:
#     #     print(env.G.nodes[node]['is_SR'])
#     # print(env.G.number_of_nodes())
#     # print(env.nodes_degree(2))
#     # print(env.get_link_traffic())
#     # nx.draw(env.initialize_environment(),with_labels=True)
#     # plt.show()
#     # print(env.nodes_betweenness(env.G))
#     env.split_ratio()
#     env.compute_reward_measure()
#     # flow_ratio = {i: {j: {p: list() for p in range(len(env.all_path_sets[i][j]))} for j in range(env.G.number_of_nodes() - 1) if j != i} for i in range(env.G.number_of_nodes() - 1)}
#     # print(flow_ratio)
#     # print(len(env.all_path_sets[0][6]))
#     # edge_ratio = {i: {j: {p: list()for p in range(len(env.all_path_sets[i][j]))} for j in range(env.G.number_of_nodes() - 1) if j != i} for i in
#     #               range(env.G.number_of_nodes() - 1)}
#     # print(edge_ratio)
#     # env.compute_reward_measure()


