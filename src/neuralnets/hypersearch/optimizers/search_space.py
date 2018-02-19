from collections import OrderedDict

import numpy as np


class ContinuousVariable:
    def __init__(self, tup):
        self.trans = tup[0]
        self.lb = tup[1]
        self.ub = tup[2]

    def transform(self, num):
        return self.trans(num)

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub


class DiscreteVariable:
    def __init__(self, lst):
        self.values = lst
        self.lb = 0
        self.ub = len(lst) - np.finfo(float).eps  # not inclusive range

    def transform(self, num):
        idx = int(num)
        return self.values[idx]

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub


class Node:
    def __init__(self, key, value):
        self.key = key
        if isinstance(value, tuple):
            self.value = ContinuousVariable(value)
        elif isinstance(value, list):
            self.value = DiscreteVariable(value)

    def get_key(self):
        return self.key

    def transform(self, num):
        return self.value.transform(num)

    def get_lb(self):
        return self.value.get_lb()

    def get_ub(self):
        return self.value.get_ub()


class SearchSpace:
    def __init__(self, dic):
        self.nodes = self.create_nodes(dic)
        self.lb = np.array([n.get_lb() for n in self.nodes])
        self.ub = np.array([n.get_ub() for n in self.nodes])

    def create_nodes(self, dic):
        nodes = []
        for key, value in dic.iteritems():
            if not isinstance(value, dict):
                node = Node(key, value)
                nodes.append(node)
            else:
                node = Node(key, value.keys())
                nodes.append(node)

                for val in value.values():
                    new_nodes = self.create_nodes(val)
                    nodes.extend(new_nodes)
        return nodes

    def transform(self, lst):
        transformed = OrderedDict()
        for i in range(len(lst)):
            transformed[self.nodes[i].key] = self.nodes[i].transform(lst[i])
        return transformed

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub


def param_decorator(f, ss):
    def decoratet_f(*args):
        params = ss.transform(args)
        return f(**params)

    return decoratet_f
