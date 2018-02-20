from collections import OrderedDict

import numpy as np


class SingleValue:
    def __init__(self, val):
        self.value = val

    def num_prams(self):
        return 0

    def transform(self, num):
        return self.value

    def get_lb(self):
        return []

    def get_ub(self):
        return []


class ContinuousVariable:
    def __init__(self, tup):
        self.trans = tup[0]
        self.lb = float(tup[1])
        self.ub = float(tup[2])

    def num_params(self):
        return 1

    def transform(self, num):
        return self.trans(num[0])

    def get_lb(self):
        return [self.lb]

    def get_ub(self):
        return [self.ub]


class DiscreteVariable:
    def __init__(self, lst):
        self.values = lst
        lb = 0.0
        ub = len(lst) - np.finfo(float).eps  # not inclusive range
        self.variable = ContinuousVariable((int, lb, ub))

    def num_params(self):
        return 1

    def transform(self, num):
        idx = self.variable.transform(num)
        return self.values[idx]

    def get_lb(self):
        return self.variable.get_lb()

    def get_ub(self):
        return self.variable.get_ub()


class ListVariable:
    def __init__(self, lst):
        self.values = []
        for index, item in enumerate(lst):
            self.values.append(Node(index, item))

        self.lb = []
        self.ub = []
        for node in self.values:
            self.lb.extend(node.get_lb())
            self.ub.extend(node.get_ub())

    def num_params(self):
        return sum([n.num_params() for n in self.values])

    def transform(self, tr_lst):
        return [node.transform([num]) for node, num in zip(self.values, tr_lst)]

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub


class Node:
    def __init__(self, key, value):
        self.key = key
        if isinstance(value, tuple) and callable(value[0]):
            self.value = ContinuousVariable(value)
        elif isinstance(value, tuple):
            self.value = DiscreteVariable(value)
        elif isinstance(value, list):
            self.value = ListVariable(value)

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return str(self.key)

    def num_params(self):
        return self.value.num_params()

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

        self.lb = []
        self.ub = []
        for node in self.nodes:
            self.lb.extend(node.get_lb())
            self.ub.extend(node.get_ub())

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
        for i, node in enumerate(self.nodes):
            trans_params = []
            for j in range(node.num_params()):
                trans_params.append(lst[i + j])
            transformed[node.key] = node.transform(trans_params)
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


class pizza():
    def __init__(self):
        self.values = 1

    def itr(self):
        yield self.values


def main():
    params = OrderedDict()
    params['a'] = (1, 2, 3, 4, 5, 6)
    params['b'] = (int, 2, 3)

    ss = SearchSpace(params)
    print ss.nodes
    print ss.get_lb()
    print ss.get_ub()
    print ss.transform([0.5, 1, 6.5, 2.5])


if __name__ == '__main__':
    main()
