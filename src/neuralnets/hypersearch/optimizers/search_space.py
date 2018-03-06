import numpy as np
from collections import OrderedDict, Iterable


class BaseVariable(object):

    def get_value(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class BasePrimitiveVariable(BaseVariable):
    def __init__(self, lb, ub):
        self.internal_state = None
        self.lb = lb
        self.ub = ub

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub

    def set_internal_state(self, num):
        self.internal_state = num

    def get_value(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __str__(self):
        string = 'lb=' + str(self.lb) + ',ub=' + str(self.ub) + ',state=' + str(self.internal_state)
        return string

    def __repr__(self):
        return str(self)


class ContinuousVariable(BasePrimitiveVariable):

    def __init__(self, lb, ub, type=int):
        self.type = type
        super(ContinuousVariable, self).__init__(lb, ub)

    def get_value(self):
        return self.type(self.internal_state)

    def __iter__(self):
        yield self

    def __str__(self):
        string = super(ContinuousVariable, self).__str__()
        string = 'cvar(' + string + ')'
        return string


class DiscreteVariable(BasePrimitiveVariable):

    def __init__(self, *args):
        self.choices = args
        lb = 0
        ub = len(args) - np.finfo(np.float32).eps
        super(DiscreteVariable, self).__init__(lb, ub)

    def get_value(self):
        choice = self.choices[int(self.internal_state)]
        if isinstance(choice, BaseVariable):
            return choice.get_value()
        else:
            return choice

    def __iter__(self):
        yield self
        for i in self.choices:
            if isinstance(i, BaseVariable):
                for j in i:
                    yield j

    def __str__(self):
        string = super(DiscreteVariable, self).__str__()
        string = 'dvar(' + string + ',choices=' + str(self.choices) + ')'
        return string


class BaseContainerVariable(BaseVariable):
    def __init__(self, values):
        self.values = values

    def get_value(self):
        raise NotImplementedError

    def __iter__(self):
        for i in self.values:
            if isinstance(i, BaseVariable):
                for j in i:
                    yield j

    def __repr__(self):
        return str(self)


class DictVariable(BaseContainerVariable):

    def __init__(self, dictionary):
        self.keys = dictionary.keys()
        super(DictVariable, self).__init__(dictionary.values())

    def get_value(self):
        output = OrderedDict()
        for i, j in zip(self.keys, self.values):
            if isinstance(j, BaseVariable):
                output[i] = j.get_value()
            else:
                output[i] = j
        return output

    def __str__(self):
        dictionary = {i: j for i, j in zip(self.keys, self.values)}
        string = 'DictVar(' + str(dictionary) + ')'
        return string


class ListVariable(BaseContainerVariable):
    def __init__(self, lst):
        super(ListVariable, self).__init__(lst)

    def get_value(self):
        output = []
        for i in self.values:
            if isinstance(i, BaseVariable):
                output.append(i.get_value())
            else:
                output.append(i)
        return output

    def __str__(self):
        string = 'ListVar(' + str(self.values) + ')'
        return string


class SearchSpace(object):
    def __init__(self, params):
        self.variables = self.init_variables(params)

    def init_variables(self, param):
        if isinstance(param, dict):
            dic = OrderedDict()
            for key, value in param.iteritems():
                dic[key] = self.init_variables(value)
            return DictVariable(dic)

        elif isinstance(param, list):
            lst = []
            for i in param:
                lst.append(self.init_variables(i))
            return ListVariable(lst)

        elif isinstance(param, tuple) or isinstance(param, Choice):
            lst = []
            for i in param:
                lst.append(self.init_variables(i))
            return DiscreteVariable(*lst)

        elif isinstance(param, set) or isinstance(param, Variable):
            return ContinuousVariable(*param)

        else:
            return param

    def get_lb(self):
        return np.array([i.get_lb() for i in self.variables])

    def get_ub(self):
        return np.array([i.get_ub() for i in self.variables])

    def transform(self, nums):
        for num, var in zip(nums, self.variables):
            var.set_internal_state(num)

        return self.variables.get_value()


class Choice(Iterable):
    def __init__(self, *args):
        self.args = args

    def __iter__(self):
        for i in self.args:
            yield i


class Variable(Iterable):
    def __init__(self, lb, ub, type):
        self.args = [lb, ub, type]

    def __iter__(self):
        for i in self.args:
            yield i


def param_decorator(f, ss):
    def decorated_f(*args):
        params = ss.transform(args)
        return f(**params)

    return decorated_f


def main():
    par = Variable(1, 5, int)
    cvar = ContinuousVariable(*par)
    print cvar
    return

    params = OrderedDict()
    params['epochs'] = {1, 10}
    params['neurons'] = (
        [{1, 4}],
        [{1, 4}, {1, 4}]
    )

    st = SearchSpace(params)
    print st.variables
    lb = st.get_lb()
    ub = st.get_ub()
    print st.transform(np.random.uniform(lb, ub))

    return
    # params = DescreteVariable(6, ContinousVariable(2, 10))
    print params

    # return
    lb = [i.get_lb() for i in params]
    ub = [i.get_ub() for i in params]
    print lb, ub

    for i in params:
        i.set_internal_state(1.2)

    print params.get_value()
    # for i in params:
    #     print i.get_value()


if __name__ == '__main__':
    main()
