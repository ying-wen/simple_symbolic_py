#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from abc import abstractmethod


def isNumber(x):
    return isinstance(x, (int, long, float))


def isValid(x):
    return isinstance(x, (Block, int, long, float))


def isBlock(x):
    return isinstance(x, Block)


def isConstant(x):
    return isinstance(x, Constant)


def isVar(x):
    return isinstance(x, Var)


class Block(object):

    def __init__(self):

        # caches output after call of forward

        self.output = None
        self.var_table = {}

    def _add_to_var_table(self, *args):
        for arg in args:
            if isBlock(arg):
                if isConstant(arg) is False:
                    for key in arg.var_table.keys():
                        self.var_table[key] = arg.var_table[key]

    def eval(self, **kargs):
        for var_name in kargs.keys():
            if self.var_table.get(var_name) != None:
                if isNumber(kargs[var_name]):
                    self.var_table[var_name].set(kargs[var_name])
                else:
                    raise ValueError("TypeError: cannot use '%s' objects"
                             % type(kargs[var_name]))
            else:
                raise ValueError("TypeError: var '%s' undefined"
                                 % type(var_name))
        return self.forward()

    def eval_gradient(self, var_name):
        if self.var_table.get(var_name) != None:
            self.forward()
            self.backward()
            return self.var_table[var_name].gradParam
        else:
            raise ValueError("TypeError: var '%s' undefined"
                             % type(var_name))

    @abstractmethod
    def deriv(self, var_name):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def forward_partial(self, var_name):
        pass

    @abstractmethod
    def backward(self, gradient=1):
        pass

    @abstractmethod
    def update(self, learning_rate=1):
        pass

    def _check_other(self, other):
        if isValid(other):
            if isNumber(other):
                return Constant(other)
            return other
        else:
            raise ValueError("TypeError: cannot use '%s' objects"
                             % type(other))

    def __mul__(self, other):
        return Mul(self, self._check_other(other))

    def __pow__(self, other):
        return Pow(self, self._check_other(other))

    def __rpow__(self, other):
        return Pow(self._check_other(other), self)

    def __div__(self, other):
        return Div(self, self._check_other(other))

    def __add__(self, other):
        return Add(self, self._check_other(other))

    def __sub__(self, other):
        return Sub(self, self._check_other(other))

    def __rmul__(self, other):
        return Mul(self, self._check_other(other))

    def __rdiv__(self, other):
        return Div(self._check_other(other), self)

    def __radd__(self, other):
        return Add(self._check_other(other), self)

    def __rsub__(self, other):
        return Sub(self._check_other(other), self)

    def __neg__(self):
        return Neg(self)


class ParamBlock(Block):

    def __init__(self):
        Block.__init__(self, clip = 10)
        self.clip = clip
        # caches output after call of forward
        self.param = None
        self.gradParam = 0

    def update(self, learning_rate):
        if self.gradParam < self.clip:
            self.param += self.gradParam
        else:
            self.param += self.clip
        self.reset_gradient()

    def reset_gradient(self):
        self.gradParam = 0

    def set(self, value):
        if isNumber(value):
            self.param = float(value)
        else:
            raise ValueError("TypeError: cannot set '%s' object as param value"
                              % type(value))


class Constant(Block):

    def __init__(self, arg):
        Block.__init__(self)
        self.output = arg
        self._name = str(arg)

    def forward(self):
        return self.output

    def forward_partial(self, var_name):
        return self.forward()

    def backward(self, gradient):
        pass

    def deriv(self, var_name):
        return self

    def update(self, learning_rate):
        pass

    def __str__(self):
        return self._name


class Var(ParamBlock):

    def __init__(self, name, clip=10):
        ParamBlock.__init__(self, clip)
        self._name = name
        self.var_table[name] = self


    @property
    def name(self):
        return self._name

    def forward(self):
        if self.param == None:
            raise ValueError("TypeError: param '%s' is unset"
                             % self._name)
        self.output = self.param
        return self.output

    def forward_partial(self, var_name):
        if self._name == var_name:
            return self
        else:
            return Constant(self.forward())

    def backward(self, gradient=1, clip=10):
        self.gradParam += gradient

    def deriv(self, var_name):
        if self.name == var_name:
            return Constant(1)
        else:
            return Constant(self.forward())

    def __str__(self):
        return self._name


class Add(Block):

    def __init__(self, arg1, arg2):
        Block.__init__(self)
        self.arg1 = arg1
        self.arg2 = arg2
        self._add_to_var_table(arg1, arg2)

    def forward(self):
        self.output = self.arg1.forward() + self.arg2.forward()
        return self.output

    def forward_partial(self, var_name):
        return self.arg1.forward_partial(var_name) \
            + self.arg2.forward_partial(var_name)

    def backward(self, gradient=1):
        self.arg1.backward(gradient)
        self.arg2.backward(gradient)

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def deriv(self, var_name, gradient=1):
        if self.arg1.var_table.get(var_name) != None \
            and self.arg2.var_table.get(var_name) != None:
            return self.arg1.deriv(var_name) + self.arg2.deriv(var_name)
        if self.arg1.var_table.get(var_name) == None \
            and self.arg2.var_table.get(var_name) == None:
            return Constant(self.forward()).deriv()
        if self.arg1.var_table.get(var_name) == None:
            return self.arg2.deriv(var_name)
        if self.arg2.var_table.get(var_name) == None:
            return self.arg1.deriv(var_name)

    def __str__(self):
        return '%s + %s' % (str(self.arg1), str(self.arg2))


class Mul(Block):

    def __init__(self, arg1, arg2):
        Block.__init__(self)
        self.arg1 = arg1
        self.arg2 = arg2
        self._add_to_var_table(arg1, arg2)

    def forward(self):
        self.output = self.arg1.forward() * self.arg2.forward()
        return self.output

    def forward_partial(self, var_name):
        return self.arg1.forward_partial(var_name) \
            * self.arg2.forward_partial(var_name)

    def backward(self, gradient=1):
        self.arg1.backward(gradient * self.arg2.output)
        self.arg2.backward(gradient * self.arg1.output)

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def deriv(self, var_name):
        deriv1 = self.arg1.forward_partial(var_name) \
            * self.arg2.deriv(var_name)
        deriv2 = self.arg2.forward_partial(var_name) \
            * self.arg1.deriv(var_name)
        if self.arg1.var_table.get(var_name) != None \
            and self.arg2.var_table.get(var_name) != None:
            return deriv1 + deriv2
        if self.arg1.var_table.get(var_name) == None \
            and self.arg2.var_table.get(var_name) == None:
            return Constant(self.forward())
        if self.arg1.var_table.get(var_name) == None:
            return deriv1
        if self.arg2.var_table.get(var_name) == None:
            return deriv2

    def __str__(self):
        if isinstance(self.arg2, (Add, Sub)) and isinstance(self.arg1,
                (Add, Sub)):
            return '(%s) * (%s)' % (str(self.arg1), str(self.arg2))
        if isinstance(self.arg1, (Add, Sub)):
            return '(%s) * %s' % (str(self.arg1), str(self.arg2))
        if isinstance(self.arg2, (Add, Sub)):
            return '%s * (%s)' % (str(self.arg1), str(self.arg2))
        return '%s * %s' % (str(self.arg1), str(self.arg2))


class Sub(Block):

    def __init__(self, arg1, arg2):
        Block.__init__(self)
        self.arg1 = arg1
        self.arg2 = arg2
        self._add_to_var_table(arg1, arg2)

    def forward(self):
        self.output = self.arg1.forward() - self.arg2.forward()
        return self.output

    def forward_partial(self, var_name):
        return self.arg1.forward_partial(var_name) \
            - self.arg2.forward_partial(var_name)

    def backward(self, gradient=1):
        self.arg1.backward(gradient)
        self.arg2.backward(gradient * -1)

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def deriv(self, var_name, gradient=1):
        if self.arg1.var_table.get(var_name) != None \
            and self.arg2.var_table.get(var_name) != None:
            return self.arg1.deriv(var_name) - self.arg2.deriv(var_name)
        if self.arg1.var_table.get(var_name) == None \
            and self.arg2.var_table.get(var_name) == None:
            return Constant(self.forward())
        if self.arg1.var_table.get(var_name) == None:
            return -1 * self.arg2.deriv(var_name)
        if self.arg2.var_table.get(var_name) == None:
            return self.arg1.deriv(var_name)

    def __str__(self):
        return '%s - %s' % (str(self.arg1), str(self.arg2))


class Div(Block):

    def __init__(self, arg1, arg2):
        Block.__init__(self)
        self.arg1 = arg1
        self.arg2 = arg2
        self._add_to_var_table(arg1, arg2)

    def forward(self):
        self.output = self.arg1.forward() / self.arg2.forward()
        return self.output

    def forward_partial(self, var_name):
        return self.arg1.forward_partial(var_name) \
            / self.arg2.forward_partial(var_name)

    # f = x/y
    # df/dx = 1/y
    # df/dy = -(y/x^2)
    def backward(self, gradient=1):
        self.arg1.backward(gradient * (1.0 / self.arg2.output))
        self.arg2.backward(gradient * (self.output / self.arg2.output)
                           * -1)

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def deriv(self, var_name):
        dervi1 = self.arg1.forward_partial(var_name) \
            * self.arg2.forward_partial(var_name) ** -2 \
            * self.arg2.deriv(var_name)
        dervi2 = 1 / self.arg2.forward_partial(var_name) \
            * self.arg1.deriv(var_name)
        if self.arg1.var_table.get(var_name) != None \
            and self.arg2.var_table.get(var_name) != None:
            return dervi1 + dervi2
        if self.arg1.var_table.get(var_name) == None \
            and self.arg2.var_table.get(var_name) == None:
            return Constant(self.forward())
        if self.arg1.var_table.get(var_name) == None:
            return dervi1
        if self.arg2.var_table.get(var_name) == None:
            return dervi2

    def __str__(self):
        if isinstance(self.arg2, (Add, Sub)) and isinstance(self.arg1,
                (Add, Sub)):
            return '(%s) / (%s)' % (str(self.arg1), str(self.arg2))
        if isinstance(self.arg1, (Add, Sub)):
            return '(%s) / %s' % (str(self.arg1), str(self.arg2))
        if isinstance(self.arg2, (Add, Sub)):
            return '%s / (%s)' % (str(self.arg1), str(self.arg2))
        return '%s / %s' % (str(self.arg1), str(self.arg2))


class Pow(Block):

    def __init__(self, arg1, arg2):
        Block.__init__(self)
        self.arg1 = arg1
        self.arg2 = arg2
        self._add_to_var_table(arg1, arg2)

    def forward(self):
        self.output = self.arg1.forward() ** self.arg2.forward()
        return self.output

    def forward_partial(self, var_name):
        return self.arg1.forward_partial(var_name) \
            ** self.arg2.forward_partial(var_name)

    def deriv(self, var_name):
        deriv1 = Log(self.arg1.forward_partial(var_name)) \
            * self.arg1.forward_partial(var_name) \
            ** self.arg2.forward_partial(var_name) \
            * self.arg2.deriv(var_name)
        deriv2 = self.arg2.forward_partial(var_name) \
            * self.arg1.forward_partial(var_name) \
            ** (self.arg2.forward_partial(var_name) - 1) \
            * self.arg1.deriv(var_name)
        if self.arg1.var_table.get(var_name) != None \
            and self.arg2.var_table.get(var_name) != None:
            return deriv1 + deriv2
        if self.arg1.var_table.get(var_name) == None \
            and self.arg2.var_table.get(var_name) == None:
            return _check_other(self.forward())
        if self.arg1.var_table.get(var_name) == None:
            return deriv1
        if self.arg2.var_table.get(var_name) == None:
            return deriv2

    # f = x^y
    # df/dy = ln(x)x^y
    # df/dx = yx^(y - 1)
    def backward(self, gradient=1):
        self.arg1.backward(gradient * self.arg2.output
                           * self.arg1.output ** (self.arg2.output - 1))
        self.arg2.backward(gradient * math.log(self.arg1.output)
                           * self.output)

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def __str__(self):
        if isinstance(self.arg1, (Constant, Var, Pow)) \
            and isinstance(self.arg2, (Constant, Var, Pow)):
            return '%s ** %s' % (str(self.arg1), str(self.arg2))
        else:
            if not isinstance(self.arg1, (Constant, Var, Pow)) \
                and not isinstance(self.arg2, (Constant, Var, Pow)):
                return '(%s) ** (%s)' % (str(self.arg1), str(self.arg2))
            if not isinstance(self.arg1, (Constant, Var, Pow)):
                return '(%s) ** %s' % (str(self.arg1), str(self.arg2))
            if not isinstance(self.arg2, (Constant, Var, Pow)):
                return '%s ** (%s)' % (str(self.arg1), str(self.arg2))


class Neg(Block):

    def __init__(self, arg):
        Block.__init__(self)
        self.arg = arg
        self._add_to_var_table(arg)

    def forward(self):
        self.output = self.arg.forward() * -1
        return self.output

    def forward_partial(self, var_name):
        return self.arg.forward_partial(var_name) * -1

    def backward(self, gradient=1):
        self.arg.backward(gradient * -1)

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def deriv(self, var_name):
        if self.arg.var_table.get(var_name) == None:
            return Constant(self.forward_partial(var_name))
        else:
            return -self.arg.deriv(var_name)

    def __str__(self):
        if isinstance(self.arg, (Constant, Var, Pow)):
            return '-%s' % str(self.arg)
        else:
            return '-(%s)' % str(self.arg)


class Log(Block):

    def __init__(self, arg):
        Block.__init__(self)
        self.arg = arg
        self._add_to_var_table(arg)

    def forward(self):
        self.output = math.log(self.arg.forward())
        return self.output

    def forward_partial(self, var_name):
        return Log(self.arg.forward_partial(var_name))

    def backward(self, gradient=1):
        self.arg.backward(gradient * (1 / self.arg.output))

    def update(self, learning_rate):
        self.arg1.update(learning_rate)
        self.arg2.update(learning_rate)

    def deriv(self, var_name):
        if arg.var_table.get(var_name) == None:
            return Constant(self.forward_partial(var_name))
        else:
            return 1 / self.arg.deriv(var_name)

    def __str__(self):
        return 'ln(%s)' % str(self.arg)
