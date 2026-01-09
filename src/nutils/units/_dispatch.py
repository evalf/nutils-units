from inspect import getmodule
from functools import partial, partialmethod


class DispatchTable(dict):
    """Table of handler functions.

    The dispatch table is a dictionary that maps the fully qualified name of a
    callable to a handler function that should be called with the callable as
    the first positional argument, followed by its arguments. Handlers are
    registered via the .register method:

    >>> dispatch_table = DispatchTable()
    >>> @dispatch_table.register("math.sqrt")
    ... def handle_root(op, x):
    ...     return str(op(float(x)))

    In this (rather contrived) example we create a handler that extends the
    math.sqrt function to operate on string arguments. To obtain the relevant
    handler we use the .bind method:

    >>> import math
    >>> f = dispatch_table.bind(math.sqrt)
    >>> f("4")
    '2.0'

    Note that by registering the fully qualified name of the callable, rather
    than the callable itself, we are able to prepare a table for any number of
    external functions without ever importing any third party modules.
    """

    def register(self, qualname):
        return partial(_setitem_retvalue, self, qualname)

    def bind(self, func):
        return partial(self[get_qualname(func)], func)

    def bind_method(self, func):
        return partialmethod(self.bind(func))

    def bind_method_and_reverse(self, func):
        bound = self.bind(func)
        return partialmethod(bound), partialmethod(_swap, bound)

    def call_or_notimp(self, func, args, kwargs):
        try:
            bound = self.bind(func)
        except KeyError:
            return NotImplemented
        else:
            return bound(*args, **kwargs)


def get_qualname(func):
    """Get the qualified domain name of a callable.

    This function returns the string to be used in DispatchTable.register to
    identify a callable.
    """

    return getmodule(func).__name__ + "." + func.__qualname__


def _setitem_retvalue(d, k, v):
    d[k] = v
    return v


def _swap(self, func, arg):
    return func(arg, self)
