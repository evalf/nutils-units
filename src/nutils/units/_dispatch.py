from inspect import getmodule
from functools import partial, partialmethod


class DispatchTable(dict):
    """Table of handler functions.

    The dispatch table is a dictionary that maps the fully qualified name of a
    callable to a handler function that should be called with the callable as
    the first positional argument, followed by its arguments. Handlers are
    registered by the fully qualified domain name via the .register method:

    >>> dispatch_table = DispatchTable()
    >>> @dispatch_table.register("_operator.pow")
    ... def handle_pow(cls, func, x, y):
    ...     if not isinstance(x, cls) or not isinstance(y, cls):
    ...         return NotImplemented
    ...     return cls(func(float(x), float(y)))

    In this (rather contrived) example we created a handler that extends the
    power operator to operate on string arguments. To obtain the relevant
    handler we use the .bind_method method:

    >>> import operator
    >>> class MyStringType(str):
    ...     __pow__ = dispatch_table.bind_method(operator.pow)
    >>> MyStringType("5") ** MyStringType("2")
    '25.0'

    Note that by registering the fully qualified name of the callable, rather
    than the callable itself, we are able to prepare a table for any number of
    external functions without ever importing any third party modules.
    """

    def register(self, qualname, func=None):
        if func is None:
            return partial(self.register, qualname)
        self[qualname] = func
        return func

    @staticmethod
    def _bind_method(self, handler, func, *args, **kwargs):
        # NOTE: `self` refers to the method's owner, not DispatchTable
        return handler(type(self), func, self, *args, **kwargs)

    def bind_method(self, func):
        handler = self[get_qualname(func)]
        return partialmethod(self._bind_method, handler, func)

    @staticmethod
    def _bind_reverse(self, handler, func, other, **kwargs):
        # NOTE: `self` refers to the method's owner, not DispatchTable
        return handler(type(self), func, other, self, **kwargs)

    def bind_method_and_reverse(self, func):
        handler = self[get_qualname(func)]
        return partialmethod(self._bind_method, handler, func), partialmethod(
            self._bind_reverse, handler, func
        )

    def call_or_notimp(self, cls, func, args, kwargs):
        try:
            handler = self[get_qualname(func)]
        except KeyError:
            return NotImplemented
        else:
            return handler(cls, func, *args, **kwargs)


def get_qualname(func):
    """Get the qualified domain name of a callable.

    This function returns the string to be used in DispatchTable.register to
    identify a callable.
    """

    return getmodule(func).__name__ + "." + func.__qualname__
