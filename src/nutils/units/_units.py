PREFIX = dict(
    Y=1e24,
    Z=1e21,
    E=1e18,
    P=1e15,
    T=1e12,
    G=1e9,
    M=1e6,
    k=1e3,
    h=1e2,
    d=1e-1,
    c=1e-2,
    m=1e-3,
    μ=1e-6,
    n=1e-9,
    p=1e-12,
    f=1e-15,
    a=1e-18,
    z=1e-21,
    y=1e-24,
)


class Units:
    """Registry of numerical values.

    The Units object is a registry of numerical values. Values can be assigned
    under any name that is not previously used, and cannot be removed.

    >>> u = Units()
    >>> u.register("m", 5.)
    >>> u.m
    5.0

    Intended for units, the registry by default adds all metric prefixes with
    approprately scaled values. If "m" represents the number 5, then "km"
    represents 1000 times that number:

    >>> u.km
    5000.0

    Attempting to redefine a previously assigned value results in an error and
    leaves the registry unmodified.

    >>> u.register("am", 10.)
    Traceback (most recent call last):
      ...
    ValueError: registration failed: unit collides with 'am'
    """

    def register(self, name, value, prefix=tuple(PREFIX)):
        d = self.__dict__
        new = [(name, value)]
        new.extend((p + name, value * PREFIX[p]) for p in prefix)
        collisions = ", ".join(repr(s) for s, v in new if s in d and d[s] != v)
        if collisions:
            raise ValueError(f"registration failed: unit collides with {collisions}")
        d.update(new)
