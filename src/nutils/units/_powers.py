from fractions import Fraction


class Powers:
    """Immutable counter with fractional values.

    This class is similar to `collections.Counter`, with two key differences:

    - Values are instances of `fractions.Fraction`;
    - The collection is immutable.

    A `Powers` instance can be created from a string, which sets its counter
    equal to 1:
    >>> Powers("A")
    Powers(A)

    We can then use multiplication and division to scale all counts:
    >>> Powers("A") * 3
    Powers(A3)

    And we can use addition and subtraction to combine powers:
    >>> Powers("A") * 3 + Powers("B") / 2
    Powers(A3*B1/2)

    Note that the different items are separated by a `*`. Note what happens
    when we reach a negative count:
    >>> Powers("A") * 3 - Powers("B") * 2
    Powers(A3/B2)

    Rather than making the power negatve, we switch the `*` for a `/`. This
    notation reflects the interpretation of counts as monomial powers.

    When we created a power from a string before, we actually made use of the
    fact that we can parse back the entire string representation now
    established:
    >>> Powers("A3/B2")
    Powers(A3/B2)

    Alternatively we can create it from a dictionary:
    >>> Powers({"A": Fraction(3), "B": Fraction(-2)})
    Powers(A3/B2)

    Or even from an existing powers object:
    >>> Powers(Powers("A"))
    Powers(A)
    """

    def __init__(self, d):
        if isinstance(d, Powers):
            self.__d = d.__d
        elif isinstance(d, str):
            self.__d = _split_factors(d)
        elif isinstance(d, dict) and all(
            isinstance(k, str) and isinstance(v, Fraction) for k, v in d.items()
        ):
            self.__d = {k: v for k, v in d.items() if v}
        else:
            raise ValueError(f"cannot create Powers from {d!r}")

    def __hash__(self):
        return hash(tuple(sorted(self.__d.items())))

    def __eq__(self, other):
        return self is other or isinstance(other, Powers) and self.__d == other.__d

    def __add__(self, other):
        if not isinstance(other, Powers):
            return NotImplemented
        d = self.__d.copy()
        for name, n in other.__d.items():
            d[name] = d.get(name, 0) + n
        return Powers(d)

    def __sub__(self, other):
        if not isinstance(other, Powers):
            return NotImplemented
        d = self.__d.copy()
        for name, n in other.__d.items():
            d[name] = d.get(name, 0) - n
        return Powers(d)

    def __mul__(self, other):
        if isinstance(other, float):
            tmp = Fraction(other).limit_denominator()
            if other == float(tmp):
                other = tmp
        if not isinstance(other, (int, Fraction)):
            try:
                other = other.__index__()
            except Exception:
                return NotImplemented
        return Powers({name: n * other for name, n in self.__d.items()})

    def __truediv__(self, other):
        if isinstance(other, float):
            tmp = Fraction(other).limit_denominator()
            if other == float(tmp):
                other = tmp
        if not isinstance(other, (int, Fraction)):
            return NotImplemented
        return Powers({name: n / other for name, n in self.__d.items()})

    def __neg__(self):
        return Powers({name: -n for name, n in self.__d.items()})

    def __bool__(self):
        return bool(self.__d)

    def items(self):
        return self.__d.items()

    def __str__(self):
        return _join_factors(self.__d)

    def __repr__(self):
        return f"Powers({self})"


def _split_factors(s):
    items = []
    for factors in s.split("*"):
        numer, *denoms = factors.split("/")
        if numer:
            base = numer.rstrip("0123456789")
            num = Fraction(int(numer[len(base) :] or 1))
            items.append((base, num))
        elif items:  # s contains "*/"
            raise ValueError
        for denom in denoms:
            base = denom.rstrip("0123456789")
            if base:
                num = -Fraction(int(denom[len(base) :] or 1))
            else:  # fractional power
                base, num = items.pop()
                num /= int(denom)
            items.append((base, num))
    return dict(items)


def _join_factors(factors):
    s = ""
    for base, power in sorted(
        factors.items(), key=lambda item: item[::-1], reverse=True
    ):
        if power < 0:
            power = -power
            s += "/"
        else:
            s += "*"
        s += base
        if power != 1:
            s += (
                str(power)
                if power.denominator == 1
                else f"{power.numerator}/{power.denominator}"
            )
    return s.lstrip("*")
