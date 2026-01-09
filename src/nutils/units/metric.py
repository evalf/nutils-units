from ._powers import Powers
from ._units import Units
from .core import Monomial, unwrap
from .error import DimensionError


units = Units()


class UMonomial(Monomial):
    """Dimensional wrapper with support for SI units.

    This class extends the Monomial with two functionalities that both
    reference a global units table.

    1. String formatting

    Any unit-wrapped object can be string formatted by substituting the usual
    "f" float formatter by the desired SI unit:

    >>> F = parse('5N')
    >>> A = parse('2mm2')
    >>> f"pressure: {F/A:.1MPa}"
    'pressure: 2.5MPa'

    2. Unit division

    Any unit-wrapped object can be divided by a string containing an SI unit to
    obtain the value against that unit.

    >>> V = parse('4cm')**3
    >>> V / 'mL'
    64.0

    This is more than a shorthand for `V / parse("nL")`, as the string division
    also checks that the result is indeed dimensionless.

    >>> V / 'kg'
    Traceback (most recent call last):
        ...
    nutils.units.error.DimensionError: 'kg' is not a unit for dimension L3
    """

    def __truediv__(self, other):
        if type(other) is not str:
            return super().__truediv__(other)
        val1, dim1 = self.unwrap()
        val2, dim2 = unwrap(parse(other))
        if dim1 != dim2:
            raise DimensionError(f"{other!r} is not a unit for dimension {dim1}")
        return val1 / val2

    def __format__(self, format_spec):
        if not format_spec:
            return repr(self)
        head, tail = _splitnum(format_spec)
        return (self / tail).__format__(head + "f") + tail


def parse(s):
    """Parse any string into a dimensional quantity.

    Valid strings consist of a numerical prefix, followed by a unit that
    follows the same notational conventions as layed out in power.Powers.

    >>> F = parse("5.3g*cm/ms2")
    >>> F / 'N'
    53.0

    Scientific notation is not supported. Instead of "2e3m" (invalid), use the
    appropriate metric prefix: "2km".
    """

    if not isinstance(s, str):
        raise ValueError(f"expected a str, received {type(s).__name__}")
    head, tail = _splitnum(s)
    q = float(head or 1)
    for expr, power in Powers(tail).items():
        try:
            v = getattr(units, expr)
        except AttributeError:
            raise ValueError(f"invalid unit {expr!r}") from None
        q *= v**power
    return q


def _splitnum(s):
    tail = s.lstrip("+-0123456789.,")
    head = s[: len(s) - len(tail)]
    return head, tail


# base units

units.register("m", UMonomial(1.0, "L"))
units.register("s", UMonomial(1.0, "T"))
units.register("g", UMonomial(1e-3, "M"))
units.register("A", UMonomial(1.0, "I"))
units.register("K", UMonomial(1.0, "Θ"))
units.register("mol", UMonomial(1.0, "N"))
units.register("cd", UMonomial(1.0, "J"))

# derived units

units.register("N", parse("kg*m/s2"))  # newton
units.register("Pa", parse("N/m2"))  # pascal
units.register("J", parse("N*m"))  # joule
units.register("W", parse("J/s"))  # watt
units.register("Hz", parse("/s"))  # hertz
units.register("C", parse("A*s"))  # coulomb
units.register("V", parse("J/C"))  # volt
units.register("F", parse("C/V"))  # farad
units.register("Ω", parse("V/A"))  # ohm
units.register("S", parse("/Ω"))  # siemens
units.register("Wb", parse("V*s"))  # weber
units.register("T", parse("Wb/m2"))  # tesla
units.register("H", parse("Wb/A"))  # henry
units.register("lm", parse("cd"))  # lumen
units.register("lx", parse("lm/m2"))  # lux
units.register("Bq", parse("/s"))  # becquerel
units.register("Gy", parse("J/kg"))  # gray
units.register("Sv", parse("J/kg"))  # sievert
units.register("kat", parse("mol/s"))  # katal

# other units

units.register("au", parse("149597870700m"))  # astronomical unit
units.register("ha", parse("hm2"))  # hectare
units.register("L", parse("dm3"))  # liter
units.register("t", parse("1000kg"))  # ton
units.register("Da", parse("1.66053904020yg"))  # dalton
units.register("eV", parse(".1602176634aJ"))  # electronvolt
units.register("min", parse("60s"), prefix="")  # minute
units.register("h", parse("60min"), prefix="")  # hour
units.register("day", parse("24h"), prefix="")  # day
units.register("in", parse("25.4mm"), prefix="")  # inch (no prefixes)
units.register("°C", units.K, prefix="")  # degrees Celcius
units.register("°F", units.K * (5 / 9), prefix="")  # degrees Fahrenheit
