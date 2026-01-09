from ._powers import Powers
from .error import DimensionError
from .metric import UMonomial, parse, unwrap

import typing


class Quantity(UMonomial):
    """Scalar quantity.

    The Quantity object is a UMonomial that:

    1. wraps a floating point number;
    2. is parsed from a string;
    3. remembers the string from which it is parsed.

    >>> Quantity("3km")
    Quantity(3km)

    Quantity can be used interchangebly with parse, with the single
    difference that the latter may return an unwrapped float in case the result
    is dimensionless, whereas Quantity wraps an empty dimension.

    >>> Quantity("km/cm").unwrap()
    (100000.0, Powers())

    The main purpose of Quantity is to serve as the base class for dynamic
    subclassing via the Dimension metaclass.
    """

    def __init__(self, s):
        if isinstance(s, Quantity):
            s = s.__str
        elif not isinstance(s, str):
            raise ValueError(f"expected Quantity or str, got {type(s).__name__}")
        super().__init__(*unwrap(parse(s)))
        self.__str = s

    def __str__(self):
        return self.__str

    def __repr__(self):
        return f"{type(self).__name__}({self.__str})"

    def __reduce__(
        self,
    ) -> tuple[type[getattr(typing, "Self", "Quantity")], tuple[str]]:
        return type(self), (self.__str,)

    def __into_ags__(self) -> str:
        return self.__str

    @classmethod
    def __from_ags__(cls, obj: str):
        return cls(obj)


class Dimension(type):
    """Create dynamic subclass of Quantity.

    >>> Length = Dimension("L")
    >>> Length("2m")
    Quantity[L](2m)

    Unlike the Quantity base class, the derived class knows its expected
    dimension and rejects any non matching argument.

    >>> Length("5kg")
    Traceback (most recent call last):
       ...
    nutils.units.error.DimensionError: cannot parse 5kg as L

    Furthermore, dimensions support the multiplication, division, and power
    operations to form derived dimensions.

    >>> Time = Dimension("T")
    >>> Velocity = Length / Time
    >>> Velocity("km/h")
    Quantity[L/T](km/h)
    """

    __cache = {}

    def __new__(cls, dim):
        dim = Powers(dim)
        self = cls.__cache.get(dim)
        if self is None:
            self = type.__new__(cls, f"Quantity[{dim}]", (Quantity,), {})
            self.__dim = dim
            cls.__cache[dim] = self
        return self

    def __call__(self, s):
        quantity = super().__call__(s)
        val, dim = quantity.unwrap()
        if dim != self.__dim:
            raise DimensionError(f"cannot parse {s} as {quantity.__dim}")
        return quantity

    def __mul__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(self.__dim + other.__dim)

    def __truediv__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(self.__dim - other.__dim)

    def __pow__(self, other):
        return Dimension(self.__dim * other)

    @property
    def zero(self):
        return UMonomial(0.0, self.__dim)


def __getattr__(attr):
    if attr.startswith("Quantity[") and attr.endswith("]"):
        # required for pickle
        return Dimension(attr[9:-1])
    raise AttributeError(attr)


Length = Dimension("L")
Time = Dimension("T")
Length = Dimension("L")
Mass = Dimension("M")
ElectricCurrent = Dimension("I")
Temperature = Dimension("Θ")
AmountOfSubstance = Dimension("N")
LuminousFlux = LuminousIntensity = Dimension("J")

Area = Length**2
Volume = Length**3
WaveNumber = Vergence = Length**-1
Velocity = Speed = Length / Time
Acceleration = Velocity / Time
Force = Weight = Mass * Acceleration
Pressure = Stress = Force / Area
Tension = Force / Length
Energy = Work = Heat = Force * Length
Power = Energy / Time
Density = Mass / Volume
SpecificVolume = MassConcentration = Density**-1
SurfaceDensity = Mass / Area
Viscosity = Pressure * Time
Frequency = Radioactivity = Time**-1
CurrentDensity = ElectricCurrent / Area
MagneticFieldStrength = ElectricCurrent / Length
Charge = ElectricCurrent * Time
ElectricPotential = Power / ElectricCurrent
Capacitance = Charge / ElectricPotential
Resistance = Impedance = Reactance = ElectricPotential / ElectricCurrent
Conductance = Resistance**-1
MagneticFlux = ElectricPotential * Time
MagneticFluxDensity = MagneticFlux / Area
Inductance = MagneticFlux / ElectricCurrent
Llluminance = LuminousFlux / Area
AbsorbedDose = EquivalentDose = Energy / Mass
Concentration = AmountOfSubstance / Volume
CatalyticActivity = AmountOfSubstance / Time
