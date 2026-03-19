# Nutils Units

The nutils.units module provides a way to track dimensions of numerical objects
and their change as the objects pass through functions. Its purposes are
twofold:

1. Numerical type checking, safeguarding against such mistakes as adding two
   objects of different dimensions together.
2. Unit consistency, ensuring that different metric systems can coexist without
   crashing into Mars.

Units offers three API levels: core, metric and typing, with each building on
top of the former.

## Core

The core API offers the `Monomial` class, which wraps an object to assign it
physical dimensions. A dimension is identified by a string, such as "L" for
length. By wrapping the unit float we define the reference length for our
calculations to be one meter.

```python
>>> from nutils.units.core import Monomial
>>> meter = Monomial(1., "L")

```

Once the initial unit is seeded, we can build on it to define derived units:

```python
>>> inch = meter * 0.0254

```

This demonstrates that `Monomial` objects support numerical manipulation such
as multiplication. These manipulations extend to (supported) external packages
such as numpy:

```python
>>> import numpy as np
>>> v1 = np.array([1, -2]) * meter
>>> v2 = np.array([3, 1]) * inch
>>> array = np.stack([v1, v2])
>>> np.linalg.det(array)
np.float64(0.17779999999999999)[L2]

```

The [L2] in the string representation indicates that the result has dimension
length squared, i.e. area. The value before it is the representation of the
wrapped object, but its value is shielded by the wrapper. The wrapper falls
away when an operation yields a dimensionless result, for instance by dividing
out a unit:

```python
>>> array / meter
array([[ 1.    , -2.    ],
       [ 0.0762,  0.0254]])

```

Crucially, dimensionless results do not depend on the definition of our
reference lengths, as any scaling cancels out by definition. We could have
defined our meter to be `Monomial(np.pi, "L")` instead and still obtained the
same result up to rounding errors. The numerical value of the reference length
is merely a conduit for the internal manipulations.

## Metric

The metric API adds support for units via the `parse` function. This returns a
`UMonomial` object that has access to a predefined set of units, using the SI
base units as internal reference measures.

```python
>>> from nutils.units.metric import parse
>>> length = parse('2cm')
>>> width = parse('3.5in')
>>> force = parse('5N')

```

We can then manipulate the `UMonomial` objects as before.

```
>>> area = length * width
>>> pressure = force / area
>>> pressure / 'kPa'
2.8121484814398205

```

Note that in dividing out the unit we omitted `parse`, which is a convenience
shorthand for this precise situation. For added convenience, the `UMonomial`
class also supports direct string formatting of the wrapped value.

```python
>>> f'pressure: {pressure:.1kPa}'
'pressure: 2.8kPa'

```

The units registry is an append-only state machine that is part of the metric
module. Additional units can be added if necessary via the `units.register`
function.

```python
>>> from nutils.units.metric import units
>>> units.register("lbf", parse("4.448222N"), prefix="")
>>> units.register("psi", parse("lbf/in2"), prefix="")
>>> f'pressure: {pressure:.1psi}'
'pressure: 0.4psi'

```

## Typing

The typing API adds specific types for scalar quantities.

```python
>>> from nutils.units.typing import Length, Time, Velocity
>>> Velocity('.4km/h')
Quantity[L/T](.4km/h)

```

The `Quantity` types function similarly to `parse`, with two differences: 1.
the object reduces back to its original string argument, with potential uses
for object introspection, and 2. it protects against using wrong units.

```python
>>> Velocity('.4km/g')
Traceback (most recent call last):
    ...
nutils.units.error.DimensionError: cannot parse .4km/g as L/T

```

Derived quantities can be formed by operating directly on the types:

```python
>>> Velocity == Length / Time
True

```

The quantity types can be used as function annotations for general readability
a for the potential aid external introspection tools.

```python
>>> def distance(velocity: Velocity, time: Time):
...     return velocity * time

```

Note that the return value of this function is not a `Length` but a general
`UMonomial`, as the result of a numerical operation does not have an inherent
unit.
