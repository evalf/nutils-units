from typing import Any
from functools import reduce
import operator

from ._dispatch import DispatchTable
from ._powers import Powers
from .error import DimensionError


class Monomial:
    """Assign powers to any object.

    This class assigns any Python object a set of powers. In the typical use
    case, the powers represent physical dimensions, for instance "length" to
    the first power for distance or to the second power for area, or "length"
    to +1 and "time" to -1 for velocity.

    The wrapper is opaque, in the sense that the wrapped value cannot be
    trivially retrieved (even though an `.unwrap` method exists). It does,
    however, support a range of numerical operation, including Numpy and Nutils
    operations via their respective dispatch methods. When any of these
    operations yield a dimensionless result, the wrapper automatically falls
    away.

    The following demonstrates the principle. We first introduce our units m
    (meter) with dimension "length", and s (seconds) with dimension "time", to
    which we arbitrarily assign the numerical values 9 and 2.

    >>> m = Monomial(9., "length")
    >>> s = Monomial(2., "time")

    Any subsequent operatons keep track of the asociated dimensions, as seen
    below in the representation of km/h.

    >>> km = m * 1000
    >>> h = s * 3600
    >>> km / h
    1.25[length/time]

    If an operation yields a dimensionless result, then the wrapper falls away
    and the underlying value is revealed.

    >>> (km / h) / (m / s)
    0.2777777777777778

    The crucial observation is that this final result is independent of the
    numerical values we initially assigned to m and s. That is, while we need
    to choose _something_ to conduct the numerical operations (typically we
    choose 1.), this choice will never affect the result of operations that
    cause the dimension wrapper to fall away naturally.
    """

    def __init__(self, value: Any, dimension: Powers):
        self.__val = value
        self.__dim = Powers(dimension)

    def unwrap(self):
        return self.__val, self.__dim

    def __bool__(self):
        return bool(self.__val)

    def __len__(self):
        return len(self.__val)

    def __iter__(self):
        cls = type(self)
        return (cls(v, self.__dim) for v in self.__val)

    def __repr__(self):
        return f"{self.__val!r}[{self.__dim}]"

    def __str__(self):
        return f"{self.__val}[{self.__dim}]"

    def __hash__(self):
        return hash((self.__val, self.__dim))

    ## POPULATE DISPATCH TABLE

    __table = DispatchTable()

    @__table.register("nutils.function.derivative")
    @__table.register("nutils.function.factor")
    @__table.register("nutils.function.jump")
    @__table.register("nutils.function.kronecker")
    @__table.register("nutils.function.linearize")
    @__table.register("nutils.function.swap_spaces")
    @__table.register("nutils.function.opposite")
    @__table.register("nutils.function.replace_arguments")
    @__table.register("nutils.function.scatter")
    @__table.register("numpy.absolute")
    @__table.register("numpy.amax")
    @__table.register("numpy.amin")
    @__table.register("numpy.broadcast_to")
    @__table.register("numpy.conjugate")
    @__table.register("numpy.imag")
    @__table.register("numpy.linalg.norm")
    @__table.register("numpy.max")
    @__table.register("numpy.mean")
    @__table.register("numpy.min")
    @__table.register("numpy.negative")
    @__table.register("numpy.positive")
    @__table.register("numpy.ptp")
    @__table.register("numpy.real")
    @__table.register("numpy.repeat")
    @__table.register("numpy.reshape")
    @__table.register("numpy.ravel")
    @__table.register("numpy.sum")
    @__table.register("numpy.take")
    @__table.register("numpy.trace")
    @__table.register("numpy.transpose")
    @__table.register("_operator.abs")
    @__table.register("_operator.getitem")
    @__table.register("_operator.neg")
    @__table.register("_operator.pos")
    def __unary(op, *args, **kwargs):
        cls = _get_monomial_class(args[0])
        arg0, dim0 = unwrap(args[0])
        val = op(arg0, *args[1:], **kwargs)
        return wrap(cls, val, dim0)

    @__table.register("numpy.add")
    @__table.register("numpy.hypot")
    @__table.register("numpy.maximum")
    @__table.register("numpy.minimum")
    @__table.register("numpy.subtract")
    @__table.register("_operator.add")
    @__table.register("_operator.mod")
    @__table.register("_operator.sub")
    def __add_like(op, *args, **kwargs):
        cls = _get_monomial_class(args[0], args[1])
        arg0, dim0 = unwrap(args[0])
        arg1, dim1 = unwrap(args[1])
        if dim0 != dim1:
            raise DimensionError(
                f"incompatible dimensions for {op.__name__}: {dim0}, {dim1}"
            )
        val = op(arg0, arg1, *args[2:], **kwargs)
        return wrap(cls, val, dim0)

    @__table.register("numpy.matmul")
    @__table.register("numpy.multiply")
    @__table.register("_operator.matmul")
    @__table.register("_operator.mul")
    def __mul_like(op, *args, **kwargs):
        cls = _get_monomial_class(args[0], args[1])
        arg0, dim0 = unwrap(args[0])
        arg1, dim1 = unwrap(args[1])
        val = op(arg0, arg1, *args[2:], **kwargs)
        return wrap(cls, val, dim0 + dim1)

    @__table.register("nutils.function.curl")
    @__table.register("nutils.function.div")
    @__table.register("nutils.function.grad")
    @__table.register("nutils.function.surfgrad")
    @__table.register("numpy.divide")
    @__table.register("_operator.truediv")
    def __div_like(op, *args, **kwargs):
        cls = _get_monomial_class(args[0], args[1])
        arg0, dim0 = unwrap(args[0])
        arg1, dim1 = unwrap(args[1])
        val = op(arg0, arg1, *args[2:], **kwargs)
        return wrap(cls, val, dim0 - dim1)

    @__table.register("nutils.function.laplace")
    def __laplace(op, *args, **kwargs):
        cls = _get_monomial_class(args[0], args[1])
        arg0, dim0 = unwrap(args[0])
        arg1, dim1 = unwrap(args[1])
        val = op(arg0, arg1, *args[2:], **kwargs)
        return wrap(cls, val, dim0 - dim1 * 2)

    @__table.register("numpy.sqrt")
    def __sqrt(op, *args, **kwargs):
        cls = _get_monomial_class(args[0])
        arg0, dim0 = unwrap(args[0])
        val = op(arg0, *args[1:], **kwargs)
        return wrap(cls, val, dim0 / 2)

    @__table.register("_operator.setitem")
    def __setitem(op, *args, **kwargs):
        cls = _get_monomial_class(args[0], args[2])
        arg0, dim0 = unwrap(args[0])
        arg2, dim2 = unwrap(args[2])
        if dim0 != dim2:
            raise DimensionError(f"cannot assign {dim2} to {dim0}")
        val = op(arg0, args[1], arg2, *args[3:], **kwargs)
        return wrap(cls, val, dim0)

    @__table.register("nutils.function.jacobian")
    @__table.register("numpy.power")
    @__table.register("_operator.pow")
    def __pow_like(op, *args, **kwargs):
        cls = _get_monomial_class(args[0])
        arg0, dim0 = unwrap(args[0])
        val = op(arg0, *args[1:], **kwargs)
        return wrap(cls, val, dim0 * args[1])

    @__table.register("nutils.function.normal")
    @__table.register("nutils.function.normalized")
    @__table.register("numpy.isfinite")
    @__table.register("numpy.isnan")
    @__table.register("numpy.ndim")
    @__table.register("numpy.shape")
    @__table.register("numpy.size")
    @__table.register("numpy.argsort")
    def __unary_op(op, *args, **kwargs):
        arg0, _ = args[0].unwrap()
        return op(arg0, *args[1:], **kwargs)

    @__table.register("numpy.equal")
    @__table.register("numpy.greater")
    @__table.register("numpy.greater_equal")
    @__table.register("numpy.less")
    @__table.register("numpy.less_equal")
    @__table.register("numpy.not_equal")
    @__table.register("_operator.eq")
    @__table.register("_operator.ge")
    @__table.register("_operator.gt")
    @__table.register("_operator.le")
    @__table.register("_operator.lt")
    @__table.register("_operator.ne")
    def __binary_op(op, *args, **kwargs):
        arg0, dim0 = unwrap(args[0])
        arg1, dim1 = unwrap(args[1])
        if dim0 != dim1 and not (isinstance(args[1], (int, float)) and args[1] == 0):
            raise DimensionError(
                f"incompatible dimensions for {op.__name__}: {dim0}, {dim1}"
            )
        return op(arg0, arg1, *args[2:], **kwargs)

    @__table.register("numpy.stack")
    @__table.register("numpy.concatenate")
    def __stack_like(op, *args, **kwargs):
        cls = _get_monomial_class(*args[0])
        arg0, dims = zip(*map(unwrap, args[0]))
        dim = dims[0]
        if any(d != dim for d in dims[1:]):
            raise DimensionError(
                f"incompatible dimensions for {op.__name__}: "
                + ", ".join(map(str, dims))
            )
        val = op(arg0, *args[1:], **kwargs)
        return wrap(cls, val, dim)

    @__table.register("nutils.function.curvature")
    def __curvature(op, *args, **kwargs):
        cls = _get_monomial_class(*args[0])
        arg0, dim0 = unwrap(args[0])
        val = op(*args, **kwargs)
        return wrap(cls, val, -dim0)

    @__table.register("nutils.function.evaluate")
    def __evaluate(op, *args, **kwargs):
        cls = _get_monomial_class(*args)
        args, dims = zip(*map(unwrap, args))
        vals = op(*args, **kwargs)
        return tuple(wrap(cls, val, dim) for val, dim in zip(vals, dims))

    @__table.register("nutils.function.field")
    def __field(op, *args, **kwargs):
        cls = _get_monomial_class(*args)
        args, dims = zip(*map(unwrap, args))
        val = op(*args, **kwargs)
        dim = reduce(operator.add, dims)
        # we abuse the fact that unpack str returns dimensionless
        return wrap(cls, val, dim)

    @__table.register("nutils.function.arguments_for")
    def __attribute(op, *args, **kwargs):
        args = [unwrap(arg)[0] for arg in args]
        return op(*args, **kwargs)

    @__table.register("numpy.interp")
    def __interp(op, x, xp, fp, *args, **kwargs):
        cls = _get_monomial_class(x, xp, fp)
        x, dimx = unwrap(x)
        xp, dimxp = unwrap(xp)
        fp, dimfp = unwrap(fp)
        if dimx != dimxp:
            raise DimensionError(
                f"incompatible dimensions for {op.__name__}: {dimx}, {dimxp}"
            )
        val = op(x, xp, fp, *args, **kwargs)
        return wrap(cls, val, dimfp)

    @__table.register("nutils.topology.Topology.locate")
    def __locate(
        op,
        topo,
        geom,
        coords,
        *,
        tol=0,
        maxdist=None,
        **kwargs,
    ):
        geom, dimgeom = unwrap(geom)
        coords, dimcoords = unwrap(coords)
        if dimgeom != dimcoords:
            raise DimensionError(
                f"incompatible dimensions for locate: {dimgeom}, {dimcoords}"
            )
        if tol != 0:
            tol, dimtol = unwrap(tol)
            if dimtol != dimgeom:
                raise DimensionError(
                    f"invalid dimension for tol: got {dimtol}, expected {dimgeom}"
                )
        if maxdist is not None:
            maxdist, dimmaxdist = unwrap(maxdist)
            if dimmaxdist != dimgeom:
                raise DimensionError(
                    f"invalid dimension for maxdist: got {dimmaxdist}, expected {dimgeom}"
                )
        return op(
            topo,
            geom,
            coords,
            tol=tol,
            maxdist=maxdist,
            **kwargs,
        )

    @__table.register("nutils.sample.Sample.bind")
    @__table.register("nutils.sample.Sample.integral")
    def __sample(op, sample, func):
        cls = _get_monomial_class(func)
        func, dim = unwrap(func)
        val = op(sample, func)
        return wrap(cls, val, dim)

    @__table.register("numpy.linalg.det")
    def __det(op, arg):
        cls = _get_monomial_class(arg)
        arg, dim = unwrap(arg)
        val = op(arg)
        dim = dim * arg.ndim
        return wrap(cls, val, dim)

    @__table.register("numpy.linspace")
    def __linspace(
        op, start, stop, num=50, endpoint=True, retstep=False, *args, **kwargs
    ):
        cls = _get_monomial_class(start, stop)
        start, dim = unwrap(start)
        stop, dim_ = unwrap(stop)
        if dim != dim_:
            raise DimensionError(f"incompatible dimensions for linspace: {dim}, {dim_}")
        samples = op(start, stop, num, endpoint, retstep, *args, **kwargs)
        if retstep:
            samples, step = samples
            return wrap(cls, samples, dim), wrap(cls, step, dim)
        else:
            return wrap(cls, samples, dim)

    @__table.register("numpy.meshgrid")
    def __meshgrid(op, *xi, **kwargs):
        cls = _get_monomial_class(*xi)
        xi, dims = zip(*map(unwrap, xi))
        arrays = op(*xi, **kwargs)
        return tuple(wrap(cls, array, dim) for array, dim in zip(arrays, dims))

    ## DEFINE OPERATORS

    __getitem__ = __table.bind_method(operator.getitem)
    __setitem__ = __table.bind_method(operator.setitem)
    __neg__ = __table.bind_method(operator.neg)
    __pos__ = __table.bind_method(operator.pos)
    __abs__ = __table.bind_method(operator.abs)
    __lt__ = __table.bind_method(operator.lt)
    __le__ = __table.bind_method(operator.le)
    __eq__ = __table.bind_method(operator.eq)
    __ne__ = __table.bind_method(operator.ne)
    __gt__ = __table.bind_method(operator.gt)
    __ge__ = __table.bind_method(operator.ge)
    __add__, __radd__ = __table.bind_method_and_reverse(operator.add)
    __sub__, __rsub__ = __table.bind_method_and_reverse(operator.sub)
    __mul__, __rmul__ = __table.bind_method_and_reverse(operator.mul)
    __matmul__, __rmatmul__ = __table.bind_method_and_reverse(operator.matmul)
    __truediv__, __rtruediv__ = __table.bind_method_and_reverse(operator.truediv)
    __mod__, __rmod__ = __table.bind_method_and_reverse(operator.mod)
    __pow__, __rpow__ = __table.bind_method_and_reverse(operator.pow)

    ## DISPATCH THIRD PARTY CALLS

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        return self.__table.call_or_notimp(ufunc, inputs, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return self.__table.call_or_notimp(func, args, kwargs)

    __nutils_dispatch__ = __table.call_or_notimp


def wrap(cls, obj, dim):
    return cls(obj, dim) if dim else obj


def unwrap(obj):
    return obj.unwrap() if isinstance(obj, Monomial) else (obj, Powers({}))


def _get_monomial_class(*args):
    """Return common Monomial base class.

    This helper function returns the highest subclass of which all monomial
    arguments are an instance. Concretely, if one argument is a Monomial and
    the other a UMonomial, then Monomial is returned. If all are UMonomial then
    the return value is UMonomial."""

    types = {type(arg) for arg in args if isinstance(arg, Monomial)}
    bases = _collect_bases(types.pop())
    for cls in types:
        bases_ = _collect_bases(cls)
        while bases != bases_[: len(bases)]:
            bases = bases[:-1]
    assert bases[0] is Monomial
    while bases[-1].__init__ != Monomial.__init__:
        bases = bases[:-1]
    return bases[-1]


def _collect_bases(cls, sentinel=object):
    if cls is sentinel:
        return ()
    assert issubclass(cls, sentinel)
    (base,) = cls.__bases__
    return *_collect_bases(base, sentinel), cls
