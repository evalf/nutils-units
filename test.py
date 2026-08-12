from nutils.units.error import DimensionError
from nutils.units.metric import units, parse
from nutils.units.typing import Length, Time, Velocity, Force, Area

from importlib import import_module
from unittest import TestCase
from doctest import DocTestSuite, DocFileSuite

import numpy
import pickle
import ags.ucsl


def load_tests(loader, tests, ignore):
    for m in "_dispatch", "_powers", "_units", "core", "error", "metric", "typing":
        tests.addTests(DocTestSuite(import_module("nutils.units." + m)))
    tests.addTests(DocFileSuite("README.md"))
    return tests


class UMonomial(TestCase):
    def test_fromstring(self):
        F = parse("5kN")
        self.assertEqual(F / "N", 5000)
        v = parse("-36km/h")
        self.assertEqual(v / "m/s", -10)
        v = parse("0.4m/cm")
        self.assertEqual(v, 40)

    def test_fromvalue(self):
        F = parse("10N")
        self.assertEqual(F / parse("2N"), 5)

    def test_getitem(self):
        F = units.N * numpy.arange(6).reshape(2, 3)
        self.assertEqual(F[0, 0], parse("0N"))
        self.assertEqual(F[0, 1], parse("1N"))
        self.assertEqual(F[0, 2], parse("2N"))
        self.assertEqual(F[1, 0], parse("3N"))
        self.assertEqual(F[1, 1], parse("4N"))
        self.assertEqual(F[1, 2], parse("5N"))

    def test_setitem(self):
        F = units.N * numpy.zeros(3)
        F[0] = parse("1N")
        F[1] = parse("2N")
        with self.assertRaisesRegex(DimensionError, r"cannot assign L2 to M\*L/T2"):
            F[2] = parse("10m2")
        F[2] = parse("3N")
        self.assertTrue(numpy.all(F == units.N * numpy.array([1, 2, 3])))

    def test_iter(self):
        F = units.N * numpy.arange(6).reshape(2, 3)
        for i, Fi in enumerate(F):
            for j, Fij in enumerate(Fi):
                self.assertEqual(Fij, units.N * (i * 3 + j))

    def test_multiply(self):
        self.assertEqual(parse("2kg") * parse("10m/s2"), parse("20N"))
        self.assertEqual(2 * parse("10m/s2"), parse("20m/s2"))
        self.assertEqual(parse("2kg") * 10, parse("20kg"))
        self.assertEqual(parse("2s") * parse("10/s"), 20)
        self.assertEqual(numpy.multiply(parse("2kg"), parse("10m/s2")), parse("20N"))

    def test_matmul(self):
        self.assertEqual(
            (units.kg * numpy.array([2, 3])) @ (parse("m/s2") * numpy.array([5, -3])),
            parse("1N"),
        )

    def test_divide(self):
        self.assertEqual(parse("2m") / parse("10s"), parse(".2m/s"))
        self.assertEqual(2 / parse("10s"), parse(".2/s"))
        self.assertEqual(parse("2m") / 10, parse(".2m"))
        self.assertEqual(parse("2kg/m3") / parse("10kg/m3"), 0.2)
        self.assertEqual(numpy.divide(parse("2m"), parse("10s")), parse(".2m/s"))

    def test_power(self):
        self.assertEqual(parse("3m") ** 2, parse("9m2"))
        self.assertEqual(parse("3m") ** 0, 1)
        self.assertEqual(numpy.power(parse("3m"), 2), parse("9m2"))

    def test_add(self):
        self.assertEqual(parse("2kg") + parse("3kg"), parse("5kg"))
        self.assertEqual(numpy.add(parse("2kg"), parse("3kg")), parse("5kg"))
        with self.assertRaises(DimensionError):
            parse("2kg") + parse("3m")

    def test_sub(self):
        self.assertEqual(parse("2kg") - parse("3kg"), parse("-1kg"))
        self.assertEqual(numpy.subtract(parse("2kg"), parse("3kg")), parse("-1kg"))
        with self.assertRaises(DimensionError):
            parse("2kg") - parse("3m")

    def test_hypot(self):
        self.assertEqual(numpy.hypot(parse("3kg"), parse("4kg")), parse("5kg"))
        with self.assertRaisesRegex(
            DimensionError, r"incompatible dimensions for hypot: M, L"
        ):
            numpy.hypot(parse("3kg"), parse("4m"))

    def test_neg(self):
        self.assertEqual(-parse("2kg"), parse("-2kg"))
        self.assertEqual(numpy.negative(parse("2kg")), parse("-2kg"))

    def test_pos(self):
        self.assertEqual(+parse("2kg"), parse("2kg"))
        self.assertEqual(numpy.positive(parse("2kg")), parse("2kg"))

    def test_abs(self):
        self.assertEqual(numpy.abs(parse("-2kg")), parse("2kg"))

    def test_real(self):
        self.assertEqual(
            numpy.real(parse("1V") + 1j * parse("2V")),
            parse("1V"),
        )

    def test_imag(self):
        self.assertEqual(
            numpy.imag(parse("1V") + 1j * parse("2V")),
            parse("2V"),
        )

    def test_conjugate(self):
        self.assertEqual(
            numpy.conjugate(parse("1V") + 1j * parse("2V")),
            parse("1V") - 1j * parse("2V"),
        )

    def test_sqrt(self):
        self.assertEqual(numpy.sqrt(parse("4m2")), parse("2m"))

    def test_sum(self):
        self.assertTrue(
            numpy.all(
                numpy.sum(units.kg * numpy.arange(6).reshape(2, 3), 0)
                == units.kg * numpy.array([3, 5, 7])
            )
        )
        self.assertTrue(
            numpy.all(
                numpy.sum(units.kg * numpy.arange(6).reshape(2, 3), 1)
                == units.kg * numpy.array([3, 12])
            )
        )

    def test_mean(self):
        self.assertTrue(
            numpy.all(
                numpy.mean(units.kg * numpy.arange(6).reshape(2, 3), 0)
                == units.kg * numpy.array([1.5, 2.5, 3.5])
            )
        )
        self.assertTrue(
            numpy.all(
                numpy.mean(units.kg * numpy.arange(6).reshape(2, 3), 1)
                == units.kg * numpy.array([1, 4])
            )
        )

    def test_broadcast_to(self):
        v = numpy.array([1, 2, 3])
        A = units.kg * v
        B = numpy.broadcast_to(A, (2, 3))
        val, dim = B.unwrap()
        self.assertEqual(val.shape, (2, 3))
        self.assertEqual(B[1, 1], parse("2kg"))

    def test_trace(self):
        A = units.kg * numpy.arange(18).reshape(3, 2, 3)
        self.assertTrue(
            numpy.all(
                numpy.trace(A, axis1=0, axis2=2) == units.kg * numpy.array([21, 30])
            )
        )

    def test_ptp(self):
        A = units.kg * numpy.array([2, -10, 5, 0])
        self.assertEqual(numpy.ptp(A), parse("15kg"))

    def test_min(self):
        A = units.kg * numpy.array([2, -10, 5, 0])
        self.assertEqual(numpy.max(A), parse("5kg"))

    def test_max(self):
        A = units.kg * numpy.array([2, -10, 5, 0])
        self.assertEqual(numpy.min(A), parse("-10kg"))

    def test_cmp(self):
        A = parse("2kg")
        B = parse("3kg")
        self.assertTrue(A < B)
        self.assertTrue(numpy.less(A, B))
        self.assertTrue(A <= B)
        self.assertTrue(numpy.less_equal(A, B))
        self.assertFalse(A > B)
        self.assertFalse(numpy.greater(A, B))
        self.assertFalse(A >= B)
        self.assertFalse(numpy.greater_equal(A, B))
        self.assertFalse(A == B)
        self.assertFalse(numpy.equal(A, B))
        self.assertTrue(A != B)
        self.assertTrue(numpy.not_equal(A, B))

    def test_cmp_zero(self):
        A = parse("2kg")
        B = parse("-3m/s")
        self.assertFalse(A < 0)
        self.assertFalse(A <= 0)
        self.assertTrue(A > 0)
        self.assertTrue(A >= 0)
        self.assertFalse(A == 0)
        self.assertTrue(A != 0)
        self.assertTrue(B < 0)
        self.assertTrue(B <= 0)
        self.assertFalse(B > 0)
        self.assertFalse(B >= 0)
        self.assertFalse(B == 0)
        self.assertTrue(B != 0)

    def test_shape(self):
        A = parse("2kg")
        self.assertEqual(numpy.shape(A), ())
        A = units.kg * numpy.arange(3)
        self.assertEqual(numpy.shape(A), (3,))
        val, dim = A.unwrap()
        self.assertEqual(val.shape, (3,))

    def test_ndim(self):
        A = parse("2kg")
        self.assertEqual(numpy.ndim(A), 0)
        A = units.kg * numpy.arange(3)
        self.assertEqual(numpy.ndim(A), 1)
        val, dim = A.unwrap()
        self.assertEqual(val.ndim, 1)

    def test_size(self):
        A = parse("2kg")
        self.assertEqual(numpy.size(A), 1)
        A = units.kg * numpy.arange(3)
        self.assertEqual(numpy.size(A), 3)
        val, dim = A.unwrap()
        self.assertEqual(val.size, 3)

    def test_isnan(self):
        self.assertTrue(numpy.isnan(units.kg * float("nan")))
        self.assertFalse(numpy.isnan(parse("2kg")))

    def test_isfinite(self):
        self.assertFalse(numpy.isfinite(units.kg * float("nan")))
        self.assertFalse(numpy.isfinite(units.kg * float("inf")))
        self.assertTrue(numpy.isfinite(parse("2kg")))

    def test_stack(self):
        A = parse("2kg")
        B = parse("3kg")
        C = parse("4kg")
        D = parse("5s")
        self.assertTrue(
            numpy.all(numpy.stack([A, B, C]) == units.kg * numpy.array([2, 3, 4]))
        )
        with self.assertRaisesRegex(
            DimensionError,
            r"incompatible dimensions for stack: M, M, M, T",
        ):
            numpy.stack([A, B, C, D])

    def test_concatenate(self):
        A = units.kg * numpy.array([1, 2])
        B = units.kg * numpy.array([3, 4])
        C = units.s * numpy.array([5, 6])
        self.assertTrue(
            numpy.all(numpy.concatenate([A, B]) == units.kg * numpy.array([1, 2, 3, 4]))
        )
        with self.assertRaisesRegex(
            DimensionError,
            r"incompatible dimensions for concatenate: M, M, T",
        ):
            numpy.concatenate([A, B, C])

    def test_format(self):
        s = "velocity: {:.1m/s}".format(parse("9km/h"))
        self.assertEqual(s, "velocity: 2.5m/s")

    def test_pickle(self):
        v = parse("2m/s")
        s = pickle.dumps(v)
        self.assertEqual(pickle.loads(s), v)

    def test_string_representation(self):
        F = numpy.array([1.0, 2.0]) * units.N
        self.assertEqual(str(F), "[1. 2.][M*L/T2]")
        self.assertEqual(repr(F), "array([1., 2.])[M*L/T2]")

    def test_hash(self):
        v1 = parse("2m/s")
        v2 = parse("2m/s")
        self.assertEqual(hash(v1), hash(v2))

    def test_reshape(self):
        F32 = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * units.N
        F23 = numpy.reshape(F32, (2, 3))
        self.assertEqual(numpy.shape(F23), (2, 3))
        self.assertEqual(F23[0, 0], parse("1N"))
        self.assertEqual(F23[0, 1], parse("2N"))
        self.assertEqual(F23[0, 2], parse("3N"))
        self.assertEqual(F23[1, 0], parse("4N"))
        self.assertEqual(F23[1, 1], parse("5N"))
        self.assertEqual(F23[1, 2], parse("6N"))

    def test_ravel(self):
        F32 = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * units.N
        F6 = numpy.ravel(F32)
        self.assertEqual(numpy.shape(F6), (6,))
        self.assertEqual(F6[0], parse("1N"))
        self.assertEqual(F6[1], parse("2N"))
        self.assertEqual(F6[2], parse("3N"))
        self.assertEqual(F6[3], parse("4N"))
        self.assertEqual(F6[4], parse("5N"))
        self.assertEqual(F6[5], parse("6N"))

    def test_norm(self):
        F = numpy.array([3.0, 4.0]) * units.N
        Fnorm = numpy.linalg.norm(F, axis=0)
        self.assertEqual(Fnorm, parse("5N"))

    def test_interp(self):
        xp = numpy.array([0.0, 1.0, 3.0]) * units.m
        fp = numpy.array([10.0, 12.0, 12.0]) * units.N
        x = numpy.array([0.5, 1.5]) * units.m
        f = numpy.interp(x, xp, fp)
        self.assertEqual(numpy.shape(f), (2,))
        self.assertEqual(f[0], parse("11N"))
        self.assertEqual(f[1], parse("12N"))

    def test_linspace(self):
        start = 2.0 * units.m
        stop = 5.0 * units.km
        num = 9
        samples = numpy.linspace(start, stop, num)
        self.assertEqual(len(samples), num)
        self.assertEqual(samples[0], start)
        self.assertEqual(samples[num // 2], 0.5 * start + 0.5 * stop)
        self.assertEqual(samples[-1], stop)
        samples_, step = numpy.linspace(start, stop, num, retstep=True)
        self.assertTrue((samples == samples_).all())
        self.assertEqual(step, (stop - start) / (num - 1))


class Dimension(TestCase):
    def test_multiply(self):
        self.assertEqual(Velocity * Time, Length)

    def test_divide(self):
        self.assertEqual(Length / Time, Velocity)

    def test_power(self):
        self.assertEqual(Length**2, Area)
        self.assertEqual(Area**0.5, Length)

    def test_name(self):
        self.assertEqual(Force.__name__, "Quantity[M*L/T2]")
        self.assertEqual((Force**0.5).__name__, "Quantity[M1/2*L1/2/T]")
        self.assertEqual((Force**1.5).__name__, "Quantity[M3/2*L3/2/T3]")

    def test_pickle(self):
        T = Length / Time
        s = pickle.dumps(T)
        self.assertEqual(pickle.loads(s), T)
        v = T("2m/s")
        s = pickle.dumps(v)
        self.assertEqual(pickle.loads(s), v)
        self.assertIsInstance(T, type)


class Quantity(TestCase):
    def test_good(self):
        v = Velocity("2m/s")
        self.assertEqual(v, parse("2m/s"))

    def test_bad(self):
        with self.assertRaisesRegex(DimensionError, "cannot parse 2kg as L/T"):
            Velocity("2kg")

    def test_pickle(self):
        v = Velocity("2m/s")
        s = pickle.dumps(v)
        v_ = pickle.loads(s)
        self.assertEqual(v, v_)

    def test_ags(self):
        v = Velocity("2m/s")
        s = ags.ucsl.dumps(v, Velocity)
        self.assertEqual(s, "2m/s")
        v_ = ags.ucsl.loads(s, Velocity)
        self.assertEqual(v, v_)
