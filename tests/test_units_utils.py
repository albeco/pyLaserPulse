""" tests for units_utils.py"""

import unittest
import pyLaserPulse.units_utils as ut
from astropy import units as u
from astropy.units.core import CompositeUnit
from astropy.units.quantity import Quantity
import random as rnd


# monkey-patching TestCase to support array quantities
def assertQuantityEqual(self, q1, q2):
    self.assertEqual(len(q1), len(q2))
    for x, y in zip(q1, q2):
        self.assertEqual(x, y)


unittest.TestCase.assertQuantityEqual = assertQuantityEqual


def get_random_unit(test_bases=(u.m, u.s, u.kg, u.A)):
    scale = rnd.randrange(-1e5, 1e5)
    start_no_bases = rnd.randint(2, len(test_bases))
    # choose bases randomly, then remove duplicates
    bases = sorted(set(rnd.choices(test_bases, k=start_no_bases)),
                   key=lambda x: x.to_string())
    no_bases = len(bases)
    # if there are two many duplicates try again
    if no_bases >= 2:
        powers = rnd.choices([-4, -3, -2, -1, 1, 2, 3, 4], k=no_bases)
        unit = CompositeUnit(scale, bases, powers)
    else:
        unit = get_random_unit(test_bases)
    return unit


class TestFixHertz(unittest.TestCase):
    """ test class for the s/Hz functions in units_utils module """

    def test_is_unity(self):
        for unit in {1, 1.0}:
            self.assertTrue(ut.is_unity(unit))
        for x in {0, -1, -1, 1.001, 1e10}:
            self.assertFalse(ut.is_unity(x))

    def test_flip_prefix(self):
        flip_twice = ''.join(ut.flip_prefix(p)
                             for p in reversed(ut.si_prefixes))
        self.assertEqual(flip_twice, ut.si_prefixes)

    def test_fixhertz_item(self):
        # tests with simple units
        # positive exponents with second and hertz -> should do nothing
        for pwr in range(0, 11):
            self.assertTupleEqual(ut.fixhertz_item(u.second, pwr),
                                                  (u.second, pwr))
            self.assertTupleEqual(ut.fixhertz_item(u.hertz, pwr),
                                                  (u.hertz, pwr))
        # negative exponents with second and hertz -> should flip units
        for pwr in range(-10, 0):
            self.assertTupleEqual(ut.fixhertz_item(u.second, pwr),
                                                  (u.hertz, -pwr))
            self.assertTupleEqual(ut.fixhertz_item(u.hertz, pwr),
                                                  (u.second, -pwr))
        # base units other than second and hertz -> should do nothing
        for baseUnit in (u.meter, u.kilogram, u.ampere):
            for pwr in range(-10, 11):
                x = (baseUnit, pwr)
                self.assertTupleEqual(ut.fixhertz_item(*x), x)
        # tests with prefix units
        # positive exponents with second and hertz -> should do nothing
        self.assertTupleEqual(ut.fixhertz_item(u.ms, 0),
                                              (u.ms, 0))
        self.assertTupleEqual(ut.fixhertz_item(u.s, 0),
                                              (u.s, 0))
        self.assertTupleEqual(ut.fixhertz_item(u.fs, 3),
                                              (u.fs, 3))
        self.assertTupleEqual(ut.fixhertz_item(u.mHz, 0),
                                              (u.mHz, 0))
        self.assertTupleEqual(ut.fixhertz_item(u.fHz, 3),
                                              (u.fHz, 3))
        # negative exponents with second and hertz -> should flip units
        self.assertTupleEqual(ut.fixhertz_item(u.fs, -3),
                                              (u.PHz, 3))
        self.assertTupleEqual(ut.fixhertz_item(u.s, -3),
                                              (u.Hz, 3))
        self.assertTupleEqual(ut.fixhertz_item(u.MHz, -2),
                                              (u.us, 2))
        # base units other than second and hertz -> should do nothing
        self.assertTupleEqual(ut.fixhertz_item(u.fJ, -3),
                                              (u.fJ, -3))
        self.assertTupleEqual(ut.fixhertz_item(u.pA, 0),
                                              (u.pA, 0))
        self.assertTupleEqual(ut.fixhertz_item(u.zA, 40),
                                              (u.zA, 40))
        self.assertTupleEqual(ut.fixhertz_item(u.m, -10),
                                              (u.m, -10))
        self.assertTupleEqual(ut.fixhertz_item(u.m, 0),
                                              (u.m, 0))
        self.assertTupleEqual(ut.fixhertz_item(u.m, 13),
                                              (u.m, 13))
        self.assertTupleEqual(ut.fixhertz_item(u.YA, -11),
                                              (u.YA, -11))
        self.assertTupleEqual(ut.fixhertz_item(u.TA, 0),
                                              (u.TA, 0))
        self.assertTupleEqual(ut.fixhertz_item(u.GA, 73),
                                              (u.GA, 73))

    def test_fixhertz_comp(self):
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [], [])),
            CompositeUnit(1, [], []))
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [u.m], [2])),
            CompositeUnit(1, [u.m], [2]))
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [u.m, u.s, u.kg], [2, 3, 0])),
            CompositeUnit(1, [u.m, u.s, u.kg], [2, 3, 0]))
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [u.km, u.GHz, u.kg], [2, 3, 6])),
            CompositeUnit(1, [u.km, u.GHz, u.kg], [2, 3, 6]))
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [u.mA, u.GHz, u.kg, u.s],
                          [20, 31, 7, 10])),
            CompositeUnit(1, [u.mA, u.GHz, u.kg, u.s],
                          [20, 31, 7, 10]))
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [u.m, u.s, u.kg], [20, -3, 0])),
            CompositeUnit(1, [u.m, u.Hz, u.kg], [20, 3, 0]))
        self.assertEqual(ut.fixhertz_comp(
            CompositeUnit(1, [u.km, u.GHz, u.kg], [2, -3, 6])),
            CompositeUnit(1, [u.km, u.ns, u.kg], [2, 3, 6]))

    def test_fixhertz(self):
        # should work for composite units
        self.assertEqual(ut.fixhertz(
            CompositeUnit(1, [u.m, u.s, u.kg], [2, 3, 0])),
            CompositeUnit(1, [u.m, u.s, u.kg], [2, 3, 0]))
        self.assertEqual(ut.fixhertz(
            CompositeUnit(1, [u.km, u.GHz, u.kg], [2, 3, 6])),
            CompositeUnit(1, [u.km, u.GHz, u.kg], [2, 3, 6]))
        self.assertEqual(ut.fixhertz(
            CompositeUnit(1, [u.m, u.s, u.kg], [2, -3, 0])),
            CompositeUnit(1, [u.m, u.Hz, u.kg], [2, 3, 0]))
        self.assertEqual(ut.fixhertz(
            CompositeUnit(1, [u.km, u.GHz, u.kg], [2, -3, 6])),
            CompositeUnit(1, [u.km, u.ns, u.kg], [2, 3, 6]))
        # should do nothing for simple units and numbers
        self.assertEqual(ut.fixhertz(3), 3)
        self.assertEqual(ut.fixhertz(0), 0)
        self.assertEqual(ut.fixhertz(12), 12)
        self.assertEqual(ut.fixhertz(u.fs), u.fs)
        self.assertEqual(ut.fixhertz(u.cm), u.cm)
        # should work for simple physical quantities
        self.assertAlmostEqual(ut.fixhertz(2.3 * u.m), 2.3 * u.m)
        self.assertAlmostEqual(ut.fixhertz(2 / u.s), 2 * u.Hz)
        self.assertAlmostEqual(ut.fixhertz(2 / u.ms), 2 * u.kHz)
        self.assertAlmostEqual(ut.fixhertz(-2 / u.ms), -2 * u.kHz)
        self.assertAlmostEqual(ut.fixhertz(5 / u.GHz), 5 * u.ns)
        self.assertAlmostEqual(ut.fixhertz(-1.5 / u.GHz), -1.5 * u.ns)
        # should work with composite physical quantities
        self.assertAlmostEqual(ut.fixhertz(2 * u.kg ** 2 / u.s),
                                           2 * u.kg ** 2 * u.Hz)
        self.assertAlmostEqual(ut.fixhertz(2 * u.kg ** 2 / u.s ** 3),
                                           2 * u.kg ** 2 * u.Hz ** 3)
        self.assertAlmostEqual(ut.fixhertz(2 * u.kg ** 2 / u.Hz ** 4),
                                           2 * u.kg ** 2 * u.s ** 4)
        self.assertAlmostEqual(ut.fixhertz(3.2 * u.fs / u.fs),
                                           3.2 * u.m / u.m)
        # should also work for arrays
        self.assertQuantityEqual(ut.fixhertz(Quantity([-1.5, 1, 11], 1/u.GHz)),
                                             Quantity([-1.5, 1, 11], u.ns))
        self.assertQuantityEqual(ut.fixhertz(Quantity([2, 20, 3.2], 1)),
                                             Quantity([2, 20, 3.2], 1))


class TestOptimUnits(unittest.TestCase):
    """ test class for optimization functions in units_utils module """

    def test_opt_single_base_toHigherSIUnit_power_1(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e3, [u.m], [1])), 1 * u.km)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e3, [u.kg], [1])), 1 * u.Mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e3, [u.A], [1])), 1 * u.kA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.m], [1])), 1 * u.Mm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.kg], [1])), 1 * u.Gg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.A], [1])), 1 * u.MA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.m], [1])), 1 * u.Gm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.kg], [1])), 1 * u.Tg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.A], [1])), 1 * u.GA)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.Gg], [1])), 1 * u.Eg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.MA], [1])), 1 * u.EA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e15, [u.Ms], [1])), 1 * u.Zs)
        # starting from lower unit with small prefix to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.ug], [1])), 1 * u.g)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.pA], [1])), 1 * u.A)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e15, [u.fs], [1])), 1 * u.s)
        # starting from lower unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e3, [u.ug], [1])), 1 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.ng], [1])), 1 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.fHz], [1])), 1 * u.uHz)
        # starting from lower unit to high unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e15, [u.ug], [1])), 1 * u.Gg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.ug], [1])), 1 * u.kg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.fHz], [1])), 1 * u.kHz)

    def test_opt_single_base_inBetweenHigherSIUnit_power_1(self):
        # the prefix changes when value>=1000: 990nm instead of 0.99um
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(12e3, [u.m], [1])), 12 * u.km)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(90e3, [u.kg], [1])), 90 * u.Mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e4, [u.A], [1])), 10 * u.kA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(8e8, [u.m], [1])), 800 * u.Mm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e7, [u.kg], [1])), 20 * u.Gg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e7, [u.A], [1])), 10 * u.MA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e10, [u.m], [1])), 10 * u.Gm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(4e9, [u.kg], [1])), 4 * u.Tg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(555e9, [u.A], [1])), 555 * u.GA)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e11, [u.Gg], [1])), 100 * u.Eg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e13, [u.MA], [1])), 10 * u.EA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(9e17, [u.Ms], [1])), 900 * u.Zs)
        # starting from lower unit with small prefix to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e7, [u.ug], [1])), 10 * u.g)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e12, [u.pA], [1])), 2 * u.A)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(30e16, [u.fs], [1])), 300 * u.s)
        # starting from lower unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(12e4, [u.ug], [1])), 120 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(3e8, [u.ng], [1])), 300 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(999e9, [u.fHz], [1])), 999 * u.uHz)
        # starting from lower unit to high unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e16, [u.ug], [1])), 10 * u.Gg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e10, [u.ug], [1])), 10 * u.kg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e19, [u.fHz], [1])), 10 * u.kHz)

    def test_opt_single_base_toLowerSIUnit_power_1(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-3, [u.m], [1])), 1 * u.mm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-3, [u.kg], [1])), 1 * u.g)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-3, [u.A], [1])), 1 * u.mA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.m], [1])), 1 * u.um)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.kg], [1])), 1 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.A], [1])), 1 * u.uA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.m], [1])), 1 * u.nm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.kg], [1])), 1 * u.ug)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.A], [1])), 1 * u.nA)
        # starting from higher unit to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.Mg], [1])), 1 * u.g)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.Tg], [1])), 1 * u.kg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.EHz], [1])), 1 * u.Hz)
        # starting from higher unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.Mg], [1])), 1 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.kA], [1])), 1 * u.nA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-15, [u.Gs], [1])), 1 * u.us)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.Eg], [1])), 1 * u.Gg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.EA], [1])), 1 * u.MA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-15, [u.Zs], [1])), 1 * u.Ms)
        # starting from lower unit with small prefix to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.ug], [1])), 1 * u.pg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.uA], [1])), 1 * u.aA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-15, [u.ns], [1])), 1 * u.ys)

    def test_opt_single_base_inBetweenLowerSIUnit_power_1(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(12e-3, [u.m], [1])), 12 * u.mm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-4, [u.kg], [1])), 100 * u.mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e-5, [u.A], [1])), 20 * u.uA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-7, [u.m], [1])), 100 * u.nm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-8, [u.kg], [1])), 10 * u.ug)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(32e-6, [u.A], [1])), 32 * u.uA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.m], [1])), 100 * u.pm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-11, [u.kg], [1])), 10 * u.ng)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(3e-10, [u.A], [1])), 300 * u.pA)
        # starting from higher unit to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(11e-6, [u.Mg], [1])), 11 * u.g)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(999e-9, [u.Tg], [1])), 999 * u.kg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(155e-18, [u.EHz], [1])), 155 * u.Hz)
        # starting from higher unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.Mg], [1])), 100 * u.ug)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(5e-13, [u.kA], [1])), 500 * u.pA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-17, [u.Gs], [1])), 10 * u.ns)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.Eg], [1])), 100 * u.Mg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e-13, [u.EA], [1])), 200 * u.kA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-17, [u.Zs], [1])), 10 * u.ks)
        # starting from lower unit with small prefix to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.ug], [1])), 1 * u.pg)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.uA], [1])), 1 * u.aA)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-15, [u.ns], [1])), 1 * u.ys)

    def test_opt_single_base_toHigherSIUnit_power_pos(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.m], [2])), 1 * u.km**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.kg], [2])), 1 * u.Mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.A], [3])), 1 * u.kA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.m], [2])), 1 * u.Mm**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.kg], [3])), 1 * u.Gg**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e24, [u.A], [4])), 1 * u.MA**4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.m], [2])), 1 * u.Gm**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.kg], [2])), 1 * u.Tg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e45, [u.A], [5])), 1 * u.GA**5)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.Gg], [2])), 1 * u.Eg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e36, [u.MA], [3])), 1 * u.EA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e75, [u.Ms], [5])), 1 * u.Zs**5)
        # starting from lower unit with small prefix to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.ug], [3])), 1 * u.g**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e36, [u.pA], [3])), 1 * u.A**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e75, [u.fs], [5])), 1 * u.s**5)
        # starting from lower unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.ug], [2])), 1 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.ng], [2])), 1 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e27, [u.fHz], [3])), 1 * u.uHz**3)
        # starting from lower unit to high unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e30, [u.ug], [2])), 1 * u.Gg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e90, [u.ug], [10])), 1 * u.kg**10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e180, [u.fHz], [10])), 1 * u.kHz**10)

    def test_opt_single_base_inBetweenHigherSIUnit_power_pos(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e7, [u.m], [2])), 10 * u.km**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.kg], [2])), 1e3 * u.Mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e14, [u.A], [3])), 1e5 * u.kA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e13, [u.m], [2])), 20 * u.Mm**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e20, [u.kg], [3])), 100 * u.Gg**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e34, [u.A], [4])), 1e10 * u.MA**4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(555e18, [u.m], [2])), 555 * u.Gm**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(99e18, [u.kg], [2])), 99 * u.Tg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e50, [u.A], [5])), 1e5 * u.GA**5)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e20, [u.Gg], [2])), 100 * u.Eg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e40, [u.MA], [3])), 1e4 * u.EA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e80, [u.Ms], [5])), 1e5 * u.Zs**5)
        # starting from lower unit with small prefix to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e19, [u.ug], [3])), 10 * u.g**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e40, [u.pA], [3])), 1e4 * u.A**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(6e85, [u.fs], [5])), 6e10 * u.s**5)
        # starting from lower unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(99e6, [u.ug], [2])), 99 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(8e16, [u.ng], [2])), 8e4 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e30, [u.fHz], [3])), 1e3 * u.uHz**3)
        # starting from lower unit to high unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e33, [u.ug], [2])), 1e3 * u.Gg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e99, [u.ug], [10])), 1e9 * u.kg**10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(11e181, [u.fHz], [10])), 110 * u.kHz**10)

    def test_opt_single_base_toLowerSIUnit_power_pos(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.m], [2])), 1 * u.mm**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.kg], [3])), 1 * u.g**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.A], [4])), 1 * u.mA**4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-24, [u.m], [4])), 1 * u.um**4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.kg], [2])), 1 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.A], [3])), 1 * u.uA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-45, [u.m], [5])), 1 * u.nm**5)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-90, [u.kg], [10])), 1 * u.ug**10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-27, [u.A], [3])), 1 * u.nA**3)
        # starting from higher unit to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.Mg], [2])), 1 * u.g**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-27, [u.Tg], [3])), 1 * u.kg**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-180, [u.EHz], [10])), 1 * u.Hz**10)
        # starting from higher unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.Mg], [2])), 1 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-24, [u.kA], [2])), 1 * u.nA**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-45, [u.Gs], [3])), 1 * u.us**3)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.Eg], [2])), 1 * u.Gg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-24, [u.EA], [2])), 1 * u.MA**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-45, [u.Zs], [3])), 1 * u.Ms**3)
        # starting from lower unit with small prefix to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.ug], [2])), 1 * u.pg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-36, [u.uA], [3])), 1 * u.aA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-60, [u.ns], [4])), 1 * u.ys**4)

    def test_opt_single_base_inBetweenLowerSIUnit_power_pos(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-5, [u.m], [2])), 10 * u.mm**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-5, [u.kg], [3])), 1e4 * u.g**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.A], [4])), 1 * u.mA**4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-14, [u.m], [4])), 1e10 * u.um**4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.kg], [2])), 100 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-11, [u.A], [3])), 1e7 * u.uA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-40, [u.m], [5])), 1e5 * u.nm**5)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-80, [u.kg], [10])), 1e10 * u.ug**10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.A], [3])), 1e9 * u.nA**3)
        # starting from higher unit to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.Mg], [2])), 100 * u.g**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-23, [u.Tg], [3])), 1e4 * u.kg**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-160, [u.EHz], [10])), 1e20 * u.Hz**10)
        # starting from higher unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-15, [u.Mg], [2])), 1e3 * u.mg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-20, [u.kA], [2])), 1e4 * u.nA**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-40, [u.Gs], [3])), 1e5 * u.us**3)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-17, [u.Eg], [2])), 10 * u.Gg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-21, [u.EA], [2])), 1e3 * u.MA**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-41, [u.Zs], [3])), 1e4 * u.Ms**3)
        # starting from lower unit with small prefix to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.ug], [2])), 100 * u.pg**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-30, [u.uA], [3])), 1e6 * u.aA**3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-50, [u.ns], [4])), 1e10 * u.ys**4)

    def test_opt_single_base_toHigherSIUnit_power_neg(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.m], [-2])), 1 * u.km**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.kg], [-2])), 1 * u.Mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-9, [u.A], [-3])), 1 * u.kA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.m], [-2])), 1 * u.Mm**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.kg], [-3])), 1 * u.Gg**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-24, [u.A], [-4])), 1 * u.MA**-4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.m], [-2])), 1 * u.Gm**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.kg], [-2])), 1 * u.Tg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-45, [u.A], [-5])), 1 * u.GA**-5)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.Gg], [-2])), 1 * u.Eg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-36, [u.MA], [-3])), 1 * u.EA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-75, [u.Ms], [-5])), 1 * u.Zs**-5)
        # starting from lower unit with small prefix to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-18, [u.ug], [-3])), 1 * u.g**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-36, [u.pA], [-3])), 1 * u.A**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-75, [u.fs], [-5])), 1 * u.s**-5)
        # starting from lower unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-6, [u.ug], [-2])), 1 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-12, [u.ng], [-2])), 1 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-27, [u.fHz], [-3])), 1 * u.uHz**-3)
        # starting from lower unit to high unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-30, [u.ug], [-2])), 1 * u.Gg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-90, [u.ug], [-10])), 1 * u.kg**-10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-180, [u.fHz], [-10])), 1 * u.kHz**-10)

    def test_opt_single_base_inBetweenHigherSIUnit_power_neg(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-5, [u.m], [-2])), 10 * u.km**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e-4, [u.kg], [-2])), 200 * u.Mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(999e-9, [u.A], [-3])), 999 * u.kA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.m], [-2])), 100 * u.Mm**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-15, [u.kg], [-3])), 1000 * u.Gg**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-20, [u.A], [-4])), 1e4 * u.MA**-4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-14, [u.m], [-2])), 1e4 * u.Gm**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(9e-18, [u.kg], [-2])), 9 * u.Tg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-40, [u.A], [-5])), 1e5 * u.GA**-5)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(3e-17, [u.Gg], [-2])), 30 * u.Eg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-35, [u.MA], [-3])), 10 * u.EA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-70, [u.Ms], [-5])), 1e5 * u.Zs**-5)
        # starting from lower unit with small prefix to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e-16, [u.ug], [-3])), 200 * u.g**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(3e-33, [u.pA], [-3])), 3e3 * u.A**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(88e-75, [u.fs], [-5])), 88 * u.s**-5)
        # starting from lower unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-4, [u.ug], [-2])), 100 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-11, [u.ng], [-2])), 10 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-23, [u.fHz], [-3])), 1e4 * u.uHz**-3)
        # starting from lower unit to high unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-28, [u.ug], [-2])), 100 * u.Gg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-89, [u.ug], [-10])), 10 * u.kg**-10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-170, [u.fHz], [-10])), 1e10 * u.kHz**-10)

    def test_opt_single_base_toLowerSIUnit_power_neg(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e6, [u.m], [-2])), 1 * u.mm**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e9, [u.kg], [-3])), 1 * u.g**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.A], [-4])), 1 * u.mA**-4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e24, [u.m], [-4])), 1 * u.um**-4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.kg], [-2])), 1 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.A], [-3])), 1 * u.uA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e45, [u.m], [-5])), 1 * u.nm**-5)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e90, [u.kg], [-10])), 1 * u.ug**-10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e27, [u.A], [-3])), 1 * u.nA**-3)
        # starting from higher unit to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.Mg], [-2])), 1 * u.g**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e27, [u.Tg], [-3])), 1 * u.kg**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e180, [u.EHz], [-10])), 1 * u.Hz**-10)
        # starting from higher unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.Mg], [-2])), 1 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e24, [u.kA], [-2])), 1 * u.nA**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e45, [u.Gs], [-3])), 1 * u.us**-3)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e18, [u.Eg], [-2])), 1 * u.Gg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e24, [u.EA], [-2])), 1 * u.MA**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e45, [u.Zs], [-3])), 1 * u.Ms**-3)
        # starting from lower unit with small prefix to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e12, [u.ug], [-2])), 1 * u.pg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e36, [u.uA], [-3])), 1 * u.aA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e60, [u.ns], [-4])), 1 * u.ys**-4)

    def test_opt_single_base_inBetweenLowerSIUnit_power_neg(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e7, [u.m], [-2])), 20 * u.mm**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e11, [u.kg], [-3])), 100 * u.g**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(99e12, [u.A], [-4])), 99 * u.mA**-4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e30, [u.m], [-4])), 1e6 * u.um**-4)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e13, [u.kg], [-2])), 10 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e20, [u.A], [-3])), 100 * u.uA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e50, [u.m], [-5])), 1e5 * u.nm**-5)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e89, [u.kg], [-10])), 0.1 * u.ug**-10)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e29, [u.A], [-3])), 100 * u.nA**-3)
        # starting from higher unit to base
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e16, [u.Mg], [-2])), 1e4 * u.g**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e31, [u.Tg], [-3])), 1e4 * u.kg**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e181, [u.EHz], [-10])), 10 * u.Hz**-10)
        # starting from higher unit to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e20, [u.Mg], [-2])), 100 * u.mg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(7e25, [u.kA], [-2])), 70 * u.nA**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e50, [u.Gs], [-3])), 1e5 * u.us**-3)
        # starting from higher unit to higher unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(9e21, [u.Eg], [-2])), 9000 * u.Gg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e28, [u.EA], [-2])), 1e4 * u.MA**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e50, [u.Zs], [-3])), 1e5 * u.Ms**-3)
        # starting from lower unit with small prefix to lower unit
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(7e14, [u.ug], [-2])), 700 * u.pg**-2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(2e37, [u.uA], [-3])), 20 * u.aA**-3)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e71, [u.ns], [-4])), 1e11 * u.ys**-4)

    def test_opt_centimeters(self):
        # unit power
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e2, [u.cm], [1])), 1 * u.m)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e5, [u.cm], [1])), 1 * u.km)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e7, [u.cm], [1])), 100 * u.km)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(0.1, [u.cm], [1])), 1 * u.mm)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-4, [u.cm], [1])), 1 * u.um)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(9e-2, [u.cm], [1])), 900 * u.um)
        # positive power
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e4, [u.cm], [2])), 1 * u.m ** 2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e10, [u.cm], [5])), 1 * u.m ** 5)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e7, [u.cm], [2])), 1e3 * u.m**2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-8, [u.cm], [2])), 1 * u.um ** 2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-5, [u.cm], [2])), 1e3 * u.um ** 2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.cm], [3])), 100 * u.um ** 3)
        # negative power
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-4, [u.cm], [-2])), 1 * u.m ** -2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-10, [u.cm], [-2])), 1 * u.km ** -2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e-8, [u.cm], [-2])), 100 * u.km ** -2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(100, [u.cm], [-2])), 1 * u.mm ** -2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(1e8, [u.cm], [-2])), 1 * u.um ** -2)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(5e10, [u.cm], [-2])), 500 * u.um ** -2)

    def test_opt_single_base_negative_scale(self):
        # starting from unit without prefix (or Kg because is SI unit)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(-1e3, [u.m], [1])), -1 * u.km)
        self.assertEqual(ut.opt_single_base(
            CompositeUnit(-1e-3, [u.m], [-1])), -1 / u.km)

    def test_opt_comp_units(self):
        test_bases = (u.m, u.s, u.Hz, u.W, u.kJ, u.GA,
                      u.Tmol, u.uN, u.pPa, u.aRy)
        for _ in range(1000):
            unit = get_random_unit(test_bases)
            scale, bases, powers = unit.scale, unit.bases, unit.powers
            # we expect an optimization on the first base
            first_base = CompositeUnit(scale, [bases[0]], [powers[0]])
            opt_first_base = ut.opt_single_base(first_base)
            opt_unit_first = CompositeUnit(opt_first_base.scale,
                                           opt_first_base.bases + bases[1:],
                                           opt_first_base.powers + powers[1:])
            opt_unit_full = ut.opt_comp_units(unit)
            self.assertEqual(opt_unit_full, opt_unit_first)

    def test_optimize_unit(self):
        """ testing optimize_units regarding quantities """
        # should do nothing for zero
        self.assertEqual(ut.optimize_unit(0), 0)
        self.assertEqual(ut.optimize_unit(
            Quantity(0, u.km)),
            Quantity(0, u.km))
        self.assertEqual(ut.optimize_unit(
            Quantity(0, u.km), fix_time_freq=True),
            Quantity(0, u.km))
        # should no nothing for numbers and dimensionless quantities
        self.assertEqual(ut.optimize_unit(3), 3)
        self.assertEqual(ut.optimize_unit(-12.4), -12.4)
        # should do nothing for Named units
        self.assertEqual(u.km, u.km)
        self.assertEqual(u.ps, u.ps)
        # should optimize CompositeUnit
        u1 = CompositeUnit(1000, [u.m], [1])
        self.assertEqual(u1, ut.opt_comp_units(u1))
        u2 = CompositeUnit(1e-3, [u.m], [-1])
        self.assertEqual(u2, ut.opt_comp_units(u2))
        u3 = CompositeUnit(1e6, [u.m, u.MHz], [2, 3])
        self.assertEqual(u3, ut.opt_comp_units(u3))
        u4 = CompositeUnit(5e-8, [u.m, u.MHz], [-2, 3])
        self.assertEqual(u4, ut.opt_comp_units(u4))
        # should optimize Quantity
        self.assertEqual(ut.optimize_unit(1000 * u.m), 1 * u.km)
        self.assertEqual(ut.optimize_unit(-1000 * u.m), -1 * u.km)
        self.assertEqual(ut.optimize_unit(1e-3 / u.m), 1 / u.km)
        self.assertEqual(ut.optimize_unit(1e6 * u.m), 1 * u.Mm)
        self.assertEqual(ut.optimize_unit(1e7 * u.m), 10 * u.Mm)
        self.assertEqual(ut.optimize_unit(1e6 * u.m**2), 1 * u.km**2)
        self.assertEqual(ut.optimize_unit(1e8 * u.m ** 2), 100 * u.km ** 2)
        self.assertEqual(ut.optimize_unit(1e-6 / u.m**2), 1 / u.km**2)
        # this case reflects the choice of only using unit scales > 1
        self.assertEqual(ut.optimize_unit(1e-5 / u.m ** 2), 1e-5 / u.m ** 2)
        # testing the fix_hertz option
        self.assertEqual(ut.optimize_unit(1e3 / u.Hz, fix_time_freq=True),
                                          1 * u.ks)
        self.assertEqual(ut.optimize_unit(1e4 / u.Hz, fix_time_freq=True),
                                          10 * u.ks)
        self.assertEqual(ut.optimize_unit(1e6 / u.Hz**2, fix_time_freq=True),
                                          1 * u.ks**2)
        self.assertEqual(ut.optimize_unit(1e8 / u.Hz ** 2, fix_time_freq=True),
                                          100 * u.ks ** 2)
        self.assertEqual(ut.optimize_unit(1e3 / u.s, fix_time_freq=True),
                                          1 * u.kHz)
        self.assertEqual(ut.optimize_unit(5e3 / u.s, fix_time_freq=True),
                                          5 * u.kHz)
        self.assertEqual(ut.optimize_unit(1e6 / u.s ** 2, fix_time_freq=True),
                                          1 * u.kHz ** 2)
        self.assertEqual(ut.optimize_unit(2e8 / u.s ** 2, fix_time_freq=True),
                                          200 * u.kHz ** 2)
        # should work also for arrays
        self.assertQuantityEqual(ut.optimize_unit(
            Quantity([1000, 2000, 3000], u.m)),
            Quantity([1, 2, 3], u.km))
        self.assertQuantityEqual(ut.optimize_unit(
            Quantity([1e4, 2e3, -3e3], u.m)),
            Quantity([10, 2, -3], u.km))
        self.assertQuantityEqual(ut.optimize_unit(
            Quantity([1e-3, 2e-3, 3e-3], 1/u.kHz), fix_time_freq=True),
            Quantity([1, 2, 3], u.us))
        # random tests
        for _ in range(100):
            # picking a random unit
            unit = get_random_unit()
            value = rnd.randrange(-1e5, 1e5)
            quant = value * unit
            # manual optimization for comparison
            opt_unit = ut.opt_comp_units(CompositeUnit(
                value * unit.scale, unit.bases, unit.powers))
            new_value = opt_unit.scale
            self.assertEqual(
                ut.optimize_unit(quant),
                new_value * CompositeUnit(1, opt_unit.bases, opt_unit.powers))


if __name__ == "__main__":
    unittest.main()
