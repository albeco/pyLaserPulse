"""
units_utils provides extra functions to deal
with astropy units package. Specifically to reformat
inverse units and physical quantities
"""

from astropy import units
from astropy.units.core import UnitBase, CompositeUnit, NamedUnit, PrefixUnit
from astropy.units.quantity import Quantity
import numpy as np
from typing import Tuple, Union

RealNumber = Union[int, float]
ComplexNumber = Union[int, float, complex]

# define SI prefixed and dictionaries to convert between exponent and prefix
si_prefixes = 'yzafpnumkMGTPEZY'
si_exponents = tuple(x for x in range(-24, 25, 3) if x != 0)
attosecond_exponent = -18

prefix_map = {pref: exp for (pref, exp) in zip(si_prefixes, si_exponents)}
exponent_map = {exp: pref for (pref, exp) in prefix_map.items()}


def pref_to_expon(pref: str) -> int:
    """ returns the exponent of a given SI prefix"""
    try:
        expon = prefix_map[pref]
    except KeyError:
        raise ValueError("'{}' is not a valid SI prefix".format(pref))
    return expon


def expon_to_pref(expon: int) -> str:
    """ returns the SI prefix corresponding to a given exponent """
    try:
        pref = exponent_map[expon]
    except KeyError:
        raise ValueError("{} is must be in range(-24,25,3)".format(expon))
    return pref


def is_close(x: ComplexNumber,
             y: ComplexNumber,
             rtol: RealNumber = 1e-6,
             atol: RealNumber = 1e-10) -> bool:
    """ compare two numbers for equality with tolerance """
    return abs(x - y) <= atol + rtol * abs(y)


def is_unity(x: ComplexNumber) -> bool:
    """checks if the input is a number close to 1"""
    return is_close(x, 1)


def is_time_freq(x: NamedUnit) -> bool:
    """ true if dimension is time or frequency, false otherwise """
    return x.physical_type == 'time' or x.physical_type == 'frequency'


def flip_prefix(pref: str) -> str:
    """returns 1/prefix"""
    idx = len(si_prefixes) - 1 - si_prefixes.find(pref)
    return si_prefixes[idx]


def fixhertz_item(base: NamedUnit,
                  pwr: RealNumber) -> Tuple[NamedUnit, RealNumber]:
    """ replaces (1/s -> Hertz) and (1/Hertz -> second) in a NamedUnit """
    if pwr >= 0 or not is_time_freq(base):
        return base, pwr
    if not isinstance(base, PrefixUnit):
        if base == units.second:
            return units.hertz, -pwr
        elif base == units.hertz:
            return units.second, -pwr
        else:
            return base, pwr
    # from now dealing with prefix units
    # in astropy.units attosecond cannot be abbreviated
    if base == units.attosecond:
        return units.exahertz, -pwr
    prefix, base_name = base.name[0], base.name[1:]
    new_prefix = flip_prefix(prefix)
    if base_name == 's':
        return units.Unit(new_prefix + 'Hz'), -pwr
    elif base_name == 'Hz':
        return units.Unit(new_prefix + 's'), -pwr
    else:
        return base, pwr


def fixhertz_comp(u2: CompositeUnit) -> UnitBase:
    """ replaces (1/s -> Hertz) and (1/Hertz -> second) in a CompositeUnit """
    fixes = tuple(map(fixhertz_item, u2.bases, u2.powers))
    try:
        new_bases, new_powers = zip(*fixes)
    except ValueError:
        # return dimensionless units unchanged
        return u2
    if is_unity(u2.scale) and len(new_bases) == 1 and is_unity(new_powers[0]):
        new_unit = new_bases[0]
    else:
        new_unit = CompositeUnit(u2.scale, new_bases, new_powers)
    return new_unit


def fixhertz(x: Union[ComplexNumber, Quantity, UnitBase]) \
        -> Union[ComplexNumber, Quantity, UnitBase]:
    """ replaces 1/s with Hertz in physical units and quantities """
    if isinstance(x, Quantity):
        return Quantity(x.value, fixhertz_comp(x.unit))
    elif isinstance(x, CompositeUnit):
        return fixhertz_comp(x)
    else:
        return x


def opt_single_base(simple_unit: CompositeUnit) -> CompositeUnit:
    """ optimizes composite units with a single base """
    if not isinstance(simple_unit, CompositeUnit):
        raise TypeError("opt_single_base expects a CompositeUnit")
    if len(simple_unit.bases) > 1:
        raise TypeError("opt_single_base expects CompositeUnit with 1 base")
    sign = 1 if simple_unit.scale >= 0 else -1
    value = abs(simple_unit.scale)
    unit = simple_unit.bases[0]
    # in astropy.units attosecond cannot be abbreviated
    if unit == units.attosecond:
        unit_name, expon = 's', attosecond_exponent
    # in astropy.units kilogram is not PrefixUnit (it is IrreducibleUnit)
    elif unit == units.kilogram:
        unit_name, expon = 'g', 3
    # centimeter is not a PrefixUnit
    elif unit == units.centimeter:
        unit_name, expon = 'm', -2
    elif isinstance(unit, PrefixUnit):
        unit_name, expon = unit.name[1:], pref_to_expon(unit.name[0])
    else:
        unit_name, expon = unit.name, 0
    pwr = simple_unit.powers[0]
    new_expon = min(
        24, max(-24, np.floor((expon + np.log10(value ** (1 / pwr))) / 3) * 3))
    new_value = value * 10 ** (expon * pwr - new_expon * pwr)
    new_prefix = expon_to_pref(new_expon) if new_expon != 0 else ''
    new_unit_name = new_prefix + unit_name
    # astropy.units treats attoseconds differently
    if new_unit_name == 'as':
        new_unit = units.attosecond
    else:
        new_unit = units.Unit(new_unit_name)
    return CompositeUnit(sign * new_value, [new_unit], [pwr])


def opt_named_unit(unit: NamedUnit) -> NamedUnit:
    """ optimized named unit """
    # Named unit do not need to be optimized, because scale always == 1
    # (they do not overwrite the scale property of BasedUnit, which returns 1)
    return unit


def opt_comp_units(unit: CompositeUnit) -> CompositeUnit:
    """
    optimizes CompositeUnit instances
    it optimizes only the first base
    """
    if isinstance(unit.bases[0], CompositeUnit):
        dec_base = unit.bases[0].decompose()
        sub_base = CompositeUnit(
            unit.scale, dec_base.bases[0], dec_base.powers[0] * unit.powers[0])
        opt_sub_base = opt_comp_units(sub_base)
        new_base = CompositeUnit(
            opt_sub_base.scale, opt_sub_base.bases + dec_base.bases[1:],
            opt_sub_base.powers + dec_base.powers[1:])
    else:
        temp_base = CompositeUnit(
            unit.scale, [unit.bases[0]], [unit.powers[0]])
        new_base = opt_single_base(temp_base)
    return CompositeUnit(new_base.scale,
                         new_base.bases + unit.bases[1:],
                         new_base.powers + unit.powers[1:])


def optimize_unit(quant: Union[ComplexNumber, Quantity, UnitBase],
                  fix_time_freq: bool = False) \
        -> Union[ComplexNumber, Quantity, UnitBase]:
    """ optimize physical units and quantities """
    # if input is a unit rather than a quantity delegate its handling
    if isinstance(quant, NamedUnit):
        return opt_named_unit(quant)
    elif isinstance(quant, CompositeUnit):
        return opt_comp_units(quant)
    # if input is not unit or quantity return it unchanged
    if not isinstance(quant, Quantity):
        return quant

    value, unit = quant.value, quant.unit
    try:
        mean_value = np.mean([abs(x) for x in value])
    except TypeError:
        mean_value = value
    if mean_value == 0:
        return quant
    scaled_value = value / mean_value
    if isinstance(unit, CompositeUnit):
        new_unit = CompositeUnit(mean_value * unit.scale,
                                 unit.bases, unit.powers)
    else:
        # quant is a NamedUnit
        new_unit = CompositeUnit(mean_value, [quant.unit], [1])
    if fix_time_freq:
        new_unit = fixhertz(new_unit)
    opt_unit = opt_comp_units(new_unit)
    return (scaled_value *
            opt_unit.scale *
            CompositeUnit(1, opt_unit.bases, opt_unit.powers))
