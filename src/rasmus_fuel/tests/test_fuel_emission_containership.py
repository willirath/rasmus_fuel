import numpy as np
import pytest

from rasmus_fuel.fuel_emission_containership import (
    convert_emission_kgpermeter_kgperNM,
    convert_fuel_consumption_tonnsperday,
    emission_CO2_diesel_MANandBW,
    fuel_consumption_diesel_MANandBW,
)


def test_fuel_consumption_diesel_MANandBW_no_engine():
    """Check for no power if no movement is happening."""
    assert 0 == fuel_consumption_diesel_MANandBW(
        speed_ship_og=10.0,
        engine_power=0,
    )


def test_fuel_consumption_diesel_MANandBW_errors():
    """Check if correct errors are raised"""
    speed_ship = np.array([1, 2, 3])
    power = np.array([1, 2])
    with pytest.raises(ValueError) as execinfo:
        fuel_consumption_diesel_MANandBW(
            speed_ship_og=speed_ship,
            engine_power=power,
        )
    assert str(execinfo.value) == 'Shape of speed_ship_og and engine_power need to agree.'


def fuel_consumption_diesel_MANandBW_kwargs_ignored():
    """Check that arbitrary kwargs are ignored."""
    np.testing.assert_equal(
        fuel_consumption_diesel_MANandBW(
            speed_ship_og=1,
            engine_power=1,
        ),
        fuel_consumption_diesel_MANandBW(
            speed_ship_og=1,
            engine_power=1,
            nonexisting_kwarg=1234,
        ),
    )


def test_emission_CO2_diesel_MANandBW_zero_fuel_consumption():
    """Check for zero emission in case of zero fuel consumption provided."""
    assert 0 == emission_CO2_diesel_MANandBW(
        fuel_consumption=0,
        sailing_time=3600.0,
    )


def test_emission_CO2_diesel_MANandBW_ship_at_port():
    """Check for zero emission if sailing time is zero (vessel at port condition)."""
    assert 0 == emission_CO2_diesel_MANandBW(
        fuel_consumption=23.0,
        sailing_time=0,
    )


def emission_CO2_diesel_MANandBW_kwargs_ignored():
    """Check that arbitrary kwargs are ignored."""
    np.testing.assert_equal(
        emission_CO2_diesel_MANandBW(
            fuel_consumption=23.0,
            sailing_time=10,
        ),
        emission_CO2_diesel_MANandBW(
            fuel_consumption=23.0,
            sailing_time=10,
            nonexisting_kwarg=1234,
        ),
    )


def test_convert_fuel_consumption_tonnsperday_conversion_zero():
    """Check for zero return if zero fuel consumption is provided."""
    assert 0 == convert_fuel_consumption_tonnsperday(
        fuel_consumption=0.0,
    )


def test_convert_emission_kgpermeter_kgperNM_conversion_zero():
    """Check conversion of emission from zero input to zero output."""
    assert 0 == convert_emission_kgpermeter_kgperNM(
        emission=0.0,
    )


def test_emission_CO2_diesel_MANandBW_example_values():
    """Check correct units of input time """
    np.testing.assert_allclose(
        64.12,
        emission_CO2_diesel_MANandBW(
            fuel_consumption=10.0,
            sailing_time=2.0,
            vessel_conversion_factor_fuel_toCO2=3.206,
        ),
    )
