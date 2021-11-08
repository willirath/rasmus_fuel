import numpy as np
import pytest

from rasmus_fuel.realistic import (
    fuel_consumption_diesel_MANandBW,
    convert_fuel_consumption_tonnsperday,
    convert_emission_kgpermeter_kgperNM,
    emission_CO2_diesel_MANandBW,
)

def test_fuel_consumption_diesel_MANandBW_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == fuel_consumption_diesel_MANandBW(
        speed_ship_og = 0,
        engine_power = 0,
    )

def test_emission_CO2_diesel_MANandBW_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == emission_CO2_diesel_MANandBW(
       fuel_consumption = 0,
       sailing_time = 3600.0,
    )

def test_emission_CO2_diesel_MANandBW_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == emission_CO2_diesel_MANandBW(
       fuel_consumption = 23.0,
       sailing_time = 0,
    )    

def test_convert_fuel_consumption_tonnsperday_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == convert_fuel_consumption_tonnsperday(
       fuel_consumption = 0.0,
    )

def test_convert_emission_kgpermeter_kgperNM_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == convert_emission_kgpermeter_kgperNM(
       emission = 0.0,
    )    

# TODO: test with actual numbers for plausibility checks