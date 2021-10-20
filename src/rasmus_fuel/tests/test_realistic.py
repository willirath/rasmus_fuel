import numpy as np
import pytest

from rasmus_fuel.realistic import power_maintain_sog, power_to_fuel_burning_rate, power_to_fuel_consump, energy_efficiency_per_time_distance

def test_realistic_power_maintain_sog_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == power_maintain_sog(
        u_ship_og = 0,
        v_ship_og = 0,
        u_current = 0,
        v_current = 0,
        u_wind = 0,
        v_wind = 0,
        w_wave_hight = 0,
    )

def test_realistic_power_maintain_sog_no_drag():
    """Check for zero power if drag is removed."""
    u_ship_og = np.random.normal(size = (10,))
    v_ship_og = np.random.normal(size = (10,))
    u_current = np.random.normal(size = (10,))
    v_current = np.random.normal(size = (10,)) 
    v_wind = np.random.normal(size = (10,))
    u_wind = np.random.normal(size = (10,))
    w_wave_hight = np.random.normal(size = (10,))
    np.testing.assert_equal(
        np.zeros(shape = (10,)),
        power_maintain_sog(
            u_ship_og = u_ship_og,
            v_ship_og = v_ship_og,
            u_current = u_current,
            v_current = v_current,
            u_wind = u_wind,
            v_wind = v_wind,
            w_wave_hight = w_wave_hight,
            vessel_water_drag_coefficient = 0,
            vessel_wind_resistance_coefficient = 0,
            vessel_draught = 0,
        ),
    )


def test_power_maintain_sog_isotropic():
    """Ensure unchanged results if swapping directions."""
    np.testing.assert_allclose(
        power_maintain_sog(u_ship_og = 1, u_current = 5, v_ship_og = 0, v_current = 0, 
            u_wind = 1,v_wind = 2, w_wave_hight = 0),
        power_maintain_sog(u_ship_og = 0, u_current = 0, v_ship_og = 1, v_current = 5, 
            u_wind = 1,v_wind = 2, w_wave_hight = 0),
    )


def test_power_maintain_sog_errors():
    """Check if correct errors are raised"""
    u_ship_og = np.array([1, 2, 3])
    v_ship_og = np.array([1, 2])
    with pytest.raises(ValueError) as execinfo:
        power_maintain_sog(u_ship_og = u_ship_og, v_ship_og = v_ship_og, u_current = 0, v_current = 0,
        u_wind = 1, v_wind = 2, w_wave_hight = 0)
    assert str(execinfo.value) == 'Shape of u_ship_og and v_ship_og need to agree.'


def test_power_maintain_sog_kwargs_ignored():
    """Check that arbitrary kwargs are ignored."""
    np.testing.assert_equal(
        power_maintain_sog(
            u_ship_og = 1, v_ship_og = 1, u_current = 0, v_current = 0, u_wind = 1, v_wind = 2, w_wave_hight = 0, 
            nonexisting_kwarg=1234
        ),
        power_maintain_sog(u_ship_og = 1, v_ship_og = 1, u_current = 0, v_current = 0, u_wind = 1,v_wind = 2, w_wave_hight = 0,
        ),
    )

def test_power_to_fuel_consump_just_call():
    power_to_fuel_consump(engine_power = 1.0, steaming_time = 0.5, distance = 12e3)

def test_power_to_fuel_consump_zero():
    """ Test if fuel consumption is zero at zero engine power"""
    assert 0 == power_to_fuel_consump(engine_power = 0.0, steaming_time = 0.5, distance = 1.0) 

def test_power_to_fuel_consump_correct_value():
    """ Test if fuel consumption returns correct value"""
    assert 180.0 * 0.5 /33434.0 == power_to_fuel_consump(engine_power = 1.0, steaming_time = 0.5, distance = 1.0)  

def test_power_to_fuel_burning_rate_just_call():
    power_to_fuel_burning_rate(power = 1.0, efficiency = 0.5, fuel_value = 42e6)

def test_power_to_fuel_burning_rate_zero():
    """ Test if fuel burning rate is zero at zero engine power"""
    assert 0 == power_to_fuel_burning_rate(power = 0.0, efficiency = 0.5, fuel_value = 1.0) 

def test_power_to_fuel_burning_rate_correct_value():
    """ Test if fuel burning rate return correct value"""
    assert 2.0 == power_to_fuel_burning_rate(power = 1.0, efficiency = 0.5, fuel_value = 1.0)    

def test_energy_efficiency_per_time_distance_just_call():
    energy_efficiency_per_time_distance(fuel_consumption = 100.0)       

def test_energy_efficiency_per_time_distance_zero():
    """ Test if energy efficiency is zero"""
    assert 0.0 == energy_efficiency_per_time_distance(fuel_consumption = 0.0)

def test_energy_efficiency_per_time_distance_correct_value():
    """ Test if energy efficiency return correct value"""
    assert 320.6 == energy_efficiency_per_time_distance(fuel_consumption = 100.0)    

# TODO: test with actual numbers for plausibility checks