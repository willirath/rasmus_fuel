import numpy as np
import pytest

from rasmus_fuel.simplest import power_maintain_sog, power_to_fuel_burning_rate


def test_simplest_power_maintain_sog_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == power_maintain_sog(u_ship_og=0, v_ship_og=0, u_current=0, v_current=0)


def test_simplest_power_maintain_sog_no_drag():
    """Check for zero power if drag is removed."""
    u_ship_og = np.random.normal(size=(10,))
    v_ship_og = np.random.normal(size=(10,))
    u_current = np.random.normal(size=(10,))
    v_current = np.random.normal(size=(10,))
    np.testing.assert_equal(
        np.zeros(shape=(10,)),
        power_maintain_sog(
            u_ship_og=u_ship_og,
            v_ship_og=v_ship_og,
            u_current=u_current,
            v_current=v_current,
            coeff=0,
        ),
    )


def test_simplest_power_maintain_sog_isotropic():
    """Ensure unchanged results if swapping directions."""
    np.testing.assert_allclose(
        power_maintain_sog(u_ship_og=1, u_current=5, v_ship_og=0, v_current=0),
        power_maintain_sog(u_ship_og=0, u_current=0, v_ship_og=1, v_current=5),
    )


def test_simplest_power_maintain_sog_errors():
    """Check if correct errors are raised"""
    u_ship_og = np.array([1, 2, 3])
    v_ship_og = np.array([1, 2])
    with pytest.raises(ValueError) as execinfo:
        power_maintain_sog(u_ship_og=u_ship_og, v_ship_og=v_ship_og, u_current=0, v_current=0)
    assert str(execinfo.value) == 'Shape of u_ship_og and v_ship_og need to agree.'


def test_simplest_power_maintain_sog_kwargs_ignored():
    """Check that arbitrary kwargs are ignored."""
    np.testing.assert_equal(
        power_maintain_sog(
            u_ship_og=1, v_ship_og=1, u_current=0, v_current=0, nonexisting_kwarg=1234
        ),
        power_maintain_sog(u_ship_og=1, v_ship_og=1, u_current=0, v_current=0),
    )


def test_power_to_fuel_burning_rate_just_call():
    power_to_fuel_burning_rate(power=1.0, efficiency=0.5, fuel_value=42e6)
