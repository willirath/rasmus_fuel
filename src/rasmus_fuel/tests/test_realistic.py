# import numpy as np
# import pytest

from rasmus_fuel.realistic import power_maintain_sog


def test_simplest_power_maintain_sog_no_movement():
    """Check for no power if no movement is happening."""
    assert 0 == power_maintain_sog(
        u_ship_og=0,
        v_ship_og=0,
        course_ship_og=0,
        u_current=0,
        v_current=0,
        u_wind=0,
        v_wind=0,
        w_wave_hight=0,
    )
