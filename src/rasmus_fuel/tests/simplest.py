from rasmus_fuel.simplest import power_maintain_sog


def test_simplest_power_maintain_sog_no_movement():
    assert 0 == power_maintain_sog(u_ship=0, v_ship=0, u_current=0, v_current=0)
