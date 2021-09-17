import numpy as np
import numpy.typing as npt


def power_maintain_sog(
    u_ship: npt.ArrayLike = None,
    v_ship: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    drag_coeff: float = 1.0,
    **kwargs,
):
    """Calculate quadratic drag law power needed to maintain speed over ground.

    Parameters
    ----------
    u_ship: array
        Ship eastward speed over ground in meters per second.
    v_ship: array
        Ship northward speed over ground in meters per second.
        Needs same shape as u_ship.
    u_current: array
        Ocean currents eastward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship and v_ship.
    v_current: array
        Ocean currents northward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship and v_ship.
    drag_coeff: float
        Drag coefficient in arbitrary units.

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        Fuel consumption for each element of u_ship and v_ship.
        Shape will be identical to u_ship and v_ship.

    """
    # ensure shapes of u_ship and v_ship agree
    if np.array(u_ship).shape != np.array(v_ship).shape:
        raise ValueError('Shape of u_ship and v_ship need to agree.')

    # calc power to maintain speed over ground
    power_needed = (
        drag_coeff
        * ((u_ship - u_current) ** 2 + (v_ship - v_current) ** 2) ** 0.5
        * ((u_ship - u_current) * u_ship + (v_ship - v_current) * v_ship)
    )

    return power_needed
