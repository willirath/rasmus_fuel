import numpy as np
import numpy.typing as npt


def power_maintain_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    drag_coeff: float = 1.0,
    **kwargs,
):
    """Calculate quadratic drag law power needed to maintain speed over ground.

    Parameters
    ----------
    u_ship_og: array
        Ship eastward speed over ground in meters per second.
    v_ship_og: array
        Ship northward speed over ground in meters per second.
        Needs same shape as u_ship_og.
    u_current: array
        Ocean currents eastward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship_og and v_ship_og.
    v_current: array
        Ocean currents northward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship_og and v_ship_og.
    drag_coeff: float
        Drag coefficient in arbitrary units.

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        Fuel consumption for each element of u_ship_og and v_ship_og.
        Shape will be identical to u_ship_og and v_ship_og.

    """
    # ensure shapes of u_ship_og and v_ship_og agree
    if np.array(u_ship_og).shape != np.array(v_ship_og).shape:
        raise ValueError('Shape of u_ship_og and v_ship_og need to agree.')

    # calc speed through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current

    # calc power to maintain speed over ground
    power_needed = (
        drag_coeff
        * (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5
        * (u_ship_tw * u_ship_og + v_ship_tw * v_ship_og)
    )

    return power_needed
