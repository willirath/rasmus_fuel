import numpy as np
import numpy.typing as npt


def power_maintain_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    coeff: float = 1.0,
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
    coeff: float
        Coefficient in units of kg/m. This coefficient contains info about
        the density of water, the effective cross section of the vessel and the
        drag coefficient of the vessel.  Defaults to 1.0

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        Power in W (=kg*m2/s3) needed to maintain speed over ground for each
        element of u_ship_og and v_ship_og. Shape will be identical to
        u_ship_og and v_ship_og.

    """
    # ensure shapes of u_ship_og and v_ship_og agree
    if np.array(u_ship_og).shape != np.array(v_ship_og).shape:
        raise ValueError('Shape of u_ship_og and v_ship_og need to agree.')

    # calc speed through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current

    # calc power to maintain speed over ground
    power_needed = (
        coeff
        * (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5
        * (u_ship_tw * u_ship_og + v_ship_tw * v_ship_og)
    )

    return power_needed


def power_to_fuel_burning_rate(
    power: npt.ArrayLike = None, efficiency: float = 0.1, fuel_value: float = 42.0e6
):
    """Convert power to fuel buring rate.

    Parameters
    ----------
    power: array
        Power in W (=kg*m2/s3).
    efficiency: float
        Fraction of fuel value turned into propulsive force. Defaults to 0.1.
    fuel_value: float
        Fuel value in J/kg. Defaults to 42.0e6.

    Returns
    -------
    array
        Fuel burning rate in kg/s.

    """
    raise NotImplementedError
