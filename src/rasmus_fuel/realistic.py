import numpy as np
import numpy.typing as npt


def power_maintain_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    course_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    u_wind: npt.ArrayLike = None,
    v_wind: npt.ArrayLike = None,
    w_wave_hight: npt.ArrayLike = None,
    coeff: float = 1.0,
    **kwargs,
) -> npt.ArrayLike:
    """Calculate quadratic drag law power needed to maintain speed over ground.
    Parameters
    ----------
    u_ship_og: array
        Ship eastward speed over ground in meters per second.
    v_ship_og: array
        Ship northward speed over ground in meters per second.
        Needs same shape as u_ship_og.
    course_ship_og: array
        Ship course over ground in radians
        needs same shape as u_ship_og
    u_current: array
        Ocean currents eastward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship_og and v_ship_og.
    v_current: array
        Ocean currents northward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship_og and v_ship_og.
    u_wind: array
        Eastward 10 m wind in m/s
        Needs shape that can be broadcst to shape of u_ship_og and v_ship_og
    v_wind: array
        Northward 10 m wind in m/s
        Needs shape that can be broadcst to shape of u_ship_og and v_ship_og
    w_wave_hight: array
        Spectral significant wave height (Hm0), meters
        Needs shape that can be broadcst to shape of u_ship_og and v_ship_og
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

    # calc engine power to maintain speed over ground using ocean current resistance term
    speed_tw = (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5
    power_needed = coeff * (speed_tw ** 3)

    # calc engine power to maintain speed over ground using : (1) ocean current resistance, (2) wind resistance, (3) wave resistance

    # calc relative wind speed
    wind_speed = (u_wind ** 2 + v_wind ** 2) ** 0.5
    wind_dir = arctan(v_wind/u_wind)
    wind_dir_rel = wind_dir - course_ship +  pi

    u_wind_rel = wind_speed * cos(wind_dir_rel) - u_ship_og
    v_wind_rel = wind_speed * sin(wind_dir_rel) - v_ship_og

    coeff_wind_drag = 0.5* model_param["air_mass_density"]* model_param["wind_resistance_coefficient"] * model_param["vessel_supersurface_area"]

    coeff_wave_drag = 20. * (model_param["waterline_width"] / model_param["waterline_length"]) ** (-1.2) * (1 / model_param["waterline_length"]) ** 0.62 \
         / model_param["total_propulsive_efficiency"] / model_param["reference_froede_number"] * model_param["spectral_average"] * model_param["surface_water_density"] * \
         model_param["waterline_width"] ** 2 * (model_param["acceleration_gravity"]/model_param["waterline_length"]**3)**0.05 * model_param["vessel_draught"] ** 0.62 * 0.25

    power_needed = coeff * (speed_tw ** 3) + coeff_wind_drag * (u_wind_rel ** 2 + v_wind_rel ** 2) * speed_tw + coeff_wave_drag * w_wave_hight ** 2 * speed_tw

    return power_needed


def power_to_fuel_burning_rate(
    power: npt.ArrayLike = None, efficiency: float = 0.5, fuel_value: float = 42.0e6
) -> npt.ArrayLike:
    """Convert power to fuel buring rate.
    Parameters
    ----------
    power: array
        Power in W (=kg*m2/s3).
    efficiency: float
        Fraction of fuel value turned into propulsive force. Defaults to 0.5.
    fuel_value: float
        Fuel value in J/kg. Defaults to 42.0e6.
    Returns
    -------
    array
        Fuel burning rate in kg/s.
    """
    fuel_burning_rate = power / efficiency / fuel_value
    return fuel_burning_rate

def power_to_fuel_consump(
    engine_power: npt.ArrayLike = None,
    steaming_time: npt.ArrayLike = None,
    distance: npt.ArrayLike = None,
) -> npt.ArrayLike:
    """Convert engine power to fuel consumed per vessel weight per distance and unit time.
    Parameters
    ----------
    engine_power: array
        engine power in kWh .
    steaming_time: array
        sailing time [hours] for vessel
    distance: array
        sailing distance in m
    Returns
    -------
    array
        Fuel consumption kW/kg/m.
    """
    fuel_consump = model_param["specific_fuel_consumption"] * engine_power * steaming_time/ model_param["DWT"] / distance
    return fuel_consump

def energy_efficiency_per_time_distance(
    fuel_consumption: npt.ArrayLike = None,
)-> npt.ArrayLike:
    """Convert engine power to fuel consumed per vessel weight per distance and unit time.
    Parameters
    ----------
    fuel_consumption: array
        fuel consumption kWh/kg/m
    Returns
    -------
    array
        energy efficiency indicator
    """
    energy_efficiency = model_param["conversion_factor_fuelmass2CO2"] * fuel_consumption
    return energy_efficiency

#def func_cost_function(ds_traj):
    # Cost function is equal to engine fuel consumption in this case
#    cost = ds_traj.fuel_consumption.sum("time_steps")
#    return cost
