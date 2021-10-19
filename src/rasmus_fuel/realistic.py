import numpy as np
import numpy.typing as npt

VESSEL_PARAM = {
    # """Define vessel parameters
    #   Parameters
    # --------------
    #   vessel_supersurface_area: float
    #       area of the above water vessel structure exposed to wind [m ** 2]. Defaults to 345
    #   vessel_subsurface_area: float
    #       an average area of the lateral projection of underwater vessel structure [m ** 2]. Defaults to 245
    #   water_drag_coefficient: float
    #       water drag coefficient for vessel [kg/m]. Defaults to 6000
    #   waterline_width: float
    #       width of vessel at the waterline in [m]. Defaults to 30
    #   waterline_length: float
    #       length of vessel at the waterline in [m]. Defaults to 210
    #   total_propulsive_efficiency: float
    #       total propulsive engine efficiency. Defaults to 0.7
    #   vessel_draught: float
    #       vessel draught in [m]. Defaults to 11.5
    #   specific_fuel_consumption: float
    #       specific fuel consumption  [g/kWh]. Defaults to 180
    #   DWT: float
    #       vessel dead weight in kg. Defaults to 33434
    #   conversion_factor_fuelmass2CO2: float
    #       conversion factor from fuel consumption to mass of CO2 emmitted (diesel/gas oil). Defaults to 3.2060
    # """
    "waterline_width": 30.0,
    "waterline_length": 210.0,
    "total_propulsive_efficiency": 0.7,
    "vessel_draught": 11.5,
    "vessel_supersurface_area": 345.0,
    "vessel_subsurface_area": 245.0,
    "water_drag_coefficient": 6000.0,
    "specific_fuel_consumption": 180.0,
    "DWT": 33434.0,
}

PHYSICS_PARAM = {
    # """Define physical model parameters
    #     Parameters
    # --------------
    #     air_mass_density: float
    #        mass density of air [kg/m**3]. Defaults to 1.225
    #     wind_resistance_coefficient: float
    #        wind resistance coefficent, typically in rages [0.4-1]. Defaults to 0.4
    #     reference_froede_number: float
    #        Froede number, dimensionless. Defaults to 0.12
    #     spectral_average: float
    #        spectral and angular dependency factor, dimensionless. Defaults to 0.5
    #     surface_water_density: float
    #        density of surface water [kg/m**3]. Defaults to 1029
    #     acceleration_gravity: float
    #        the Earth gravity accleration [m/s**2]. Defaults to 9.80665
    #     """
    "air_mass_density": 1.225,
    "wind_resistance_coefficient": 0.4,
    "reference_froede_number": 0.12,
    "spectral_average": 0.5,
    "surface_water_density": 1029.0,
    "acceleration_gravity": 9.80665,
}


def power_maintain_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    course_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    u_wind: npt.ArrayLike = None,
    v_wind: npt.ArrayLike = None,
    w_wave_hight: npt.ArrayLike = None,
    physics_param: dict = None,
    vessel_param: dict = None,
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
        
    All other keyword arguments will be ignored.
    
    Returns
    -------
    array:
        Power in W (=kg*m2/s3) needed to maintain speed over ground for each
        element of u_ship_og and v_ship_og. Shape will be identical to
        u_ship_og and v_ship_og.
    """

    #  assign physical and vessel parameters
    if physics_param is None:
        physics_param = PHYSICS_PARAM
    if vessel_param is None:
        vessel_param = VESSEL_PARAM
    # ensure shapes of u_ship_og and v_ship_og agree
    if np.array(u_ship_og).shape != np.array(v_ship_og).shape:
        raise ValueError("Shape of u_ship_og and v_ship_og need to agree.")

    # calc speed through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current

    # calc engine power to maintain speed over ground using ocean current resistance term
    speed_tw = (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5

    coeff_water_drag = (
        0.5
        * physics_param["surface_water_density"]
        * vessel_param["water_drag_coefficient"]
        * vessel_param["vessel_supersurface_area"]
    )

    power_needed = coeff_water_drag * (speed_tw ** 3)

    # calc engine power to maintain speed over ground using : (1) ocean current resistance, (2) wind resistance, (3) wave resistance

    # calc relative wind speed
    wind_speed = (u_wind ** 2 + v_wind ** 2) ** 0.5
    wind_dir = np.arctan2(v_wind, u_wind)
    wind_dir_rel = wind_dir - course_ship_og + np.pi

    u_wind_rel = wind_speed * np.cos(wind_dir_rel) - u_ship_og
    v_wind_rel = wind_speed * np.sin(wind_dir_rel) - v_ship_og

    coeff_wind_drag = (
        0.5
        * physics_param["air_mass_density"]
        * physics_param["wind_resistance_coefficient"]
        * vessel_param["vessel_supersurface_area"]
    )

    coeff_wave_drag = (
        20.0
        * (vessel_param["waterline_width"] / vessel_param["waterline_length"]) ** (-1.2)
        * (1 / vessel_param["waterline_length"]) ** 0.62
        / vessel_param["total_propulsive_efficiency"]
        / physics_param["reference_froede_number"]
        * physics_param["spectral_average"]
        * physics_param["surface_water_density"]
        * vessel_param["waterline_width"] ** 2
        * (
            physics_param["acceleration_gravity"]
            / vessel_param["waterline_length"] ** 3
        )
        ** 0.05
        * vessel_param["vessel_draught"] ** 0.62
        * 0.25
    )

    power_needed = (
        coeff_water_drag * (speed_tw ** 3)
        + coeff_wind_drag * (u_wind_rel ** 2 + v_wind_rel ** 2) * speed_tw
        + coeff_wave_drag * w_wave_hight ** 2 * speed_tw
    )

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
    vessel_param: dict = None,
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

    # assign vessel parameters
    if vessel_param is None:
        vessel_param = VESSEL_PARAM
    fuel_consump = (
        vessel_param["specific_fuel_consumption"]
        * engine_power
        * steaming_time
        / vessel_param["DWT"]
        / distance
    )
    return fuel_consump


def energy_efficiency_per_time_distance(
    fuel_consumption: npt.ArrayLike = None, vessel_param: dict = None
) -> npt.ArrayLike:
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

    # assign vessel parameters
    if vessel_param is None:
        vessel_param = VESSEL_PARAM
    energy_efficiency = (
        vessel_param["conversion_factor_fuelmass2CO2"] * fuel_consumption
    )
    return energy_efficiency

