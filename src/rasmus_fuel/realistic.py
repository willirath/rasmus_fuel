import numpy as np
import numpy.typing as npt


def power_maintain_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    u_wind: npt.ArrayLike = None,
    v_wind: npt.ArrayLike = None,
    w_wave_hight: npt.ArrayLike = None,
    vessel_waterline_width=30.0,
    vessel_waterline_length=210.0,
    vessel_total_propulsive_efficiency=0.7,
    vessel_draught=11.5,
    vessel_supersurface_area=345.0,
    vessel_water_drag_coefficient=6000.0,
    physics_air_mass_density=1.225,
    vessel_wind_resistance_coefficient=0.4,
    vessel_reference_froede_number=0.12,
    physics_spectral_average=0.5,
    physics_surface_water_density=1029.0,
    physics_acceleration_gravity=9.80665,
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
    vessel_supersurface_area: float
        area of the above water vessel structure exposed to wind [m ** 2]. Defaults to 345
    vessel_subsurface_area: float
        an average area of the lateral projection of underwater vessel structure [m ** 2]. Defaults to 245
    vessel_water_drag_coefficient: float
        water drag coefficient for vessel [kg/m]. Defaults to 6000
    vessel_waterline_width: float
        width of vessel at the waterline in [m]. Defaults to 30
    vessel_waterline_length: float
        length of vessel at the waterline in [m]. Defaults to 210
    vessel_total_propulsive_efficiency: float
        total propulsive engine efficiency. Defaults to 0.7
    vessel_draught: float
        vessel draught in [m]. Defaults to 11.5
    physics_air_mass_density: float
       mass density of air [kg/m**3]. Defaults to 1.225
    vessel_wind_resistance_coefficient: float
       wind resistance coefficent, typically in rages [0.4-1]. Defaults to 0.4
    physics_spectral_average: float
       spectral and angular dependency factor, dimensionless. Defaults to 0.5
    physics_surface_water_density: float
       density of surface water [kg/m**3]. Defaults to 1029
    physics_acceleration_gravity: float
       the Earth gravity accleration [m/s**2]. Defaults to 9.80665
        
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
        raise ValueError("Shape of u_ship_og and v_ship_og need to agree.")

    # calc speed through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current

    # calc engine power to maintain speed over ground using ocean current resistance term
    speed_tw = (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5
    course_ship_og = np.arctan2(v_ship_tw, u_ship_tw)

    coeff_water_drag = (
        0.5
        * physics_surface_water_density
        * vessel_water_drag_coefficient
        * vessel_supersurface_area
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
        * physics_air_mass_density
        * vessel_wind_resistance_coefficient
        * vessel_supersurface_area
    )

    coeff_wave_drag = (
        20.0
        * (vessel_waterline_width / vessel_waterline_length) ** (-1.2)
        * (1 / vessel_waterline_length) ** 0.62
        / vessel_total_propulsive_efficiency
        / vessel_reference_froede_number
        * physics_spectral_average
        * physics_surface_water_density
        * vessel_waterline_width ** 2
        * (physics_acceleration_gravity / vessel_waterline_length ** 3) ** 0.05
        * vessel_draught ** 0.62
        * 0.25
    )

    # TODO: Check if the referecnce systems used here are correct.
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
    vessel_specific_fuel_consumption=180.0,
    vessel_DWT=33434.0,
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
    vessel_specific_fuel_consumption: float
        specific fuel consumption  [g/kWh]. Defaults to 180
    vessel_DWT: float
        vessel dead weight in kg. Defaults to 33434

    Returns
    -------
    array
        Fuel consumption T/kg/m.
    """

    fuel_consump = (
        vessel_specific_fuel_consumption
        * engine_power
        * steaming_time
        / vessel_DWT
        / distance
    )

    return fuel_consump


def energy_efficiency_per_time_distance(
    fuel_consumption: npt.ArrayLike = None,
    vessel_conversion_factor_fuelmass2CO2=3.2060,
) -> npt.ArrayLike:
    """Convert engine power to fuel consumed per vessel weight per distance and unit time.
    
    Parameters
    ----------
    fuel_consumption: array
        fuel consumption kWh/kg/m
    vessel_conversion_factor_fuelmass2CO2: float
        conversion factor from fuel consumption to mass of CO2 emmitted (diesel/gas oil). Defaults to 3.2060
    
    Returns
    -------
    array
        energy efficiency indicator
    """

    energy_efficiency = vessel_conversion_factor_fuelmass2CO2 * fuel_consumption
    return energy_efficiency

