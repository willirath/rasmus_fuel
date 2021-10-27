import numpy as np
import numpy.typing as npt

def froede_number_at_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    vessel_waterline_length=210.0,
    physics_acceleration_gravity=9.80665,
    **kwargs,
) ->npt.ArrayLike:
     """Calculate vessel Froede number from known speed over ground.

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
    vessel_waterline_length: float
        length of vessel at the waterline in [m]. Defaults to 210
    physics_acceleration_gravity: float
       the Earth gravity accleration [m/s**2]. Defaults to 9.80665

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        Froede number in [(m/s) ** (0.5)). Shape will be identical to
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
    froede_number = speed_tw/np.sqrt(physics_acceleration_gravity * vessel_waterline_length)
    return froede_number

def vessel_water_drag(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    vessel_total_propulsive_efficiency=0.7,
    vessel_subsurface_area=245.0,
    vessel_maximum_engine_power=14296344.0,
    vessel_speed_calm_water=9.2592,
    physics_surface_water_density=1029.0,
) ->npt.ArrayLike:
    """Calculate quadratic drag law power needed to maintain speed over ground.

    Parameters
    ----------
    u_ship_og: array
        Ship eastward speed over ground in meters per second.
    v_ship_og: array
        Ship northward speed over ground in meters per second.
        Needs same shape as u_ship_og.
    vessel_subsurface_area: float
        an average area of the lateral projection of underwater vessel structure [m ** 2]. Defaults to 245
    vessel_total_propulsive_efficiency: float
        total propulsive engine efficiency. Defaults to 0.7
    vessel_maximum_engine_power: float
        vessel maximu engine power in [W]. Defaults to 14296344.0,
    vessel_speed_calm_water: float
        vessel speed maximum in calm water [m/s]. Defaults 9.259    
    physics_surface_water_density: float
       density of surface water [kg/m**3]. Defaults to 1029

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        vessel water drag. Shape will be identical to
        u_ship_og and v_ship_og unless q = 0. For q=0 the water drag is a float number value
    """

    # ensure shapes of u_ship_og and v_ship_og agree
    if np.array(u_ship_og).shape != np.array(v_ship_og).shape:
        raise ValueError('Shape of u_ship_og and v_ship_og need to agree.')
    
    # calc speed over ground    
    speed_og = (u_ship_og ** 2 + v_ship_og ** 2) ** 0.5
    
    # Polynomial power coefficient
    q = 0.0 
    
    water_drag = (2 * vessel_total_propulsive_efficiency * vessel_maximum_engine_power / vessel_speed_calm_water ** 3 
                  / physics_surface_water_density / vessel_subsurface_area *
                  (speed_og/vessel_speed_calm_water) ** q )
    
    return water_drag

def power_maintain_sog(
    u_ship_og: npt.ArrayLike = None,
    v_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    u_wind: npt.ArrayLike = None,
    v_wind: npt.ArrayLike = None,
    w_wave_height: npt.ArrayLike = None,
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
    w_wave_height: array
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
        raise ValueError('Shape of u_ship_og and v_ship_og need to agree.')

    # calc speed through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current
    
    # calc speed over ground    
    speed_og = (u_ship_og ** 2 + v_ship_og ** 2) ** 0.5

    # calc engine power to maintain speed over ground using ocean current resistance term
    speed_tw = (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5

    coeff_water_drag = (
        0.5
        * physics_surface_water_density
        * vessel_water_drag_coefficient
        * vessel_subsurface_area
    )

    power_needed = coeff_water_drag * (speed_tw ** 3)

    # calc engine power to maintain speed over ground using : (1) ocean current resistance, (2) wind resistance, (3) wave resistance

    # calc relative wind speed
    speed_rel_to_wind = ((u_ship_og - u_wind) ** 2 + (v_ship_og - v_wind) ** 2) ** 0.5

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
        coeff_water_drag * (speed_tw ** 2) * speed_og
        + coeff_wind_drag * speed_rel_to_wind ** 2 * speed_og
        + coeff_wave_drag * w_wave_height ** 2 * speed_og
    )

    return power_needed

 def speed_og_from_power(
    course_ship_og: npt.ArrayLike = None,
    u_current: npt.ArrayLike = None,
    v_current: npt.ArrayLike = None,
    u_wind: npt.ArrayLike = None,
    v_wind: npt.ArrayLike = None,
    w_wave_height: npt.ArrayLike = None,
    engine_power: npt.ArrayLike = None,
    vessel_waterline_width=30.0,
    vessel_waterline_length=210.0,
    vessel_total_propulsive_efficiency=0.7,
    vessel_draught=11.5,
    vessel_supersurface_area=345.0,
    vessel_subsurface_area=245.0,
    vessel_water_drag_coefficient=6000.0,
    physics_air_mass_density=1.225,
    vessel_wind_resistance_coefficient=0.4,
    vessel_reference_froede_number=0.12,
    physics_spectral_average=0.5,
    physics_surface_water_density=1029.0,
    physics_acceleration_gravity=9.80665,
    **kwargs,
 )  ->npt.ArrayLike:
    """Calculate speed over ground from a given engine power.

    Parameters
    ----------
    course_ship_og: array
        Course ship over ground in rad (0 for North, pi/2 for East direction)
    speed_ship_init_og: array
        Initial guess for speed over ground (m/s)   
    u_current: array
        Ocean currents eastward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of course_ship_og.
    v_current: array
        Ocean currents northward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of course_ship_og.
    u_wind: array
        Eastward 10 m wind in m/s
        Needs shape that can be broadcst to shape of course_ship_og
    v_wind: array
        Northward 10 m wind in m/s
        Needs shape that can be broadcst to shape of course_ship_og
    w_wave_height: array
        Spectral significant wave height (Hm0), meters
        Needs shape that can be broadcst to shape of course_ship_og
    engine_power: array
        Engine power in W (=kg*m2/s3)
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
        Speed over ground in [m/s] that ship steams at a given engine power 
        Shape will be identical to course_ship_og.
    """

    # calc speed through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current
    
    # calc speed over ground    
    speed_og = (u_ship_og ** 2 + v_ship_og ** 2) ** 0.5

    # calc engine power to maintain speed over ground using ocean current resistance term
    speed_tw = (u_ship_tw ** 2 + v_ship_tw ** 2) ** 0.5

    coeff_water_drag = (
        0.5
        * physics_surface_water_density
        * vessel_water_drag_coefficient
        * vessel_subsurface_area
    )

    # calc relative wind speed
    speed_rel_to_wind = ((u_ship_og - u_wind) ** 2 + (v_ship_og - v_wind) ** 2) ** 0.5

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
  
    coeff3 = coeff_water_drag + coeff_wind_drag
    coeff2 = (- 2 * coeff_water_drag * (u_current * np.cos(course_ship_og + np.pi/2) + v_current * np.sin(course_ship_og + np.pi/2)) 
                  - 2 * coeff_wind_drag * (u_wind * np.cos(course_ship_og + np.pi/2) + v_wind * np.sin(course_ship_og + np.pi/2)))
    coeff1 =  (coeff_water_drag * (u_current ** 2  + v_current ** 2) + 
                   coeff_wind_drag * (u_wind ** 2  + v_wind ** 2) + coeff_wave_drag * w_wave_height ** 2)  
    coeff0 = - engine_power
 
    # Make variable transformation to reduced cubic equation and solve using Cardano's formula

    p = (3 * coeff3*coeff1 - coeff2 ** 2) / 3 / coeff3 ** 3
    q = (2 * coeff2 ** 3 - 9 * coeff3 * coeff2 * coeff1 + 27 * coeff3**2 * coeff0) / 27 / coeff3 **3

    t = (-q/2 + np.sqrt(q ** 2  / 4 + p ** 3 / 27)) ** (-1/3)+ (-q/2 - np.sqrt(q ** 2  / 4 + p ** 3 / 27)) ** (-1/3)

    speed_og = (t - coeff2 / 3 / coeff3).max()

    return speed_og

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
        vessel_specific_fuel_consumption * engine_power * steaming_time / vessel_DWT / distance
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
