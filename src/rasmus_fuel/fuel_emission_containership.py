import numpy as np
import numpy.typing as npt

def fuel_consumption_diesel_MANandBW(
    speed_ship_og: npt.ArrayLike =  None,
    engine_power: npt.ArrayLike = None,
    vessel_design_speed = 12.25,    
    vessel_specific_fuel_consumption = 0.000165,
    vessel_number_engines_operational = 1.0,
    vessel_load_per_active_engine = 0.6,
    vessel_maximum_continuous_rating = 0.85,
    **kwargs,
) ->npt.ArrayLike:
    """Calculate vessel fuel consumption using diesel engine MANB&W characteristics

    Parameters
    ----------
    speed_ship_og: array
        Ship speed over ground in meters per second.
    engine_power: array
        Engine power in [W]. Shape is identical to speed_ship_og.
    vessel_design_speed: float
        Maximum speed overground or design speed of vessel in [m/s]. Defaults to 12.25    
    vessel_specific_fuel_consumption: float
        specific fuel consumption [kg/Wh]. Default = 0.000165
    vessel_number_engines_operational: float
        number of available active engines. Default = 1.0  
    vessel_load_per_active_engine:
        average load per active engine. Default = 0.6    
    vessel_maximum_continuous_rating: float
        maximum continuous rating of engine power . Default = 0.85

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        fuel consumption in [kg/s]. Shape will be identical to
        speed_ship_og. 
    """
    correction_reduced_speed_factor = (
        (speed_ship_og / vessel_design_speed) ** 3 + 0.2
    ) / 1.2

    assumed_maximum_continuous_rating = 0.75

    vessel_number_active_engines = np.min(
        vessel_number_engines_operational,
        np.round((correction_reduced_speed_factor * 
        vessel_number_engines_operational * 
        assumed_maximum_continuous_rating) + 1)
        )
    
    fraction_max_continuous_rating = vessel_number_engines_operational / 
        * correction_reduced_speed_factor 
        * vessel_maximum_continuous_rating

    if vessel_number_engines_operational == 1:
        fuel_consumption = (
            correction_reduced_speed_factor 
            * vessel_load_per_active_engine
            * vessel_maximum_continuous_rating 
            * engine_power  
            * vessel_specific_fuel_consumption 
            / 3600
        )
    else:
        fuel_consumption = (
            correction_reduced_speed_factor 
            * vessel_load_per_active_engine
            * vessel_number_active_engines 
            * fraction_max_continuous_rating 
            * engine_power  
            * vessel_specific_fuel_consumption 
            / 3600
        )
    return fuel_consumption    

def convert_fuel_consumption_tonnsperday(
    fuel_consumption: npt.ArrayLike = None,
    conversion_factor = 86400000,
    **kwargs,
) ->npt.ArrayLike:
    """Convert vessel fuel consumption from SI units kg/s to tonns/day
    
    Parameters
    ----------
    fuel_consumption: array
       fuel consumption in [kg/s]      
    conversion_factor: float
       conversion of fuel from kg/s to tonns/day. Defaults to 86400000
       
    Returns
    --------
    array:
        fuel consumption in [tonns/day]. Shape will be identical to fuel_consumption
    """
    fuel_tonnsperday = conversion_factor * fuel_consumption 
    return fuel_tonnsperday

def convert_emission_kgpermeter_kgperNM(
    emission: npt.ArrayLike =None,
    conversion_factor = 1852,
    **kwargs,
) ->npt.ArrayLike:
    """Convert vessel emission from SI units kg/m to kg/NM
    
    Parameters
    ----------
    emission: array
       emission in [kg/m]      
    conversion_factor: float
       conversion of fuel from kg/m to kg/NM. Defaults to 1852
       
    Returns
    --------
    array:
        emission_kgperNM in [kg/NM]. Shape will be identical to emission
    """
    emission_kgperNM = conversion_factor * emission 
    return emission_kgperNM

def emission_CO2_diesel_MANandBW(
    fuel_consumption: npt.ArrayLike = None,
    sailing_time: npt.ArrayLike = None,
    vessel_number_active_engines = 1.0,
    vessel_conversion_factor_fuel_toCO2 = 3.200, 
    **kwargs,
) ->npt.ArrayLike: 
    """Calculate CO_2 emission for vessel with diesel engine type MAN-B&W.

    Parameters
    ----------
    fuel_consumption: array
        fuel consumption in [kg/s]     
    sailing_time: array
        vessel sailing time in seconds.   
        Shape is identical to fuel_consumption.  
    vessel_number_active_engines: float
        Number of active engines for vessel. Default = 1 
    vessel_conversion_factor_fuel_toCO2: float
        conversion factor for diesel type engine from fuel to CO_2 mass [kg fuel/ kg CO_2]. 
        Defaults to 3.206

    All other keyword arguments will be ignored.

    Returns
    -------
    array:
        total emission in [kg]. Shape will be identical to
        fuel_consumption. 
    """

    total_emission = (
       vessel_conversion_factor_fuel_toCO2 
       * fuel_consumption 
       * sailing_time 
    )
    return total_emission    