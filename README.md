# Fuel - based emission model for container ship

[![Tests](https://github.com/willirath/rasmus_fuel/workflows/test/badge.svg)](https://github.com/willirath/rasmus_fuel/actions?query=workflow%3Atest)
[![Coverage](https://codecov.io/gh/willirath/rasmus_fuel/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/github/willirath/rasmus_fuel?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/willirath/rasmus_fuel/master?filepath=doc%2Fexamples)

The model estimates at first step a fuel consumption and at seconds step a fuel-based CO_2 emission for a slow-speed (<400 rotation per minute (RPM)) container ships with a diesel engine 
that runs on marine diesel oil (MDO). Total engine fuel consumption and emission are calculated for diesel engine type MAN-B&W with parameters described below. 

### Model parameters

In the model the convension of SI units for all parameters is implemented. Input parameters for the model are described in the table below. 

| Parameter | Description , Units | 
| --- | --- |
| Vessel design speed | maxium speed vessel reach at calm conditions [m/s] |
| Specific fuel consumption (SFOC)| specific fuel consumption, [kg/Wh]  |
| Number of operational engines | Vessel number of operational engines |
| Load per active engine | the torque output of an engine |
| Maximum continuous rating (MCR)| maximum power output of engine|
| Conversion factor of fuel to CO_2 | conversion factor from fuel to CO_2 mass [kg of fuel / kg CO_2]| 



### Emission model description

Depending on number of oepartional engines two alternative formulas for fuel consumption is implemented.
To estimate a total fuel consumption with a single operational engine and auxiliary engine power the formulas is used [1]:

$Fuel_Consumption = (CRScor * MCR * Engine_Power  * Load_Factor + FCaux)* SFOC * /3600$     

where $CRScor$ is a correction reduced speed factor evaluated as given:

$CRScor = ((V_{actual}/V_{design}) ^ 3 + 0.2) / 1.2,$

$V_{actual}$  and $V_{design}$ are vessel speeds operational and design. 

The term $FCaux$ is an auxiliary engine power estimated according to relationship:

$FCaux = Auxiliary_Engine_Power * Load_Factor$,

for simplicity in the model $FCaux = 0.63 * Engine_Power$

For a case when a number of operational engines of vessel > 1 the formula for a fuel consumption recasts to [1]:

$Fuel_Consumption = (CRScor * Number_Active_Engines * fMCR * Engine_Power  * Load_Factor + FCaux)* SFOC * /3600$ 

where $fMCR$ is a fraction of maximum continuous rating which is equal to:

$fMCR = Number_Operational_Engines / Number_Active_Engines * CRScor * MCR$

To calculated a number of active engines the relationship is used:

$Number_Active_Engines = min(Engines_Operational, round(CRScor * Engines_Operational * MCR)+1)$

As for total fuel-based CO_2 emission per a single voyage the formula is used: 

$Emission = Conversion_Factor_Fuel_toCO2 * Fuel_Consumption * Sailing_Time,$   

where $Sailing_Time$ is a single voyage time.

## Literature

[1] [D.R. Schouten, T.W.F. Hasselaar Ship emision model validation with noon reports, 2018 Marin Report No. 30799-1-TM (v3)] (http://www.emissieregistratie.nl/erpubliek/documenten/Lucht%20(Air)/Verkeer%20en%20Vervoer%20(Transport)/Zeescheepvaart/MARIN%20(2018)%20Ship%20Emission%20Validation%20with%20Noon%20reports_V3.pdf)

## Install

...

## Documentation

...

## License

MIT License, see LICENSE file.

