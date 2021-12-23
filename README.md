# Fuel and emission model for large vessels

[![Tests](https://github.com/willirath/rasmus_fuel/workflows/test/badge.svg)](https://github.com/willirath/rasmus_fuel/actions?query=workflow%3Atest)
[![Coverage](https://codecov.io/gh/willirath/rasmus_fuel/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/github/willirath/rasmus_fuel?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/willirath/rasmus_fuel/master?filepath=doc%2Fexamples) 

This package allows for estimating fuel consumption (taking into account ocean currents, waves and wind) for vessels.
It also allows to derive estimates of CO_2 emissions.

## Model parameters

In the model the convension of SI units for all parameters is implemented. Input parameters for the model are described in the table below. 

| Parameter | Description , Units | 
| --- | --- |
| Vessel design speed | maxium speed vessel reach at calm conditions [m/s] |
| Specific fuel consumption (SFOC)| specific fuel consumption, [kg/Wh]  |
| Number of operational engines | Vessel number of operational engines |
| Load per active engine | the torque output of an engine |
| Maximum continuous rating (MCR)| maximum power output of engine|
| Conversion factor of fuel to CO_2 | conversion factor from fuel to CO_2 mass [kg of fuel / kg CO_2]| 

## Emission model description

For a single operational engine and auxiliary engine power, the total fuel consumption is estimated according to (see [1]):

$Fuel\_Consumption = (CRScor \cdot MCR \cdot Engine\_Power \cdot Load\_Factor + FCaux) \cdot SFOC / 3600$     

where $CRScor$ is a correction reduced speed factor evaluated as given:

$CRScor = ((V\_{actual}/V\_{design}) ^ 3 + 0.2) / 1.2,$

$V\_{actual}$ and $V\_{design}$ are vessel speeds operational and design. 

The term $FCaux$ is an auxiliary engine power estimated according to relationship:

$FCaux = Auxiliary\_Engine_Power \cdot Load\_Factor$,

for simplicity in the model $FCaux = 0.63 \cdot Engine\_Power$

For a case when a number of operational engines of vessel > 1 the formula for a fuel consumption recasts to [1]:

$Fuel\_Consumption = (CRScor \cdot Number\_Active\_Engines \cdot fMCR \cdot Engine\_Power \cdot Load\_Factor + FCaux) \cdot SFOC / 3600$ 

where $fMCR$ is a fraction of maximum continuous rating which is equal to:

$fMCR = Number\_Operational\_Engines / Number\_Active_Engines \cdot CRScor \cdot MCR$

To estimate the number of active engines, the following relationship is used:

$Number\_Active\_Engines = {\rm min}(Engines\_Operational, {\rm ceil}(CRScor \cdot Engines\_Operational \cdot MCR))$

As for total fuel-based CO_2 emission per a single voyage the formula is used: 

$Emission = Conversion\_Factor\_Fuel\_toCO2 \cdot Fuel\_Consumption \cdot Sailing\_Time,$   

where $Sailing\_Time$ is a single voyage time.

## Literature

[1] [D.R. Schouten, T.W.F. Hasselaar Ship emision model validation with noon reports, 2018 Marin Report No. 30799-1-TM (v3)](http://www.emissieregistratie.nl/erpubliek/documenten/Lucht%20(Air)/Verkeer%20en%20Vervoer%20(Transport)/Zeescheepvaart/MARIN%20(2018)%20Ship%20Emission%20Validation%20with%20Noon%20reports_V3.pdf)

## Install

...

## Documentation

...

## License

MIT License, see LICENSE file.

