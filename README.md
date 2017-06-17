<img src="https://github.com/nenazarian/thermalcomfort/blob/master/Examples%20and%20Graphs/testfig.png" align="right" width="150" />

# Outdoor Thermal Comfort in 3D (OTC3D) 
## Description 
**Outdoor Thermal Comfort in 3D** is a numerical model for calculating the *spatial variability* of outdoor thermal comfort (OTC) in urban areas. OTC is currently described as *Standard Effective Temperature*, which is a comprehensive thermal comfort metric that represents the human response to the thermal environment.

In order to comprehensively and accurately investigates urban microclimate, OTC3D employs a modular approach, such that the model can be used in combination with existing microclimate tools of urban flow and energy analysis. 

## Motivations
1) Desribing outdoor thermal comfort with comrepehsive metrics that integrate air temperature and humidity, as well
as more complex factors such as solar radiation and wind speed, all interactting with the body’s thermal regulation processes.

2) Considering the detailed spatial variability of outdoor thermal comfort in urban areas, which is highly dependent on the urban form and radiative properties of urban areas. 

3) Describing the human radiant exposure accurately by incorporating a) the visibility of urban surfaces to the pedestrians at
any point, b) the spatial distribution of sky view factor, and c) inter-building shadowing and shortwave radiation effects on thermal comfort.

4) Streamlining and facilitating the geometry implementation by linking OTC3D with [Python Library for Urban Optimisation](https://github.com/chenkianwee/pyliburo). 

## Installation 
1)	Install Anaconda for python2.7. Instructions on how to install and use Anaconda [here](http://conda.pydata.org/docs/using/envs.html). 
Alternatively, you can insall Spyder 2.3.8 by following the steps given [here](https://pythonhosted.org/spyder/installation.html).
2)	The following libraries are automatically installed by running [thermalcomfort.py]()

## License
Currently no license is needed. However, following publications should be cited when using this model:
1. [Nazarian et al. (2017). Predicting outdoor thermal comfort in urban environments: A 3D numerical model for standard effective temperature. Urban Climate.]( https://www.researchgate.net/publication/316115262_Predicting_outdoor_thermal_comfort_in_urban_environments_A_3D_numerical_model_for_standard_effective_temperature)

2. [Chen et al. (2016). Workflow for Generating 3D Urban Models from Open City Data for Performance-Based Urban Design. In Proceedings of the Asim 2016 IBPSA Asia Conference, Jeju, Korea (pp. 27-29).](https://www.researchgate.net/publication/311534516_Workflow_for_Generating_3D_Urban_Models_from_Open_City_Data_for_Performance-Based_Urban_Design)
