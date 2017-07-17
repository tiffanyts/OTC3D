
<img src="https://github.com/nenazarian/thermalcomfort/blob/master/Examples%20and%20Graphs/testfig.png" align="left" width="220" />


# Outdoor Thermal Comfort in 3D (OTC3D)
**[Description](#1.-description)**<br>
**[Motivations](#2.-motivations)**<br>
**[Installation of OTC3D](#3.-installation-of-OTC3D)**<br>
**[Running OTC3D](#4.-running-OTC3D)**<br>
**[License](#License)**<br>

## 1. Description 
**Outdoor Thermal Comfort in 3D** is a numerical model for calculating the *spatial variability* of outdoor thermal comfort (OTC) in urban areas. OTC is currently described as **Standard Effective Temperature**, which is a comprehensive thermal comfort metric that represents the human response to the thermal environment.

In order to comprehensively and accurately investigates urban microclimate, OTC3D employs a **modular approach**, such that the model can be used in combination with existing microclimate tools of urban flow and energy analysis with different levels of spatial and temporal resolutions. 

## 2. Motivations
1) Desribing outdoor thermal comfort with a **comrepehsive metric** that integrate air temperature and humidity, as well
as more complex factors such as solar radiation and wind speed, all interactting with the body’s thermal regulation processes.

2) Considering the **detailed spatial variability** of outdoor thermal comfort in urban areas, which is highly dependent on the urban form and radiative properties of urban areas. 

3) Accurately describing the **radiant exposure of human in outdoor spaces** by incorporating a) the visibility of urban surfaces to the pedestrians at any point, b) the spatial distribution of sky view factor, and c) inter-building shadowing and shortwave radiation effects on thermal comfort.

4) Streamlining and facilitating the **geometry implementation** by linking OTC3D with [Python Library for Urban Optimisation](https://github.com/chenkianwee/pyliburo). 

## 3. Installation of OTC3D
1)	Install Anaconda for python2.7. Instructions on how to install and use Anaconda [here](http://conda.pydata.org/docs/using/envs.html). 
Alternatively, you can insall Spyder 2.3.8 by following the steps given [here](https://pythonhosted.org/spyder/installation.html).
2) Download and insall the following libraries (execute these commands in Anaconda prompt): 
```
  * conda install -c https://conda.anaconda.org/dlr-sc pythonocc-core=0.17
  * conda install scipy 
 ``` 
The following libraries are automatically installed by running [thermalcomfort.py](https://github.com/tiffanyts/OTC3D/blob/master/ExtraFunctions.py)**

----------------
###### * Pvlib / pyliburo /  pythonocc (GNU LGPL3) / numpy (BSD 3-clause "New" or "Revised" License) / Pandas ("New" or "Revised" License) / scipy (BSD 3-clause "New" or "Revised" License) / matplotlib / pymf / cvxopt (GNU General Public License v3.0) / scikit-learn (BSD 3-clause "New" or "Revised" License) /  pyshp (mit license) / pycollada (BSD 3-clause "New" or "Revised" License) /networkx (BSD 3-clause "New" or "Revised" License) / lxml ((BSD) libxml2 and libxslt2 (MIT))
----------------
*The estimated time required for installation is 1hr (?)*
## 4. Running OTC3D 
1. Run thermalcomfort.py in the home directory of Python/Spyder. 
2. Run ExtraFunctions.py in the home directory of Python/Spyder. 
3. Execute the example code in the _same_ directory as thermalcomfort.py and ExtraFunctions.py. 
An example of running OTC3D for a) an idealized array of building, and b) a complex urban configuration is provided in the  ["Examples"](https://github.com/tiffanyts/OTC3D/tree/master/Examples) folder of this repository. 

### 4.1 INPUT/OUTPUT files 
The input files required to run the model are as follows: 
1. ped_properties.csv
2. model_inputs.csv
Examples of these files are given in the ["Input_Data"](https://github.com/tiffanyts/OTC3D/tree/master/Examples/Input_Data) directory. 

### 4.2 GEOMETRY specification:
#### A.	IDEALIZED Array of Buildings  
Run the file Building_IdealModel.py. The output is idealized set of buildings.
(i)	Input the file path of the input file.
<img src="https://github.com/nenazarian/thermalcomfort/blob/master/Examples%20and%20Graphs/Idealized.png" align="center" width="900" />

#### B. Realistic Urban Configuration (based on the OpenStreetMap)
<img src="https://github.com/nenazarian/thermalcomfort/blob/master/ComplexConfiguration.png" align="center" width="700" />

### 4.3 Explanation of Calculations
In this module, functions rely on a helper class **pdcoord** that standardize a pandas Dataframe as one with four columns consisting of _{x,y,z}_ coordinates and the corresponding value _v_. The pdcoord class is used to pass microclimate data and calculations between functions.

* **solar_param(*time,latitude,longitude, timezone, groundalbedo*)** uses PVLib to calculate solar parameters needed in the thermal comfort model. Returns a DataFrame of solar vector, solar view factor, direct solar radiation intensity, and diffuse solar radiation intensities from the sky and the ground

* **check_shadow(*key, model, solarvector*)** checks if location *key* is shaded by the *model* when solar radiation is incident at the \textit{solarvector} direction, and returns 0 if the location is shaded, and 1 if the location is sunlit.

* **skyviewfactor(*key, model*)** calculates SVF at the *key* location. This function is can be replaced by fourpiradiation, which combines the calculation with groundview and wall visibility.

* **fourpiradiation(*ped, model*)** constructs the discretized sphere of directions at the *key* location, and returns SVF, ground view factor, and a list of points of intercepts on wall surfaces visible to the pedestrian. 

* **call_values(*intercepts, surfpdcoord, gridsize*)** is given a list of intercepts, a pdcoord of surface values, and the grid size, and returns a list of values at the intercepts.

* **calc_radiation_from_values(*SurfTemp, SurfReflect, SurfEmissivity*)** returns the amount of long and shortwave radiation from urban surfaces by calculating the Stefan-Boltzmann law for the values returned by **call_values()**

* **calc_Esky_emis(*Ta,RH*)** calculates longwave radiation from the sky based on the ambient temperature $T_{a}$ and relative humidity $RH$.

* **meanradtemp(*Esky,Esurf, Eground,Ereflect, solarparam, SVF, GVF, pedestrian_albedo, shadow*)** calculates mean radiant temperature $T_{mrt}$ from different sources of radiation in the urban environment.

* **all\_ mrt(*key, compound, pdAirTemp, pdReflect, pdSurfTemp, solarparam,model_inputs, ped_constants*)** is given microclimate data, pedestrian information, and the urban model, and calculates the necessary components for **meanradtemp()** at location *key*. Returns $T_{mrt}$, longwave and shortwave radiation components, SVF and shading effects.

* **calc\_ SET(*microclimate, ped_constants, ped_properties*)** returns SET at one location with the given inputs:
       * *ped_properties* : Pedestrian properties, including height, skin wetness, mass, ratio of effective radiation area of the body (Fanger 1967), body emissivity, body albedo, metabolic rate, work activity, and clothing levels.
       * *microclimate* : Microclimate parameters, including air temperature, wind speed, mean radiant temperature, and relative humidity.
## 5. License
Currently no license is needed. However, following publications should be cited when using this model:
1. [Nazarian et al. (2017). Predicting outdoor thermal comfort in urban environments: A 3D numerical model for standard effective temperature. Urban Climate.]( https://www.researchgate.net/publication/316115262_Predicting_outdoor_thermal_comfort_in_urban_environments_A_3D_numerical_model_for_standard_effective_temperature)

2. [Chen et al. (2016). Workflow for Generating 3D Urban Models from Open City Data for Performance-Based Urban Design. In Proceedings of the Asim 2016 IBPSA Asia Conference, Jeju, Korea (pp. 27-29).](https://www.researchgate.net/publication/311534516_Workflow_for_Generating_3D_Urban_Models_from_Open_City_Data_for_Performance-Based_Urban_Design)



