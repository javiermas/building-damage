# Damage
Repository containing code for infering building damage from satellite imaging

# Installation

To get started write the following commands in your terminal:

```
git clone git@github.com:javiermas/building-damage.git
cd building-damage
pip install -r requirements.txt
pip install -e .
```

## Data architecture

The architecture is designed on three modules: data reading, data processing and data modelling. Following these modules, there are three folders inside of the python library. The folders _data_, _features_ and _modelling_ contain functions and classes with the responsibility of reading, processing and modelling data, respectively.

### Data processing
Data is processed using functions or _Transformer_ classes that get passed to the _Pipeline_ object. A Transformer class is a class that has a _transform_ method with a data argument. That argument receives a data object, which is a dictionary of this form:
{
    key_0: pandas.DataFrame(),
    ...
    key_N: pandas.DataFrame()
}

_Preprocessor_ classes receive that data dictionary and return it with some modifications. Note that _Pipeline_ overwrites the data object when iterating over preprocessors.
_Feature_ classes receive that dictionary and return a pandas.DataFrame() indexed by 'city', 'patch_id' and 'date'.Note that _Pipeline_ adds a new key to the data dictionary with the feature name that was passed. Once the _Pipeline_ object has finished iterating over preprocessors and features, it merges the dataframes that came out of the feature classes, making use of their indices. Remember that features need to return a pandas.DataFrame that is indexed by 'city', 'patch_id' and 'date'. 

## Data
### Raqqa


__Annotations__

-------- 22/10/2013 -- 12/02/2014 -------- 29/05/2015

__Rasters__

-- 17/01/2013 ---------- 21/03/2014 --- 02/02/2015 --- 01/07/2016


### Daraa

__Annotations__

------------ 09/07/2013 ------- 01/05/2014 -- 04/06/2015 --------- 19/04/2016

__Rasters__

-- 17/10/2011 -- 10/11/2013 ---- 01/05/2014 --------- 25/02/2016 -- 19/04/2016 - 07/02/2017

__The coordinates problem__

Some .tiff files seem to have either the latitude or longitude constant across the entire image. This renders them useless because we can't link them to the annotation, where the damage levels are associated to a specific coordinate. See the following notebook for the corresponding analysis:

https://github.com/javiermas/building-damage/blob/master/notebooks/jma_02_coordinate_problems.ipynb
