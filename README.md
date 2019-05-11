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
