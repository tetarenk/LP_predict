# LP_predict
Python script that simulates JCMT large program observing.

## General Description
The script combines the `astroplan` package, with the large program MSBs, and past WVM data, to create an observing schedule for each night allocated to large programs over the upcoming semester(s). Additional overheads such as, calibrations (equivalent to 25\% of the target observing time), time to slew between targets, blocked out dates for E\&C, and extended observing (occurs when we are in weather better than Band 3), are also included in the observing schedules to more closely match a real observing night at JCMT. 

The scheduler mimics the JCMT's flexible observing guidelines by stepping through the night sequentially, scheduling the highest priority MSBs that are observable. 
A target is only added to the target list for a night if it meets the following requirements: target has MSB repeats remaining, the night is not in the blackout dates for the instrument, the weather is appropriate for the MSB. 
Priority is set as follows: target is from the program for the current observing block (1), target has the same weather band as the night (2), target is allocated time for a worse weather band (3).


## Requires
* `astroplan`
* `numpy`
* `matplotlib`
* `astropy`
* `datetime`
* `pandas`

## To run the simulator script you need:

*  Start/End dates for simulation.
* Blocked out dates for each instrument (e.g., for E\&C).
* Large program details file (columns: project ID, priority, remaining hrs, allocated hrs, taumin, taumax).
* MSB files for each program (columns: project ID, msb id, remaining count, obs count, msb time (s), total msb hrs, instrument, type, polarization, target, ra (radians), dec (radians), taumin, taumax).
* Large program observing blocks file (columns: start date, end date, program with priority, optional second program with priority).

## Simulator output:

* Simulator results file

```
projectid allocted_hrs remaining_hrs sim_obs_hrs remaining_aftersim
M16AL001 200.0 73.015 72.74 0.0
M16AL005 611.35 50.0 50.0 0.0
M16AL006 330.0 124.25 124.0 0.0
M17BL005 276.3 167.8 167.99 0.0
M17BL011 224.0 158.2 158.53 0.0
M17BL009 319.0 287.85 212.87 74.98
M17BL004 403.9 170.1 171.23 0.0
M17BL002 515.0 265.3 265.31 0.0
M17BL001 873.0 757.3 757.84 0.0
M17BL010 140.0 140.0 139.58 0.0
M17BL006 305.2 273.3 272.93 0.0
M17BL007 400.0 350.35 349.81 0.54

Total Allocated Hrs for Large Programs: 4597.75
Total Remaining Hrs for Large Programs before Simulation: 2817.47
Total Hrs Available for Large Program Observing in Simulation:13592.0
Total Observed Hrs for Large Programs in Simulation: 2742.83
Total Remaining Hrs for Large Programs after Simulation: 75.52
Total Hrs lost to weather (i.e., nights where nothing observed):7645.0

M16AL001 Tally:
OMC2-3
(1,2019),(2,2019),(3,2019),(4,2019),(6,2019),(7,2019),(8,2019),(9,2019),(10,2019),
(11,2019),(12,2019),(1,2020),(4,2020),(6,2020)
Serpens South
(1,2019),(2,2019),(3,2019),(4,2019),(5,2019),(6,2019),(7,2019),(8,2019),(9,2019),
(10,2019),(11,2019),(12,2019),(1,2020),(4,2020),(5,2020),(6,2020)
Serpens Main
(1,2019),(2,2019),(3,2019)
Oph Core
(1,2019),(2,2019),(3,2019),(4,2019),(5,2019),(6,2019),(7,2019),(8,2019),(9,2019),
(11,2019),(12,2019),(1,2020),(4,2020),(5,2020),(6,2020),(7,2020),(9,2020),(10,2020)
IC348
(1,2019),(2,2019),(4,2019),(5,2019),(6,2019),(7,2019),(8,2019),(9,2019),(10,2019),
(11,2019),(12,2019),(1,2020),(4,2020),(5,2020)
NGC2024
(1,2019),(2,2019),(3,2019),(4,2019),(6,2019),(7,2019),(8,2019),(9,2019),(10,2019),
(11,2019),(12,2019),(1,2020),(4,2020),(6,2020)
NGC 2071
(1,2019),(2,2019),(3,2019),(4,2019),(6,2019),(7,2019),(8,2019),(9,2019),(10,2019),
(11,2019),(12,2019),(1,2020),(4,2020)
NGC 1333
(1,2019),(2,2019),(3,2019),(4,2019),(5,2019),(6,2019),(7,2019),(8,2019),(9,2019),
(10,2019),(11,2019),(12,2019),(1,2020)

Program Finish Dates:

M16AL001:2020-10-01
M17BL001:2021-08-26
M17BL002:2020-04-18
M17BL005:2020-01-09
M17BL004:2020-04-29
M17BL007:2020-07-30
M16AL006:2020-01-09
M17BL009:not finished
M17BL006:2020-10-26
M17BL010:2020-02-18
M17BL011:2019-12-27
M16AL005:2019-06-08
```


