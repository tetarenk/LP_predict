# LP_predict
Python script that simulates JCMT large program observing.

## General Description
The script combines the `astroplan` package, with the large program MSBs, and past WVM data, to create an observing schedule for each night allocated to large programs over the upcoming semester(s). Additional overheads such as, calibrations (equivalent to 25\% of the target observing time), time to slew between targets, faults (set rate of 3% per night), blocked out dates for E\&C, and extended observing (occurs when we are in weather better than Band 3), are also included in the observing schedules to more closely match a real observing night at JCMT. 

The scheduler mimics the JCMT's flexible observing guidelines by stepping through the night sequentially, scheduling the highest priority MSBs that are observable. 
A target is only added to the target list for a night if it meets the following requirements: target has MSB repeats remaining, the night is not in the blackout dates for the instrument, the weather is appropriate for the MSB. 
Priority is set as follows: target is from the program for the current observing block (1), target has the same weather band as the night (2), target is allocated time for a worse weather band (3).

Script is compatible with both Python 2 and 3.


## Requires
* `astroplan`
* `numpy`
* `matplotlib`
* `astropy`
* `datetime`
* `pandas`

## To run the simulator script you need:

### On an EAO computer: 

* Start/End dates for simulation.
* Blocked out dates for each instrument (e.g., for E\&C).
* Large program observing blocks file (columns: start date, end date, program with priority, optional second program with priority).

### On any machine, you will additionally need: 

* Large program details file and MSB files for each program (through running the sql scripts example-project-summary.sql and  example-project-info.sql on an EAO machine).
* WVM data file (through running the partner python script, getwvm.py, on an EAO machine).

To run the sql scripts on an EAO machine and save query output to a file,

`ompsql < example-script.sql > output_file.txt`

To run the wvm script on an EAO machine, input the date range in the script, then,

`python wvm.py`

## Simulator output:

* Simulator results file: including allocated/observed/remaining hrs for each program, total observed hrs in simulation, total hrs lost to weather in simulation, program finish dates.
* File tabulating a breakdown of available, used, and unused hours in each weather band during the simulation.
* Schedules for each night in the simulation; table files and plots (shown below).

<img src="docs/201917.png" width="750" height="500" title='Example Schedule of a Band 2 Night'>

*  Simulator results plots: including program results bar chart (shown below), program completion chart (shown below), time breakdown bar chart (shown below), unused LST histograms per weather band (shown below), LST histograms per weather band for remaining MSBs.

<img src="docs/prog_results.png" width="583" height="350" title='Program Results'>
<img src="docs/prog_completion.png" width="573" height="380" title='Program Completion'>
<img src="docs/unused_bar.png" width="446" height="390" title='Time Breakdown'>
<img src="docs/unused_RA.png" width="430" height="400" title='Unused LST'>
