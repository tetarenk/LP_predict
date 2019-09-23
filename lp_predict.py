########################################
#JCMT Large Program Prediction Tool
########################################
'''Python script that simulates JCMT Large Program observing over the upcoming semester(s).

INPUT: (1) sim_start: proposed start date 
       (2) sim_end: proposed end date
       (3) Blocked out dates for each instrument
       (4) Observing blocks for large programs file (with start/end dates of each scheduled program block)
       (optional) Calibrator lists for SCUBA-2, HARP, and Pointing Cals

OUTPUT: (1) File summary of simulation results detailing the predicted number of hours observed/remaining for each project
        (2) File summary of available, used, and unused time in each weather band
        (3a) Histograms displaying unused RA range per weather band
        (3b) Histograms displaying remaining MSB RA range per weather band
        (4) Bar plot of totals (observed/remaining hrs) per program
        (5) Bar plot of totals (used/unused/cals hrs) per weather band
        (6) Incremntal program completion chart

NOTES: - This script is meant to be run on an EAO computer. If you want to run this script on any machine you need to generate the
following on an EAO machine first:
(a) wvm file through the python script provided (getwvm.py).
(b) LAP projects file and MSB files through the sql scripts provided (example-project-summary.sql and example-project-info.sql)
Details are provided below in the user input and SQL queries sections of the script.
- Uses the following python packages: astropy, astroplan, matplotlib, numpy, datetime, pandas, mysql-connector-python.
-Works in both Python 2 and 3.

Written by: Alex J. Tetarenko
Last Updated: Sep 23, 2019
'''

#packages to import
from __future__ import print_function
import numpy as np
import math as ma
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm
import matplotlib.dates as mdates
import datetime as datetime
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time,TimeDelta
import warnings
warnings.filterwarnings('ignore')
from astroplan import Observer, FixedTarget, AltitudeConstraint, is_observable, ObservingBlock, observability_table
from astroplan.constraints import TimeConstraint,AltitudeConstraint
from astroplan.scheduling import PriorityScheduler, Schedule, Transitioner,Scheduler, Scorer,TransitionBlock
from astroplan.utils import time_grid_from_range,stride_array
from astroplan.plots import plot_schedule_airmass
import os
from collections import defaultdict
from getwvm import get_wvm_fromdisk, get_sampled_values
import time
import sys
#omp-python import
if sys.version_info.major==2:
	sys.path.append('/jac_sw/omp/python/lib/')
else:
	#Sarah's python 3 version currently lives here, path will be changed at some point
	sys.path.append('/net/kapili/export/data/sgraves/software/omp-python/lib')
from omp.db.part.arc import ArcDB



def sort_blocks(blocks):
	''' Sort observing blocks in priority order.'''
	bs=[]
	for b in blocks:
		if type(b.priority) is int or type(b.priority) is float:
			bs.append(b.priority)
		else:
			bs.append(b.priority[0])
	sorted_indxs=np.argsort(bs)
	blocks_sorted=[]
	for i in sorted_indxs:
		blocks_sorted.append(blocks[i])
	return(blocks_sorted)

class JCMTScheduler(Scheduler):
    '''A scheduler that does simple sequential scheduling.  That is, it starts at the beginning of the night,
    looks at all the blocks (sorted in priority order), picks the best one, schedules it, and then moves on.
    NOTE: This is just the astroplan SequentialScheduler with the addition of sorted observing blocks as input.'''

    def __init__(self, *args, **kwargs):
        super(JCMTScheduler, self).__init__(*args, **kwargs)

    def _make_schedule(self, blocks):
        blocks=sort_blocks(blocks)
        pre_filled = np.array([[block.start_time, block.end_time] for
                               block in self.schedule.scheduled_blocks])
        if len(pre_filled) == 0:
            a = self.schedule.start_time
            filled_times = Time([a - 1*u.hour, a - 1*u.hour,
                                 a - 1*u.minute, a - 1*u.minute])
            pre_filled = filled_times.reshape((2, 2))
        else:
            filled_times = Time(pre_filled.flatten())
            pre_filled = filled_times.reshape((int(len(filled_times)/2), 2))
        for b in blocks:
            if b.constraints is None:
                b._all_constraints = self.constraints
            else:
                b._all_constraints = self.constraints + b.constraints
            # to make sure the scheduler has some constraint to work off of
            # and to prevent scheduling of targets below the horizon
            # TODO : change default constraints to [] and switch to append
            if b._all_constraints is None:
                b._all_constraints = [AltitudeConstraint(min=0 * u.deg)]
                b.constraints = [AltitudeConstraint(min=0 * u.deg)]
            elif not any(isinstance(c, AltitudeConstraint) for c in b._all_constraints):
                b._all_constraints.append(AltitudeConstraint(min=0 * u.deg))
                if b.constraints is None:
                    b.constraints = [AltitudeConstraint(min=0 * u.deg)]
                else:
                    b.constraints.append(AltitudeConstraint(min=0 * u.deg))
            b._duration_offsets = u.Quantity([0*u.second, b.duration/2,
                                              b.duration])
            b.observer = self.observer
        current_time = self.schedule.start_time
        while (len(blocks) > 0) and (current_time < self.schedule.end_time):
            print(current_time)# first compute the value of all the constraints for each block
            # given the current starting time
            block_transitions = []
            block_constraint_results = []
            for b in blocks:
                # first figure out the transition
                if len(self.schedule.observing_blocks) > 0:
                    trans = self.transitioner(
                        self.schedule.observing_blocks[-1], b, current_time, self.observer)
                else:
                    trans = None
                block_transitions.append(trans)
                transition_time = 0*u.second if trans is None else trans.duration

                times = current_time + transition_time + b._duration_offsets

                # make sure it isn't in a pre-filled slot
                if (any((current_time < filled_times) & (filled_times < times[2])) or
                        any(abs(pre_filled.T[0]-current_time) < 1*u.second)):
                    block_constraint_results.append(0)

                else:
                    constraint_res = []
                    for constraint in b._all_constraints:
                        constraint_res.append(constraint(
                            self.observer, b.target, times))
                    block_constraint_results.append(np.prod(constraint_res))# take the product over all the constraints *and* times
                if block_constraint_results[-1]==1:
                    break

            # now identify the block that's the best
            bestblock_idx = np.argmax(block_constraint_results)

            if block_constraint_results[bestblock_idx] == 0.:
                # if even the best is unobservable, we need a gap
                current_time += self.gap_time
            else:
                # If there's a best one that's observable, first get its transition
                trans = block_transitions.pop(bestblock_idx)
                if trans is not None:
                    self.schedule.insert_slot(trans.start_time, trans)
                    current_time += trans.duration

                # now assign the block itself times and add it to the schedule
                newb = blocks.pop(bestblock_idx)
                newb.start_time = current_time
                current_time += newb.duration
                newb.end_time = current_time
                newb.constraints_value = block_constraint_results[bestblock_idx]

                self.schedule.insert_slot(newb.start_time, newb)

        return self.schedule

def correct_msbs(LAPprograms,path_dir):
	'''Corrects and simplifies MSB files to ensure that the total time of all MSBs matches the
	total allocated time remaining, and that there are no duplicate target sources.'''
	program_list=np.array(LAPprograms['projectid'])
	print('Correcting MSBS...\n')
	print('(- too much in MSBs, + too little in MSBs)')
	for m in program_list:
		msbs=ascii.read(path_dir+'program_details_org/'+m.lower()+'-project-info.list',names=('projectid','msbid','remaining','obscount','timeest','msb_total_hrs','instrument','type','pol','target','ra2000','dec2000','taumin','taumax'))
		remaining=LAPprograms['remaining_hrs'][np.where(LAPprograms['projectid']==m)[0][0]]
		#check no target repeats first
		cor=[(float(i),float(j)) for i,j in zip(msbs['ra2000'],msbs['dec2000'])]
		unique_srcs=list(set(cor))
		projid=[]
		msbid=[]
		remain=[]
		obsc=[]
		timeest=[]
		inst=[]
		ty=[]
		pol=[]
		targ=[]
		tmin=[]
		tmax=[]
		for src in unique_srcs:
			ind0=np.where(np.logical_and(src[0]==msbs['ra2000'],src[1]==msbs['dec2000']))[0][0]
			projid.append(msbs['projectid'][ind0])
			msbid.append(msbs['msbid'][ind0])
			inst.append(msbs['instrument'][ind0])
			ty.append(msbs['type'][ind0])
			pol.append(msbs['pol'][ind0])
			tmin.append(msbs['taumin'][ind0])
			tmax.append(msbs['taumax'][ind0])
			targ.append(msbs['target'][ind0])
			timeest.append(msbs['timeest'][ind0])
			re=[]
			ob=[]
			for i in range(0,len(msbs['ra2000'])):
				if msbs['ra2000'][i]==src[0] and msbs['dec2000'][i]==src[1]:
					re.append(msbs['remaining'][i])
					ob.append(msbs['obscount'][i])
			remain.append(np.sum(re))
			obsc.append(np.sum(ob))
		ascii.write([projid,msbid,remain,obsc,timeest,(np.array(timeest)*np.array(remain))/3600.,inst,\
			ty,pol,targ,[src[0] for src in unique_srcs], [src[1] for src in unique_srcs],tmin,tmax],\
			path_dir+'program_details_fix/'+m.lower()+'-project-info.list',names=msbs.colnames)
		#now start with fixed msb file to check time allocation matches msb times
		msbs=ascii.read(path_dir+'program_details_fix/'+m.lower()+'-project-info.list')
		msb_remaining=np.sum(msbs['msb_total_hrs'])
		diff= remaining - msb_remaining #negative is too much, + is too little
		print('Correcting MSBs file for: ',m,' --> Time difference =', round(diff,2))
		if diff != 0.:
			coords=SkyCoord(ra=msbs['ra2000']*u.rad,dec=msbs['dec2000']*u.rad,frame='icrs')
			sep=coords[0].separation(coords)
			repeats=int(remaining/(msbs['timeest'][0]/3600.))
			if np.all(sep<1.*u.degree):
				ascii.write([[msbs['projectid'][0]],[msbs['msbid'][0]],[repeats],[msbs['obscount'][0]],\
				[msbs['timeest'][0]],[(msbs['timeest'][0]*repeats)/3600.],[msbs['instrument'][0]],\
				[msbs['type'][0]],[msbs['pol'][0]],[msbs['target'][0]],[msbs['ra2000'][0]],[msbs['dec2000'][0]],\
				[msbs['taumin'][0]],[msbs['taumax'][0]]],\
				path_dir+'program_details_fix/'+m.lower()+'-project-info.list', names=msbs.colnames)	
			else:
				hrs_add=np.abs(diff)#int(abs(diff)/np.max((msbs['timeest'][0]/3600.)))
				remain=np.array(msbs['remaining'])
				t_est=np.array(msbs['timeest'])
				while hrs_add >0.:
					for i in range(0,len(msbs['target'])):
						if hrs_add>0. and remain[i]>0:
							remain[i]=remain[i]+np.sign(diff)
							hrs_add=hrs_add-(t_est[i]/3600.)#-1
						else:
							remain[i]=remain[i]
							hrs_add=hrs_add
				ascii.write([msbs['projectid'],msbs['msbid'],remain,msbs['obscount'],\
				msbs['timeest'],(msbs['timeest']*remain)/3600.,msbs['instrument'],\
				msbs['type'],msbs['pol'],msbs['target'],msbs['ra2000'],msbs['dec2000'],\
				msbs['taumin'],msbs['taumax']],\
				path_dir+'program_details_fix/'+m.lower()+'-project-info.list', names=msbs.colnames)
	print('\n')

def time_remain_p_weatherband(LAPprograms,path_dir):
	'''Calculates remaining time per weather band for each program after the simulation has run.'''
	program_list=np.array(LAPprograms['projectid'])
	fileo=open(path_dir+'sim_results/results_wb.txt','w')
	for m in program_list:
		remainwb_tally={}
		print(m)
		msbs=ascii.read(path_dir+'program_details_sim/'+m.lower()+'-project-info.list')
		uni_wb=np.unique(msbs['taumax'])
		for i in range(0,len(uni_wb)):
			wb=get_wband(uni_wb[i])
			ind=np.where(msbs['taumax']==uni_wb[i])[0]
			remainwb_tally[wb]=round(np.sum(msbs['remaining'][ind]*msbs['timeest'][ind]/3600.),2)
		fileo.write('{0} {1}\n'.format(m+':',remainwb_tally))
	fileo.close()

def transform_blocks(blocks_file):
	'''Reads in observing blocks data file. We make sure to properly deal with the irregular observing blocks data file,
	 which has inconsistent columns.'''
	newfile=blocks_file.strip('.txt')+'_corr.txt'
	f=open(newfile,'w')
	with open(blocks_file, 'r') as ins:
		array = []
		for line in ins:
			linecode=line.strip('\n').split(' ')
			if (len(linecode)==4) and ('' not in linecode):
				f.write('{0} {1} {2} {3}\n'.format(linecode[0],linecode[1],linecode[2],linecode[3]))
			elif (len(linecode)==3) and ('' not in linecode):
				f.write('{0} {1} {2} {3}\n'.format(linecode[0],linecode[1],linecode[2],'none'))
			else:
				raise ValueError('The format of your LAP blocks file is incorrect. Please check for errors in the file.')
	f.close()
	#read observing blocks
	Blocks=ascii.read(newfile,delimiter=' ',\
		guess=False,data_start=0,names=['date_start','date_end','program','extra'])
	return(Blocks)
def calc_blocks(Blocks,sim_start,sim_end):
	'''Calculates observing blocks in MJD, and keeps track of which program has priority.'''
	OurBlocks=[]
	firstprog=[]
	for kk in range(0,len(Blocks)):
		startBMJD=Time(str(Blocks[kk]['date_start'])[0:4]+'-'+str(Blocks[kk]['date_start'])[4:6]+'-'+str(Blocks[kk]['date_start'])[6:8], format='iso', scale='utc').mjd
		endBMJD=Time(str(Blocks[kk]['date_end'])[0:4]+'-'+str(Blocks[kk]['date_end'])[4:6]+'-'+str(Blocks[kk]['date_end'])[6:8], format='iso', scale='utc').mjd
		if (startBMJD >= Time(sim_start, format='iso', scale='utc').mjd) and (endBMJD <= Time(sim_end, format='iso', scale='utc').mjd):
			OurBlocks.append(Blocks[kk])
			if Blocks[kk]['extra']=='none':
				firstprog.append([Blocks[kk]['program']])
			else:
				firstprog.append([Blocks[kk]['program'],Blocks[kk]['extra']])
	return(OurBlocks,firstprog)

def read_cal(table,cal_table):
	'''Reads in calibrator data files.
	NOTE - this calibrater method is not currently implemented.'''
	cal=table['target name'][np.where(table['fraction of time observable']==np.max(table['fraction of time observable']))[0][0]]
	ind=np.where(table['target name']==cal)[0][0]
	ra=str(cal_table['col2'][ind])+'h'+str(cal_table['col3'][ind])+'m'+str(cal_table['col4'][ind])+'s'
	dec=str(cal_table['col5'][ind])+'d'+str(cal_table['col6'][ind])+'m'+str(cal_table['col7'][ind])+'s'
	return cal,ra,dec

def pick_cals(day,obsn,path_dir):
	'''Finds calibrators that are best observable on a particular night.
	NOTE - this calibrater method is not currently implemented.'''
	callst1=[]
	names1=[]
	callst2=[]
	names2=[]
	time_range1 = Time([Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 03:30",\
		Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:30"])
	if obsn==17.:
		time_range2 = Time([Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:30",\
				Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 20:30"])
	elif obsn==13.:
		time_range2 = Time([Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:30",\
				Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 16:30"])
	cal_files=[path_dir+'Harp_cal.txt',path_dir+'SCUBA2_cals.txt',path_dir+'pointing.txt']
	for ii in range(0,len(cal_files)):
		cal_table=ascii.read(cal_files[ii])
		jcmt=Observer.at_site("JCMT")
		constraints = [AltitudeConstraint(30*u.deg, 80*u.deg,boolean_constraint=True)]
		targets=[]
		for i in range(0,len(cal_table['col1'])):
			targets.append(FixedTarget(coord=SkyCoord(ra=str(cal_table['col2'][i])+'h'+str(cal_table['col3'][i])+'m'+str(cal_table['col4'][i])+'s',\
				dec=str(cal_table['col5'][i])+'d'+str(cal_table['col6'][i])+'m'+str(cal_table['col7'][i])+'s'),name=cal_table['col1'][i]))
		table1 = observability_table(constraints, jcmt, targets, time_range= time_range1)
		cal1,ra1,dec1=read_cal(table1,cal_table)
		table2 = observability_table(constraints, jcmt, targets, time_range= time_range2)
		cal2,ra2,dec2=read_cal(table2,cal_table)
		callst1.append((cal1,ra1,dec1))
		names1.append('CAL_'+cal1)
		callst2.append((cal2,ra2,dec2))
		names2.append('CAL_'+cal2)
	return callst1,names1,callst2,names2

def pick_fault_targets(day,obsn,path_dir):
	'''Finds calibrators that are best observable on a particular night for use in fault blocks.'''
	callst1=[]
	names1=[]
	callst2=[]
	names2=[]
	time_range1 = Time([Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 03:30",\
		Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:30"])
	if obsn==17.:
		time_range2 = Time([Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:30",\
				Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 20:30"])
	elif obsn==13.:
		time_range2 = Time([Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:30",\
				Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 16:30"])
	cal_file=path_dir+'pointing.txt'
	cal_table=ascii.read(cal_file)
	jcmt=Observer.at_site("JCMT")
	constraints = [AltitudeConstraint(30*u.deg, 80*u.deg,boolean_constraint=True)]
	targets=[]
	for i in range(0,len(cal_table['col1'])):
		targets.append(FixedTarget(coord=SkyCoord(ra=str(cal_table['col2'][i])+'h'+str(cal_table['col3'][i])+'m'+str(cal_table['col4'][i])+'s',\
			dec=str(cal_table['col5'][i])+'d'+str(cal_table['col6'][i])+'m'+str(cal_table['col7'][i])+'s'),name=cal_table['col1'][i]))
	table1 = observability_table(constraints, jcmt, targets, time_range= time_range1)
	cal1,ra1,dec1=read_cal(table1,cal_table)
	table2 = observability_table(constraints, jcmt, targets, time_range= time_range2)
	cal2,ra2,dec2=read_cal(table2,cal_table)
	callst1.append((cal1,ra1,dec1))
	names1.append('CAL_'+cal1)
	callst2.append((cal2,ra2,dec2))
	names2.append('CAL_'+cal2)
	return callst1,names1,callst2,names2

def get_cal_times(obsn,day,half):
	'''Creates 15min time-blocks for calibrator sources each hour of the night,
	giving 25% of observing night to calibrators.
	NOTE - this calibrater method is not currently implemented.'''
	arr=[]
	if half==1:
		st0=Time(Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 04:15")
		st1=Time(Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 04:30")
	elif half==2:
		st0=Time(Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 11:15")
		st1=Time(Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 11:30")
	for i in range(0,int(obsn)):
		t1=(st0).datetime+(i)*datetime.timedelta(hours=1)
		t2=(st1).datetime+(i)*datetime.timedelta(hours=1)
		time_range=Time([str(t1),str(t2)])
		arr.append(TimeConstraint(time_range[0], time_range[1]))
	return(arr)
def get_time_blocks(obsn,day,jcmt):
	'''Converts UT time blocks to LST blocks.'''
	lst_list=[]
	st0=Time(Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 03:30")
	st1=Time(Time((day+1), format='mjd', scale='utc').iso.split(' ')[0]+" 04:30")
	for i in range(0,int(obsn)):
		t1=(st0).datetime+(i)*datetime.timedelta(hours=1)
		t2=(st1).datetime+(i)*datetime.timedelta(hours=1)
		lst_list.append((Time(str(Time(t1)),format='iso',scale='utc',location=(jcmt.location.longitude,jcmt.location.latitude)).sidereal_time('apparent').value,\
			Time(str(Time(t2)),format='iso',scale='utc',location=(jcmt.location.longitude,jcmt.location.latitude)).sidereal_time('apparent').value))
	return(lst_list)


def get_wvm_data(sim_start,sim_end,flag,path_dir,wvmfile=''):
	'''Get WVM weather data.'''
	sim_years=int(ma.ceil(abs((Time(sim_start,format='iso').datetime-Time(sim_end,format='iso').datetime).total_seconds())/(3600.*24*365.)))
	if flag=='fetch':
		hoursstart=4#6pmHST
		hoursend=16#6amHST
		#sim_years=4
		prev_years=Time(sim_start,format='iso').datetime.year-sim_years
		prev_yeare=Time(sim_end,format='iso').datetime.year-sim_years
		print(prev_years)
		startdatewvm=Time(str(prev_years)+'-'+sim_start.split('-')[1]+'-'+sim_start.split('-')[2],format='iso').datetime
		enddatewvm=Time(str(prev_yeare)+'-'+'12-31',format='iso').datetime
		wvmvalues=get_wvm_fromdisk(startdatewvm,enddatewvm)
		wvmvalues=wvmvalues[['finalTau']]
		with open(path_dir+'program_details_sim/writewvm.csv','w') as f:
			wvmvalues.to_csv(f)
		wvmvalues=pd.read_csv(path_dir+'program_details_sim/writewvm.csv',index_col='isoTime',parse_dates=['isoTime'])
		hours=wvmvalues.index.hour+(wvmvalues.index.minute/60.0)
		nightlywvm=wvmvalues[(hours>=hoursstart) & (hours<=hoursend)]
		data_daily = get_sampled_values(nightlywvm, 'finalTau',samplerate='D')
		with open(path_dir+'program_details_sim/writewvm_daily.csv','w') as f:
			data_daily.to_csv(f)
		data_daily=ascii.read(path_dir+'program_details_sim/writewvm_daily.csv',format='csv')
	elif flag=='file':
		data_daily=ascii.read(wvmfile,format='csv')
	mjd = (Time(data_daily['isoTime'], format='iso', scale='utc').mjd)+(365.*sim_years)
	tau = data_daily['median']
	mjd_predict=np.arange(Time(sim_start, format='iso', scale='utc').mjd,\
		Time(sim_end, format='iso', scale='utc').mjd,1)
	ind_start=np.where(mjd==Time(sim_start, format='iso', scale='utc').mjd)[0][0]
	ind_end=np.where(mjd==Time(sim_end, format='iso', scale='utc').mjd)[0][0]
	tau_predict=tau[ind_start:ind_end]
	tau_predict.fill_value = 0.2
	tau_predict_fill=tau_predict.filled()
	return(mjd_predict,tau_predict_fill)

def good_blocks(Blocks,mjd_predict,tau_predict):
	'''Get MJDs and weather for the observing blocks.'''
	start=Blocks['date_start']
	end=Blocks['date_end']
	startmjd=Time(str(start)[0:4]+'-'+str(start)[4:6]+'-'+str(start)[6:8], format='iso', scale='utc').mjd
	endmjd=Time(str(end)[0:4]+'-'+str(end)[4:6]+'-'+str(end)[6:8], format='iso', scale='utc').mjd
	dates=np.arange(startmjd,endmjd+1,1)
	obs_mjd=mjd_predict[[i for i, item in enumerate(mjd_predict) if item in dates]]
	tau_mjd=tau_predict[[i for i, item in enumerate(mjd_predict) if item in dates]]
	return(obs_mjd,tau_mjd)

def bad_block(instrument,SCUBA_2_unavailable,HARP_unavailable,UU_unavailable):
	'''Returns the proper list of unavailable dates based on instrument.'''
	if instrument=='SCUBA-2':
		checklst=SCUBA_2_unavailable
	elif instrument=='HARP':
		checklst=HARP_unavailable
	elif instrument in ['UU','RXA3M']:
		checklst=RU_unavailable
	else:
		raise ValueError('Instrument unavailable.')
	return checklst

def get_wband(tau):
	'''Matches tau values to a weather band.'''
	if tau<=0.05:
		wb='Band 1'
	elif tau<=0.08 and tau>0.05:
		wb='Band 2'
	elif tau<=0.12 and tau>0.08:
		wb='Band 3'
	elif tau<=0.2 and tau>0.12:
		wb='Band 4'
	else:
		wb='Band 5'
	return(wb)

def get_m16al001_time(tau):
	'''Some programs (e.g., M16AL001) MSBs can be run in different weather bands, and the MSB files only reflect one Band (e.g., Band 3) time.
	Here we select the proper MSB time depending on the weather band for the night.'''
	if tau<=0.05:
		time=1320.
	elif tau<=0.08 and tau>0.05:
		time=1920.
	elif tau<=0.12 and tau>0.08:
		time=2520.
	return(time)
def get_lst(sched,unused_ind,jcmt):
	'''Converts UT times of unused blocks in a schedule to LST times.'''
	lst_list=[]
	start=sched['start time (UTC)'][unused_ind]
	end=sched['end time (UTC)'][unused_ind]
	for i in range(0,len(start)):
		lst_list.append((Time(start[i],format='iso',scale='utc',location=(jcmt.location.longitude,jcmt.location.latitude)).sidereal_time('apparent').value,\
			Time(end[i],format='iso',scale='utc',location=(jcmt.location.longitude,jcmt.location.latitude)).sidereal_time('apparent').value))
	return(lst_list)

def priority_choose(tau,m,tau_max,FP,LAPprograms):
	'''Calculates scaled priorities of MSBs.'''
	#priority goes (1) faults [so they always happen at same rate], (2) block program, (3) weather band, (4) overall priority
	tab=LAPprograms['projectid','tagpriority']
	tab['scaled_p']=[i for i in range(1,len(LAPprograms['projectid'])+1)]
	p=tab['scaled_p'][np.where(tab['projectid']==m)[0]]
	WBandDay=int(get_wband(tau).split(' ')[1])
	WBandTar=int(get_wband(tau_max).split(' ')[1])
	if m.lower() in FP:
		priority=2
	else:
		if WBandDay==WBandTar:
			priority=3+p
		else:
			priority=(2*len(LAPprograms['projectid'])+WBandTar+3)+p
	return priority
def bin_lst(bin_start,bin_end,lst_tally):
    '''Bins LST blocks for histogram creation.'''
    bin_hrs=np.zeros(len(bin_start))
    for i in range(0,len(bin_start)):
        for j in range(0,len(lst_tally)):
            if bin_start[i] <= lst_tally[j][0] <= bin_end[i] and bin_start[i] <= lst_tally[j][1] <= bin_end[i]:
                bin_hrs[i]=bin_hrs[i]+(lst_tally[j][1]-lst_tally[j][0])
            elif bin_start[i] <= lst_tally[j][0] <= bin_end[i] and lst_tally[j][1] > bin_end[i]:
                bin_hrs[i]=bin_hrs[i]+(bin_end[i]-lst_tally[j][0])
            elif bin_start[i] <= lst_tally[j][1] <= bin_end[i] and lst_tally[j][0] < bin_start[i]:
                bin_hrs[i]=bin_hrs[i]+(lst_tally[j][1]-bin_start[i])
    return(bin_hrs)
def elevationcheck(jcmt,mjd,target):
	'''Some program (e.g., M17BL002) can have low elevation sources, so this checks if they transit below 40 deg,
	and adjusts elevation limit for these sources.'''
	date=Time(mjd,format='mjd').iso.split(' ')[0]
	airmass=jcmt.altaz(Time(date)+np.linspace(-12,12,100)*u.hour,target).secz
	amin=np.min(airmass[airmass>1]).value
	elev=np.degrees(np.pi/2 -np.arccos(1./amin))
	if elev < 40.:
		constraint=12.
	else:
		constraint=30.
	return(constraint)
def predict_time(sim_start,sim_end,wvmfile,LAPprograms,Block,path_dir,flag,total_observed,FP,m16al001_tally,\
	SCUBA_2_unavailable,HARP_unavailable,RU_unavailable,wb_usage_tally,cal_tally,tot_tally,nothing_obs,lst_tally,finished_dates,status):
	'''Simulates observations of Large Programs over specified observing block.'''

	#fetch wvm data from previous year(s)
	mjd_predict,tau_predict=get_wvm_data(sim_start,sim_end,flag,path_dir,wvmfile)

	#get dates for observing block, make arrays of MJD and tau for these days
	obs_mjd,tau_mjd=good_blocks(Block,mjd_predict,tau_predict)

	#set up observatory site and general elevation constraints
	jcmt=Observer.at_site("JCMT",timezone="US/Hawaii")
	constraints = [AltitudeConstraint(min=0*u.deg)]

	print('block:',obs_mjd)
	#loop over all days in the current observing block
	for k in range(0,len(obs_mjd)):
		print('day:',obs_mjd[k])
		#A standard observing night will run from 5:30pm HST to 6:30am HST (13 hrs; times below are UTC!)
		#if tau is at band 3 or better, EO is scheduled, and we observe till 10:30am HST (17 hrs)
		if tau_mjd[k] < 0.12:
			time_range = Time([Time((obs_mjd[k]+1), format='mjd', scale='utc').iso.split(' ')[0]+" 03:30",\
				Time((obs_mjd[k]+1), format='mjd', scale='utc').iso.split(' ')[0]+" 20:30"])
			obsn=17.
		else:
			time_range = Time([Time((obs_mjd[k]+1), format='mjd', scale='utc').iso.split(' ')[0]+" 03:30",\
				Time((obs_mjd[k]+1), format='mjd', scale='utc').iso.split(' ')[0]+" 16:30"])
			obsn=13.
		tot_tally.append(obsn)
		WBand=get_wband(tau_mjd[k])
		wb_usage_tally['Available'][WBand].append(obsn)
		#make a target list for the night (we keep track of target, MSB time, priority, and program),
		#looping over each target in each program
		targets=[]
		priority=[]
		msb_time=[]
		prog=[]
		tc=[]
		ta=[]
		configu=[]
		for m in LAPprograms['projectid']:
			#if m in ['M16AL001']:#['M16AL001','M16AL005','M16AL006','M17BL001','M17BL002']#,'M17BL004','M17BL005','M17BL006','M17BL007','M17BL009','M17BL010','M17BL011']:
			if LAPprograms['taumax'][np.where(LAPprograms['projectid']==m.upper())[0]] >= tau_mjd[k]:
				msbs=ascii.read(path_dir+'program_details_sim/'+m.lower()+'-project-info.list')
				target_table=msbs['target','ra2000', 'dec2000']
				obs_time_table=msbs['target','timeest','remaining','taumin','taumax','instrument']
				#targets are added if they meet all of the following requirments:
				#(a) Target has MSB repeats remaining
				#(b) The night is not in the blackout dates for the instrument
				#(c) The weather is appropriate
				for j in range(0,len(target_table['target'])):
					blackout_dates=bad_block(obs_time_table['instrument'][j],SCUBA_2_unavailable,HARP_unavailable,RU_unavailable)
					if (obs_time_table['remaining'][j] >0 and obs_time_table['taumax'][j] >= tau_mjd[k] and obs_mjd[k] not in blackout_dates):
						#The m16al001 program is to run on a monthly basis, so if we are dealing with that
						#program we must check whether each target has been observed in the current month yet.
						if m != 'M16AL001':
							for jj in range(0,obs_time_table['remaining'][j]):
								targets.append(FixedTarget(coord=SkyCoord(ra=target_table['ra2000'][j]*u.rad,\
									dec=target_table['dec2000'][j]*u.rad),name=target_table['target'][j]))
								tc.append(TimeConstraint(time_range[0], time_range[1]))
								#if in M17BL002, and transits below 40 deg, set the elevation limit to 15 rather than 30 for the target so it is observed
								if m == 'M17BL002':
									el=elevationcheck(jcmt,obs_mjd[k],FixedTarget(coord=SkyCoord(ra=target_table['ra2000'][j]*u.rad,dec=target_table['dec2000'][j]*u.rad),name=target_table['target'][j]))
									ta.append(AltitudeConstraint(min=el*u.deg))
								else:
									ta.append(AltitudeConstraint(min=30*u.deg))
								#assign priority
								priority.append(priority_choose(tau_mjd[k],m,obs_time_table['taumax'][j],FP,LAPprograms))
								msb_time.append(1.25*obs_time_table['timeest'][j]*u.second)
								configu.append(m.lower()+'_'+str(msbs['msbid'][j])+'_'+str(j)+'_'+str(jj))
								prog.append(m.lower())
						else:
							#we keep track of the dates each target in the m16al001 program is observed through the m16al001_tally dictionary,
							#so we need to first check if the target is present in the dictionary yet
							dates_obs=[all(getattr(Time(obs_mjd[k], format='mjd', scale='utc').datetime,x)==getattr(mon.datetime,x) for x in ['year','month']) for mon in m16al001_tally[target_table['target'][j]]]
							if target_table['target'][j] not in m16al001_tally.keys():
								targets.append(FixedTarget(coord=SkyCoord(ra=target_table['ra2000'][j]*u.rad,\
									dec=target_table['dec2000'][j]*u.rad),name=target_table['target'][j]))
								tc.append(TimeConstraint(time_range[0], time_range[1]))
								#assign priority
								priority.append(priority_choose(tau_mjd[k],m,obs_time_table['taumax'][j],FP,LAPprograms))
								msb_time.append(1.25*get_m16al001_time(tau_mjd[k])*u.second)
								configu.append(m.lower()+'_'+str(msbs['msbid'][j])+'_'+str(j))
								prog.append(m.lower())
								ta.append(AltitudeConstraint(min=30*u.deg))
							else:
								#then if present, check if the target has been observed in the current nights month/year combo yet
								if not any(dates_obs):
									targets.append(FixedTarget(coord=SkyCoord(ra=target_table['ra2000'][j]*u.rad,\
										dec=target_table['dec2000'][j]*u.rad),name=target_table['target'][j]))
									tc.append(TimeConstraint(time_range[0], time_range[1]))
									#assign priority
									priority.append(priority_choose(tau_mjd[k],m,obs_time_table['taumax'][j],FP,LAPprograms))
									msb_time.append(1.25*get_m16al001_time(tau_mjd[k])*u.second)
									configu.append(m.lower()+'_'+str(msbs['msbid'][j])+'_'+str(j))
									prog.append(m.lower())
									ta.append(AltitudeConstraint(min=30*u.deg))
		#check if at least one target has been added to our potential target list
		if len(targets)>0:
			#check at least one potential target is observable at some point in the night
			ever_observable = is_observable([AltitudeConstraint(min=30*u.deg)], jcmt, targets, time_range=time_range,time_grid_resolution=0.75*u.hour)
			frac_obs=observability_table([AltitudeConstraint(min=30*u.deg)], jcmt, targets, time_range=time_range,time_grid_resolution=0.75*u.hour)['fraction of time observable']
			#add in some fault blocks - pick a pointing cal best observable for each half night and schedule a 1.5% of total night block == 3% total fault rate
			callst1,names1,callst2,names2=pick_fault_targets(obs_mjd[k],obsn,path_dir)
			mid_time=Time(Time((obs_mjd[k]+1), format='mjd', scale='utc').iso.split(' ')[0]+" 10:00")
			targets.append(FixedTarget(coord=SkyCoord(ra=callst1[0][1],\
				dec=callst1[0][2]),name='FAULT'))
			msb_time.append(0.015*obsn*3600.*u.second)
			priority.append(1)
			tc.append(TimeConstraint(time_range[0], mid_time))
			prog.append('FAULT')
			ta.append(AltitudeConstraint(min=0*u.deg))
			configu.append('FAULT')
			targets.append(FixedTarget(coord=SkyCoord(ra=callst2[0][1],\
				dec=callst2[0][2]),name='FAULT'))
			msb_time.append(0.015*obsn*3600.*u.second)
			priority.append(1)
			tc.append(TimeConstraint(mid_time, time_range[1]))
			prog.append('FAULT')
			ta.append(AltitudeConstraint(min=0*u.deg))
			configu.append('FAULT')
			#pick a HARP, SCUBA-2, and pointing calibrator for each half-night, and add to target list
			'''(optional) schedule calibrators every hour
			#calibrators are chosen based on observability each half-night
			callst1,names1,callst2,names2=pick_cals(obs_mjd[k],obsn,path_dir)
			names=names1+names2
			repeats=7
			repeats2=int(obsn-repeats)
			#15min of every hour is set aside for calibrators (25% of night)
			cal_times1=get_cal_times(repeats,obs_mjd[k],1)
			cal_times2=get_cal_times(repeats2,obs_mjd[k],2)
			for ii in range(0,len(callst1)):
				for iii in range(0,repeats):
					targets.append(FixedTarget(coord=SkyCoord(ra=callst1[ii][1],\
						dec=callst1[ii][2]),name='CAL_'+callst1[ii][0]))
					msb_time.append(3*60.*u.second)
					#calibrators given highest priority (1) to ensure they are scheduled
					priority.append(1)
					tc.append(cal_times1[iii])
					if ii==0:
						prog.append('HARP cal')
					elif ii==1:
						prog.append('SCUBA-2 cal')
					elif ii==2:
						prog.append('Pointing/Focus cal')
			for ii in range(0,len(callst2)):
				for iii in range(0,repeats2):
					targets.append(FixedTarget(coord=SkyCoord(ra=callst2[ii][1],\
						dec=callst2[ii][2]),name='CAL_'+callst2[ii][0]))
					msb_time.append(3*60.*u.second)
					priority.append(1)
					tc.append(cal_times2[iii])
					if ii==0:
						prog.append('HARP cal')
					elif ii==1:
						prog.append('SCUBA-2 cal')
					elif ii==2:
						prog.append('Pointing/Focus cal')'''						
		else:
			ever_observable = [False]
			frac_obs=[0.]
		#As long as we have an observable target, we proceed to scheduling for the night.
		if np.any(ever_observable) and any(frac>=0.75/obsn for frac in frac_obs):
			#set the slew rate of telescope between sources
			slew_rate = 1.2*u.deg/u.second
			transitioner = Transitioner(slew_rate)#, {'program':{'default':2*u.minute}})
			# set up the schedule for the night
			scheduler = JCMTScheduler(constraints = constraints,observer = jcmt,\
				transitioner = transitioner,time_resolution=30*u.minute,gap_time=30*u.minute)
			schedule = Schedule(time_range[0], time_range[1])
			#create observing blocks for each target based on target list made above (these act just like normal MSBs at JCMT)
			bnew=[]
			for i in range(0,len(targets)):
				bnew.append(ObservingBlock(targets[i],msb_time[i], priority[i], configuration={'program' : prog[i]},\
							constraints=[tc[i],ta[i]]))
			#run the astroplan priority scheduling tool
			scheduler(bnew, schedule)
			sched=schedule.to_table(show_unused=True)
			#print out schedule to file for records
			namemjd=str(Time(obs_mjd[k],format='mjd').datetime.year)+\
			str(Time(obs_mjd[k],format='mjd').datetime.month)+str(Time(obs_mjd[k],format='mjd').datetime.day)
			ascii.write([sched['target'],sched['start time (UTC)'],sched['end time (UTC)'],sched['duration (minutes)']],\
				path_dir+'sim_results/schedules/'+namemjd+'.txt',names=['target','start time (UTC)','end time (UTC)','duration (min)'])
			#keep track of unused time
			WBand=get_wband(tau_mjd[k])
			unused_ind=np.where(np.logical_or(np.logical_or(sched['target']=='Unused Time',sched['target']=='TransitionBlock'),sched['target']=='FAULT'))[0]
			unused=np.sum(np.array(sched['duration (minutes)'][unused_ind]))/60.
			lst_list=get_lst(sched,unused_ind,jcmt)
			lst_tally[WBand].extend(lst_list)
			wb_usage_tally['Unused'][WBand].append(unused)
			#uncomment if selecting calibrators specifically
			#caltime=np.sum(np.array(sched['duration (minutes)'])[[p for p,n in enumerate(np.array(sched['target'])) if n in names]])/60.
			#wb_usage_tally['Cal'][WBand].append(caltime)
			#cal_tally.append(caltime)

			#FOR TESTING ONLY--
			fig=plt.figure(figsize = (14,10))
			plot_schedule_airmass(schedule,show_night=True)
			plt.title(WBand)
			lgd=plt.legend(loc = "upper right",ncol=10,fontsize=6,bbox_to_anchor=(1.05,1.15))
			plt.savefig(path_dir+'sim_results/schedules/'+namemjd+'.png',bbox_tight='inches',bbox_extra_artists=(lgd,))
			plt.close()
			#input('s')

			#record what targets have been observed, updating the MSB files and recording total time observed for each program
			for h in range(0,len(np.unique(sched['target']))):
				tar=np.unique(sched['target'])[h]
				if (tar not in ['TransitionBlock','Unused Time','FAULT'] and 'CAL' not in tar):
					prog=sched['configuration'][np.where(sched['target']==tar)[0][0]]['program']
					msbs=ascii.read(path_dir+'program_details_sim/'+prog+'-project-info.list')
					num_used=len(np.array(sched['duration (minutes)'][np.where(sched['target']==tar)[0]]))
					if prog=='m16al001':
						tim_used=float(msbs['timeest'][np.where(msbs['target']==tar)[0][0]])/3600.
						m16al001_tally[tar].append(Time(obs_mjd[k],format='mjd',scale='utc'))
					else:
						tim_used=np.sum(np.array(sched['duration (minutes)'][np.where(sched['target']==tar)[0]]))/60./1.25
					if tim_used >0:
						r0=msbs['remaining'][np.where(msbs['target']==tar)[0]]-num_used
						oc0=msbs['obscount'][np.where(msbs['target']==tar)[0]]+num_used
						t0=np.round(msbs['msb_total_hrs'][np.where(msbs['target']==tar)[0]]-tim_used,5)
						msbs['remaining'][np.where(msbs['target']==tar)[0]]=r0
						msbs['obscount'][np.where(msbs['target']==tar)[0]]=oc0
						msbs['msb_total_hrs'][np.where(msbs['target']==tar)[0]]=t0
						ascii.write([msbs['projectid'],msbs['msbid'],msbs['remaining'],msbs['obscount'],\
						msbs['timeest'],msbs['msb_total_hrs'],msbs['instrument'],
						msbs['type'],msbs['pol'],msbs['target'],msbs['ra2000'],msbs['dec2000'],\
						msbs['taumin'],msbs['taumax']],\
						path_dir+'program_details_sim/'+prog+'-project-info.list', names=msbs.colnames)
						total_observed[prog.upper()]=total_observed[prog.upper()]+(tim_used)
						WBand=get_wband(tau_mjd[k])
						wb_usage_tally['Used'][WBand].append(tim_used)
						caltime=(tim_used)*0.25#np.sum(np.array(sched['duration (minutes)'])[[p for p,n in enumerate(np.array(sched['target'])) if n in names]])/60.
						wb_usage_tally['Cal'][WBand].append(caltime)
						cal_tally.append(caltime)
							 
		else:
			WBand=get_wband(tau_mjd[k])
			wb_usage_tally['Unused'][WBand].append(obsn)
			wb_usage_tally['Used'][WBand].append(0.)
			wb_usage_tally['Cal'][WBand].append(0.)
			lst_list=get_time_blocks(obsn,obs_mjd[k],jcmt)
			lst_tally[WBand].extend(lst_list)
			nothing_obs.append(obsn)
		#check if any programs are complete and record date
		for m in LAPprograms['projectid']:
			msbs=ascii.read(path_dir+'program_details_sim/'+m.lower()+'-project-info.list')
			rem_cnt=np.sum(msbs['remaining'])
			if rem_cnt <=0. and  finished_dates[m] == 'not finished':
				finished_dates[m]=Time(obs_mjd[k],format='mjd').iso.split(' ')[0]
		if 'not finished' in finished_dates.values():
			status='not complete'
		else:
			status='complete'
	return total_observed,m16al001_tally,wb_usage_tally,cal_tally,tot_tally,nothing_obs,lst_tally,finished_dates,status

def incremental_comprate(program_list,dats,pers,block,total_observed,RH):
	'''Keep track of program progress throughout simualtion.'''
	for ii in range(0,len(program_list)):
		#print ii
		st=str(block['date_start'])
		en=str(block['date_end'])
		startime=(datetime.datetime(int(st[0:4]),int(st[4:6]), int(st[6:8])))
		endtime=(datetime.datetime(int(en[0:4]),int(en[4:6]), int(en[6:8])))
		dats[ii].append(startime+(np.array(endtime)-np.array(startime))/2)
		rem=(RH['remaining_hrs'][np.where(RH['projectid']==program_list[ii].upper())[0][0]]-total_observed[program_list[ii].upper()])
		tot=RH['allocated_hrs'][np.where(RH['projectid']==program_list[ii].upper())[0][0]]
		pers[ii].append((tot-rem)/tot)
	return(dats,pers)
def writeLSTremain(jcmt,prog_list,sim_end):
	'''Make LST histogram plots for remaining MSBs.'''
	lst_tally2=defaultdict(list)
	for m in prog_list:
		date=Time(sim_end+' 03:30')
		msbs=ascii.read(path_dir+'program_details_sim/'+m.lower()+'-project-info.list')
		for j in range(0,len(msbs['target'])):
			lst_list=[]
			if msbs['remaining'][j] >0:
				tar=FixedTarget(coord=SkyCoord(ra=msbs['ra2000'][j]*u.rad,dec=msbs['dec2000'][j]*u.rad),name=msbs['target'][j])
				WBand=get_wband(msbs['taumax'][j])
				transit_time=jcmt.target_meridian_transit_time(date,tar,'next')
				lengthmsb=msbs['timeest'][j]
				startt=(transit_time-TimeDelta(lengthmsb/2.,format='sec')).iso
				endt=(transit_time+TimeDelta(lengthmsb/2.,format='sec')).iso
				for k in range(0,msbs['remaining'][j]):
					lst_list.append((Time(startt,format='iso',scale='utc',location=(jcmt.location.longitude,jcmt.location.latitude)).sidereal_time('apparent').value,\
						Time(endt,format='iso',scale='utc',location=(jcmt.location.longitude,jcmt.location.latitude)).sidereal_time('apparent').value))
				lst_tally2[WBand].extend(lst_list)
	#plot
	fig=plt.figure()
	font={'family':'serif','weight':'bold','size' : '14'}
	rc('font',**font)
	mpl.rcParams['xtick.direction']='in'
	mpl.rcParams['ytick.direction']='in'
	colors=['b','purple','r','orange','g']
	bands=['Band 1','Band 2', 'Band 3', 'Band 4', 'Band 5']
	bin_start=np.arange(0,25)
	bin_end=np.arange(1,26)
	ax0=plt.subplot(111)
	for i in range(0,len(bands)):
		ax=plt.subplot(3,2,i+1)
		bin_hrs=bin_lst(bin_start,bin_end,lst_tally2[bands[i]])
		ax.bar(bin_start+0.5, bin_hrs, width = 1,color=colors[i],alpha=0.6)
		ax.set_xlim(min(bin_start), max(bin_end))
		ax.set_title(bands[i],fontsize=12)
		ax.set_xlabel('${\\rm \\bf LST}$',fontsize=12)
		ax.set_ylabel('${\\rm \\bf Hours}$',fontsize=12,labelpad=20)
		ax.tick_params(axis='x',which='major', labelsize=10,length=5,width=1.5,top='on',bottom='on')
		ax.tick_params(axis='x',which='minor', labelsize=10,length=3.5,width=1,top='on',bottom='on')
		ax.tick_params(axis='y',which='major', labelsize=10,length=5,width=1.5,right='on',left='on')
		ax.tick_params(axis='y',which='minor', labelsize=10,length=3.5,width=1,right='on',left='on')
		ax.xaxis.set_minor_locator(AutoMinorLocator(5))
		ax.yaxis.set_minor_locator(AutoMinorLocator(5))
		ax.get_yaxis().set_label_coords(-0.2,0.5)
	fig.subplots_adjust(wspace=0.5,hspace=1)
	plt.savefig(path_dir+'sim_results/unused_RA_remaining.pdf',bbox_inches='tight')

#############################################################
#User Input
#############################################################
start=time.time()
path_dir='/export/data2/atetarenko/LP_predict/'

sim_start='2019-09-19'
sim_end='2020-02-01'

flag='fetch'
wvmfile=''#path_dir+'wvmvalues_onepernight.csv'
#IF NOT on EAO computer, then you need to generate:
#(1) daily wvm file (csv format,4 columns:isoTime,mean,median,count)
#e.g., on EAO computer run the partner script, getwvm.py
#AND place output file in path_dir
#THEN (a) enter file name in wvmfile variable above, (b) change fetch to file

blocks_file=path_dir+'LAP-UT-blocks-real-and-model-blocks.txt'

#dates the instruments are unavailable in MJD
SCUBA_2_unavailable=[]
HARP_unavailable=[]
RU_unavailable=np.arange(58484,58604)
#############################################################

#create output directory tree structure
if not os.path.isdir(os.path.join(path_dir, 'program_details_org/')):
    os.mkdir(os.path.join(path_dir, "program_details_org"))
if not os.path.isdir(os.path.join(path_dir, 'program_details_fix/')):
    os.mkdir(os.path.join(path_dir, "program_details_fix"))
if not os.path.isdir(os.path.join(path_dir, 'program_details_sim/')):
    os.mkdir(os.path.join(path_dir, "program_details_sim"))
if not os.path.isdir(os.path.join(path_dir, 'sim_results/')):
    os.mkdir(os.path.join(path_dir, "sim_results"))

#############################################################
#SQL Queries using the omp-python code
#Here a summary file and MSB files for all LAPs are created
#for the script to use.
#############################################################
db=ArcDB() 
#create LAP summary file
querystring1='SELECT p.projectid, p.semester, q.tagpriority, (p.remaining/60/60) as remaining_hrs, (p.allocated/60/60) as allocated_hrs,  p.taumin, p.taumax FROM omp.ompproj  as p JOIN omp.ompprojqueue as q ON p.projectid=q.projectid WHERE semester="LAP" AND  remaining >0 ORDER BY q.tagpriority;'
results_LAP_proj=db.read(querystring1)
res1=list(results_LAP_proj)
with open(path_dir+'LP_priority.txt','w') as filehandle:
	filehandle.write('projectid\tsemester\ttagpriority\tremaining_hrs\tallocated_hrs\ttaumin\ttaumax\n')
	for listitem in res1:
		filehandle.write('	'.join(map(str,listitem))+'\n')
#read in LAP details
LAPprograms_file=path_dir+'LP_priority.txt'
LAPprograms=ascii.read(LAPprograms_file)
LAPprograms.sort('tagpriority')
print('Current Large programs:\n')
print(LAPprograms, '\n')
RH=LAPprograms['projectid','remaining_hrs','allocated_hrs']
program_list=np.array(LAPprograms['projectid'])

#create MSB files for all LAPs
querystring2='SELECT m.projectid, m.msbid, m.remaining, m.obscount, m.timeest, (m.remaining*m.timeest/60/60) as msb_total_hrs,  o.instrument, o.type, o.pol, o.target, o.ra2000,o.dec2000,m.taumin,m.taumax FROM omp.ompmsb as m JOIN omp.ompobs as o ON m.msbid=o.msbid WHERE m.projectid=%(proj)s AND m.remaining >=1;' 
for prog in program_list:
	params_sql={'proj':str(prog)}
	results_LAP_msbs=db.read(querystring2,params=params_sql)
	res2=list(results_LAP_msbs)
	with open(path_dir+'program_details_org/'+prog.lower()+'-project-info.list','w') as filehandle:
		filehandle.write('projectid\tmsbid\tremaining\tobscount\ttimeest\tmsb_total_hrs\tinstrument\ttype\tpol\ttarget\tra2000\tdec2000\ttaumin\ttaumax\n')
		for listitem in res2:
			filehandle.write('	'.join(map(str,listitem))+'\n')
#IF NOT on EAO computer, then you need to generate:
#(1) LP priority through sql query to output file
#(2) MSB files for all LAPs
#e.g., on EAO computer type ompsql < script.sql > output.txt 
#where script.sql is example-project-summary.sql (1) and example-project-info.sql (2)
#AND place (1) in path_dir and (2)s in path_dir+'program_details_org/'
#THEN uncomment the following line
#LAPprograms_file=path_dir+'LP_priority.txt'
#############################################################


#empty out simulations folder and add current program files
os.system('rm -rf '+path_dir+'program_details_sim/*.list')
os.system('rm -rf '+path_dir+'sim_results/schedules/*.txt')
os.system('rm -rf '+path_dir+'sim_results/schedules/*.png')
os.system('rm -rf '+path_dir+'sim_results/*.pdf')
os.system('rm -rf '+path_dir+'sim_results/*.txt')
os.system('cp -r '+path_dir+'program_details_fix/*.list '+path_dir+'program_details_sim')

print('Predicting Large Program observations between '+sim_start+' and '+sim_end+' ...\n')

#correct MSB files to match allocation
correct_msbs(LAPprograms,path_dir)

#calculate observing blocks within the selected simulation dates
Blocks=transform_blocks(blocks_file)
OurBlocks,firstprog=calc_blocks(Blocks,sim_start,sim_end)

#run observation simulator for each observing block
total_observed = {k:v for k,v in zip(program_list,np.zeros(len(program_list)))}
finished_dates = {k:'not finished' for k in program_list}
m16al001_tally=defaultdict(list)
wb_usage_tally={'Available':defaultdict(list),'Used':defaultdict(list),'Unused':defaultdict(list),'Cal':defaultdict(list)}
for i in ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5']:
	[wb_usage_tally[j][i].append(0) for j in wb_usage_tally.keys()]
lst_tally=defaultdict(list)
cal_tally=[]
tot_tally=[]
nothing_obs=[]
dats=[]
[dats.append([]) for i in program_list]
pers=[]
[pers.append([]) for i in program_list]
for i in range(0,len(program_list)):
	dats[i].append(datetime.datetime(int(sim_start.split('-')[0]),int(sim_start.split('-')[1]),int(sim_start.split('-')[2])))
	ind=np.where(RH['projectid']==program_list[i])[0][0]
	pers[i].append((RH['allocated_hrs'][ind]-RH['remaining_hrs'][ind])/RH['allocated_hrs'][ind])
blockst=[]
blockend=[]
blockprog=[]
status='not complete'
for jj in range(0,len(OurBlocks)):
	if status=='complete':
		break
	else:
		FP=firstprog[jj]
		total_observed,m16al001_tally,wb_usage_tally,cal_tally,tot_tally,nothing_obs,lst_tally,finished_dates,status=predict_time(sim_start,sim_end,wvmfile,LAPprograms,OurBlocks[jj],path_dir,flag,total_observed,FP,m16al001_tally,SCUBA_2_unavailable,HARP_unavailable,RU_unavailable,wb_usage_tally,cal_tally,tot_tally,nothing_obs,lst_tally,finished_dates,status)
		dats,pers=incremental_comprate(program_list,dats,pers,OurBlocks[jj],total_observed,RH)
		st=str(OurBlocks[jj]['date_start'])
		en=str(OurBlocks[jj]['date_end'])
		blockst.append(datetime.datetime(int(st[0:4]),int(st[4:6]), int(st[6:8])))
		blockend.append(datetime.datetime(int(en[0:4]),int(en[4:6]), int(en[6:8])))
		blockprog.append(OurBlocks[jj]['program'].upper())

#calculate final results
obs_hrs=[]
remaining_new=[]
for i in range(0,len(program_list)):
	#because MSB times are not always an exact match to total allocated time, if remaining time less than the smallest MSB then set remaining time to 0
	msbsf=ascii.read(path_dir+'program_details_sim/'+program_list[i].lower()+'-project-info.list')
	max_time=np.max(msbsf['timeest'])/3600.
	rmn=round(RH['remaining_hrs'][np.where(RH['projectid']==program_list[i].upper())[0][0]]-total_observed[program_list[i].upper()],2)
	if rmn < max_time:
		remaining_new.append(0.)
	else:
		remaining_new.append(round(RH['remaining_hrs'][np.where(RH['projectid']==program_list[i].upper())[0][0]]-total_observed[program_list[i].upper()],2))
	obs_hrs.append(round(total_observed[program_list[i].upper()],2))	

#write final results to a file and screen
ascii.write([RH['projectid'],RH['allocated_hrs'],RH['remaining_hrs'],obs_hrs,remaining_new],\
	path_dir+'sim_results/results.txt', names=['projectid','allocted_hrs','remaining_hrs','sim_obs_hrs','remaining_aftersim'])

new=ascii.read(path_dir+'sim_results/results.txt')
print('\nFinal Prediction Results...\n')
print(new)

#plot final results in a bar chart
fig=plt.figure()
font={'family':'serif','weight':'bold','size' : '14'}
rc('font',**font)
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
ax=plt.subplot(111)
progs=[i for i in RH['projectid']]
obsh=[i for i in new['sim_obs_hrs']]
remainh=[i for i in new['remaining_aftersim']]
ypos = np.arange(len(progs))
width = 0.45
p3=ax.barh(ypos,obsh,width,color='b',edgecolor='k',lw=1.2,alpha=0.6)
p4=ax.barh(ypos,remainh,width,left=obsh,color='orange',edgecolor='k',lw=1.2)
ax.set_yticks(ypos)
ax.set_yticklabels(progs)
for i in range(0,len(progs)):
	if finished_dates[progs[i]] != 'not finished':
		ax.text(obsh[i]+remainh[i]+20,ypos[i], 'Finished: '+str(finished_dates[progs[i]]),fontsize=7)
ax.legend((p3[0],p4[0]), ('Observed','Remaining'),fontsize=7,loc='top right',ncol=1)
ax.tick_params(axis='x',which='major', labelsize=9,length=5,width=1.5,top='on',bottom='on',pad=7)
ax.tick_params(axis='x',which='minor', labelsize=9,length=3.5,width=1.,top='on',bottom='on',pad=7)
ax.tick_params(axis='y',which='major', labelsize=10,length=5,width=1.5,right='off',left='on')
ax.set_xlabel('${\\rm \\bf Hours}$',fontsize=12)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
fig.subplots_adjust(wspace=0.5,hspace=1)
plt.savefig(path_dir+'sim_results/prog_results.pdf',bbox_inches='tight')

#plot program completion chart
fig=plt.figure()
font={'family':'serif','weight':'bold','size' : '14'}
rc('font',**font)
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
ax=plt.subplot(111)
colors = cm.rainbow(np.linspace(0, 1, len(program_list)))
colordict={}
colordict['M16AL004']='gray'
for jj in range(0,len(program_list)):
	colordict[program_list[jj]]=colors[jj]
	ax.plot(dats[jj],np.array(pers[jj])*100.,color=colors[jj],ls='-',marker='o',label=program_list[jj])
for j in range(0,len(blockst)):
	ax.axvspan(blockst[j], blockend[j], facecolor=colordict[blockprog[j]], alpha=0.3)
sim_dys=abs((datetime.datetime.strptime(sim_end,'%Y-%m-%d')-datetime.datetime.strptime(sim_start,'%Y-%m-%d')).days)
locator1a = mdates.DayLocator(interval=int((((sim_dys/20.)+3/2)/3)*3))
locator1 = mdates.DayLocator(interval=int(int((((sim_dys/20.)+3/2)/3)*3)/3))
ax.xaxis.set_major_locator(locator1a)
ax.xaxis.set_minor_locator(locator1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right') 
ax.set_title('Completion as of '+str(blockend[-1]).split(' ')[0])
ax.set_ylim(0,100)
ax.legend(loc='upper right',bbox_to_anchor=(1.45,1.))
ax.tick_params(axis='x',which='major', labelsize=9,length=5,width=1.5,top='on',bottom='on',pad=7)
ax.tick_params(axis='x',which='minor', labelsize=9,length=3.5,width=1.,top='on',bottom='on',pad=7)
ax.tick_params(axis='y',which='major', labelsize=9,length=5,width=1.5,right='on',left='on')
ax.tick_params(axis='y',which='minor', labelsize=9,length=3.5,width=1.,right='on',left='on')
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('${\\rm \\bf Completion\\,Percentage}\\,(\\%)}$',fontsize=12)
ax.set_xlabel('${\\rm \\bf Time\\,(DD-MM-YYYY\\,HST)}$',fontsize=12)
plt.savefig(path_dir+'sim_results/prog_completion.pdf',bbox_inches='tight')


#append totals to results file and print to screen
fileo=open(path_dir+'sim_results/results.txt','a')
fileo.write("\nTotal Allocated Hrs for Large Programs: "+str(round(np.sum(RH['allocated_hrs']),2))+'\n')
fileo.write("Total Remaining Hrs for Large Programs before Simulation: "+str(round(np.sum(RH['remaining_hrs']),2))+'\n')
fileo.write("Total Hrs Available for Large Program Observing in Simulation:"+str(round(np.sum(tot_tally),2))+'\n')
fileo.write("Total Observed Hrs for Large Programs in Simulation: "+str(round(np.sum(obs_hrs),2))+'\n')
fileo.write("Total Remaining Hrs for Large Programs after Simulation: "+str(round(np.sum(remaining_new),2))+'\n')
fileo.write("Total Hrs lost to weather (i.e., nights where nothing observed):"+str(np.sum(nothing_obs))+'\n')
fileo.close()
print("Total Allocated Hrs for Large Programs: ",round(np.sum(RH['allocated_hrs']),2))
print("Total Remaining Hrs for Large Programs before Simulation: ",round(np.sum(RH['remaining_hrs']),2))
print("Total Hrs Available for Large Program Observing in Simulation:", round(np.sum(tot_tally),2))
print("Total Observed Hrs for Large Programs in Simulation: ",round(np.sum(obs_hrs),2))
print("Total Remaining Hrs for Large Programs after Simulation: ",round(np.sum(remaining_new),2))
print("Total Hrs lost to weather (i.e., nights where nothing observed):", np.sum(nothing_obs))


#write remaining hrs in each program split by weather band to a file
#time_remain_p_weatherband(LAPprograms,path_dir)

#optionally print out month/year combos that each source in m16al001 was observed to screen and append to results file
print('\nM16AL001 Tally:\n')
fileo=open(path_dir+'sim_results/results.txt','a')
fileo.write('\nM16AL001 Tally:\n')
for key in m16al001_tally.keys():
	print(key)
	fileo.write('\n'+key+'\n')
	print([(i.datetime.month,i.datetime.year) for i in m16al001_tally[key]])
	fileo.write(",".join(['('+str(i.datetime.month)+','+str(i.datetime.year)+')' for i in m16al001_tally[key]]))
	fileo.write('\n')
fileo.close()

#print finsh dates of program if available
print('\nProgram Finish Dates:\n')
fileo=open(path_dir+'sim_results/results.txt','a')
fileo.write('\nProgram Finish Dates:\n')
for key in finished_dates.keys():
	print(key, ':', finished_dates[key])
	fileo.write('\n'+key+':'+str(finished_dates[key])+'\n')
	fileo.write('\n')
fileo.close()


#print weather band time tally to file and screen
bandslst=['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5']
ascii.write([bandslst,[round(np.sum(wb_usage_tally['Available'][j]),2) for j in bandslst],\
	[round(np.sum(wb_usage_tally['Used'][j]),2) for j in bandslst],\
	[round(np.sum(wb_usage_tally['Unused'][j]),2) for j in bandslst],\
	[round(np.sum(wb_usage_tally['Cal'][j]),2) for j in bandslst]],\
	path_dir+'sim_results/wb_usage_tally.txt',\
	names=['WeatherBand','AvailableTime','UsedTime','UnusedTime','Cal'])
wbtally_vals=ascii.read(path_dir+'sim_results/wb_usage_tally.txt')
print('\nWeather Band Time Tally:\n')
wbtally_vals.sort('WeatherBand')
print(wbtally_vals)


#Create histograms of unused RA per weather band and total tally bar chart
fig=plt.figure()
font={'family':'serif','weight':'bold','size' : '14'}
rc('font',**font)
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
colors=['b','purple','r','orange','g']
bands=['Band 1','Band 2', 'Band 3', 'Band 4', 'Band 5']
bin_start=np.arange(0,25)
bin_end=np.arange(1,26)
ax0=plt.subplot(111)
for i in range(0,len(bands)):
    ax=plt.subplot(3,2,i+1)
    bin_hrs=bin_lst(bin_start,bin_end,lst_tally[bands[i]])
    ax.bar(bin_start+0.5, bin_hrs, width = 1,color=colors[i],alpha=0.6)
    ax.set_xlim(min(bin_start), max(bin_end))
    ax.set_title(bands[i],fontsize=12)
    ax.set_xlabel('${\\rm \\bf LST}$',fontsize=12)
    ax.set_ylabel('${\\rm \\bf Hours}$',fontsize=12,labelpad=20)
    ax.tick_params(axis='x',which='major', labelsize=10,length=5,width=1.5,top='on',bottom='on')
    ax.tick_params(axis='x',which='minor', labelsize=10,length=3.5,width=1,top='on',bottom='on')
    ax.tick_params(axis='y',which='major', labelsize=10,length=5,width=1.5,right='on',left='on')
    ax.tick_params(axis='y',which='minor', labelsize=10,length=3.5,width=1,right='on',left='on')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.get_yaxis().set_label_coords(-0.2,0.5)
fig.subplots_adjust(wspace=0.5,hspace=1)
plt.savefig(path_dir+'sim_results/unused_RA.pdf',bbox_inches='tight')
fig=plt.figure()
font={'family':'serif','weight':'bold','size' : '14'}
rc('font',**font)
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
ax=plt.subplot(111)
ind = np.arange(5)
width = 0.35
p2=ax.bar(ind,wbtally_vals['UnusedTime'],width,color='b',alpha=0.6,edgecolor='k',lw=1.2)
p3=ax.bar(ind,wbtally_vals['UsedTime'],width,bottom=wbtally_vals['UnusedTime'],color='orange',edgecolor='k',lw=1.2)
p4=ax.bar(ind,wbtally_vals['Cal'],width,bottom=wbtally_vals['UsedTime']+wbtally_vals['UnusedTime'],color='r',edgecolor='k',lw=1.2,alpha=0.8)
ax.set_xticks(ind)
ax.set_xticklabels(['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5'])
ax.legend((p2[0],p3[0],p4[0]), ('Unused','Used','Cal'),fontsize=7,loc='top left',ncol=3)#bbox_to_anchor=(1.05,1.4),ncol=3)
ax.tick_params(axis='x',which='major', labelsize=9,length=5,width=1.5,top='off',bottom='on',pad=7)
ax.tick_params(axis='y',which='major', labelsize=10,length=5,width=1.5,right='on',left='on')
ax.tick_params(axis='y',which='minor', labelsize=9,length=3.5,width=1.,right='on',left='on',pad=7)
ax.set_ylabel('${\\rm \\bf Hours}$',fontsize=12)
#ax.get_yaxis().set_label_coords(-0.2,0.5)
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.setp(ax.get_xticklabels(),rotation=45, horizontalalignment='right')
plt.savefig(path_dir+'sim_results/unused_bar.pdf',bbox_inches='tight')

#make a plot of LST of remaining MSBs
jcmt=Observer.at_site("JCMT",timezone="US/Hawaii")
writeLSTremain(jcmt,program_list,sim_end)


end=time.time()
print('Code took: ', (end-start), ' to run')