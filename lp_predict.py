########################################
#JCMT Large Program Prediction Tool
########################################
'''Python script that predicts how many hours of each JCMT
Large Program will be observed over the upcoming semester (or upcoming years).

INPUT: (1) sim_start: proposed start date 
	   (2) sim_end: propsed end date
	   (3) Blocked out dates for each instrument
	   (4) Large program details file (generated by sql scripts)
	   (5) MSBs file for each program (generated by sql scripts)
	   (6) Observing blocks for large programs file (with start/end dates of each scheduled program block)

OUTPUT: (1) File summary of results; detailing the predicted number of hours observed for each project,
        and the hours predicted to be still remaining in each program.
        (2) Dictionary keeping track of unused time in each weather band.

NOTES: - If you are running this script on an EAO computer, you can use the 'fetch' option to get the wvm data.
Otherwise you must provide a wvm data file (csv format,4 columns:isoTime,mean,median,count) obtained from running
the partner script, getwvm.py, on an EAO computer.
- Uses the following python packages: astropy,astroplan

Written by: Alex J. Tetarenko
Last Updated: Jan 28, 2019
'''

#packages to import
import numpy as np
import math as ma
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time
import warnings
warnings.filterwarnings('ignore')
from astroplan import Observer, FixedTarget, AltitudeConstraint, is_observable, ObservingBlock, observability_table
from astroplan.constraints import TimeConstraint
from astroplan.scheduling import PriorityScheduler, Schedule, Transitioner
from astroplan.plots import plot_schedule_airmass
import os
from collections import defaultdict
from getwvm import get_wvm_fromdisk, get_sampled_values

def correct_msbs(LAPprograms,path_dir):
	'''Corrects and simplifies MSB files to ensure that the total time of all MSBs matches the
	total allocated time remaining, and that there are no duplicate target sources.'''
	program_list=np.array(LAPprograms['projectid'])
	for m in program_list:
		print 'Correcting MSBs file for: ',m
		msbs=ascii.read(path_dir+'program_details_org/'+m.lower()+'-project-info.list')
		remaining=LAPprograms['remaining_hrs'][np.where(LAPprograms['projectid']==m)[0][0]]
		msb_remaining=np.sum(msbs['msb_total_hrs'])
		diff= remaining - msb_remaining #negative is too much, + is too little
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
				hrs_add=int(abs(diff)/np.max((msbs['timeest'][0]/3600.)))
				remain=np.array(msbs['remaining'])
				while hrs_add >0:
					for i in range(0,len(msbs['target'])):
						if hrs_add>0:
							remain[i]=remain[i]+np.sign(diff)
						else:
							remain[i]=remain[i]
						hrs_add=hrs_add-1
				ascii.write([msbs['projectid'],msbs['msbid'],remain,msbs['obscount'],\
				msbs['timeest'],(msbs['timeest']*remain)/3600.,msbs['instrument'],\
				msbs['type'],msbs['pol'],msbs['target'],msbs['ra2000'],msbs['dec2000'],\
				msbs['taumin'],msbs['taumax']],\
				path_dir+'program_details_fix/'+m.lower()+'-project-info.list', names=msbs.colnames)

def transform_blocks(blocks_file):
	'''Reads in observing blocks data file. We make sure to properly deal with the irregular observing blocks data file,
	 which has inconsistent columns.'''
	newfile=blocks_file.strip('.txt')+'_corr.txt'
	f=open(newfile,'w')
	with open(blocks_file, 'r') as ins:
		array = []
		for line in ins:
			linecode=line.strip('\n').split(' ')
			if len(linecode)==4:
				f.write('{0} {1} {2} {3}\n'.format(linecode[0],linecode[1],linecode[2],linecode[3]))
			else:
				f.write('{0} {1} {2} {3}\n'.format(linecode[0],linecode[1],linecode[2],'none'))
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
	'''Reads calibrator data files'''
	cal=table['target name'][np.where(table['fraction of time observable']==np.max(table['fraction of time observable']))[0][0]]
	ind=np.where(table['target name']==cal)[0][0]
	ra=str(cal_table['col2'][ind])+'h'+str(cal_table['col3'][ind])+'m'+str(cal_table['col4'][ind])+'s'
	dec=str(cal_table['col5'][ind])+'d'+str(cal_table['col6'][ind])+'m'+str(cal_table['col7'][ind])+'s'
	return cal,ra,dec

def pick_cals(day,obsn,path_dir):
	'''Finds calibrators that are best observable on a particular night'''
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

def get_cal_times(obsn,day,half):
	'''Creates 15min time-blocks for calibrator sources each hour of the night,
	giving 25% of observing night to calibrators'''
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

def get_wvm_data(sim_start,sim_end,flag,path_dir,wvmfile=''):
	'''Get WVM weather data'''
	sim_years=int(ma.ceil(abs((Time(sim_start,format='iso').datetime-Time(sim_end,format='iso').datetime).total_seconds())/(3600.*24*365.)))
	if flag=='fetch':
		hoursstart=4#6pmHST
		hoursend=16#6amHST
		prev_years=Time(sim_start,format='iso').datetime.year-sim_years
		prev_yeare=Time(sim_end,format='iso').datetime.year-sim_years
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
	tau_predict.fill_value = 0.
	tau_predict_fill=tau_predict.filled()
	return(mjd_predict,tau_predict_fill)

def good_blocks(Blocks,mjd_predict,tau_predict):
	'''Get MJDs and weather for the observing blocks'''
	start=Blocks['date_start']
	end=Blocks['date_end']
	startmjd=Time(str(start)[0:4]+'-'+str(start)[4:6]+'-'+str(start)[6:8], format='iso', scale='utc').mjd
	endmjd=Time(str(end)[0:4]+'-'+str(end)[4:6]+'-'+str(end)[6:8], format='iso', scale='utc').mjd
	dates=np.arange(startmjd,endmjd+1,1)
	obs_mjd=mjd_predict[[i for i, item in enumerate(mjd_predict) if item in dates]]
	tau_mjd=tau_predict[[i for i, item in enumerate(mjd_predict) if item in dates]]
	return(obs_mjd,tau_mjd)

def bad_block(instrument,SCUBA_2_unavailable,HARP_unavailable,UU_unavailable):
	'''Returns the proper list of unavailable dates based on instrument'''
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


def predict_time(sim_start,sim_end,wvmfile,LAPprograms,Block,path_dir,flag,total_observed,FP,m16al001_tally,\
	SCUBA_2_unavailable,HARP_unavailable,RU_unavailable,unused_tally):
	'''Simulates observations of Large Programs over specified observing block'''

	#fetch wvm data from previous year(s)
	mjd_predict,tau_predict=get_wvm_data(sim_start,sim_end,flag,path_dir,wvmfile)

	#get dates for observing block, make arrays of MJD and tau for these days
	obs_mjd,tau_mjd=good_blocks(Block,mjd_predict,tau_predict)

	#set up observatory site and general elevation constraints
	jcmt=Observer.at_site("JCMT",timezone="US/Hawaii")
	constraints = [AltitudeConstraint(30*u.deg, 80*u.deg,boolean_constraint=True)]

	print 'block:',obs_mjd
	#loop over all days in the current observing block
	for k in range(0,len(obs_mjd)):
		print 'day:',obs_mjd[k]
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
		#make a target list for the night (we keep track of target, MSB time, priority, and program),
		#looping over each target in each program
		targets=[]
		priority=[]
		msb_time=[]
		prog=[]
		tc=[]
		for m in LAPprograms['projectid']:
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
								#targets in the program that is assigned the current block get highest priority (2), except for calibrators below (1)
								if m.lower() in FP:
									priority.append(2)
								else:
									#other targets get asssigned overall program priority
									priority.append(LAPprograms['tagpriority'][np.where(LAPprograms['projectid']==m)[0]])
								msb_time.append(obs_time_table['timeest'][j]*u.second)
								prog.append(m.lower())
						else:
							#we keep track of the dates each target in the m16al001 program is observed through the m16al001_tally dictionary,
							#so we need to first check if the target is present in the dictionary yet
							dates_obs=[all(getattr(Time(obs_mjd[k], format='mjd', scale='utc').datetime,x)==getattr(mon.datetime,x) for x in ['year','month']) for mon in m16al001_tally[target_table['target'][j]]]
							if target_table['target'][j] not in m16al001_tally.keys():
								targets.append(FixedTarget(coord=SkyCoord(ra=target_table['ra2000'][j]*u.rad,\
									dec=target_table['dec2000'][j]*u.rad),name=target_table['target'][j]))
								tc.append(TimeConstraint(time_range[0], time_range[1]))
								if m.lower() in FP:
									priority.append(2)
								else:
									priority.append(LAPprograms['tagpriority'][np.where(LAPprograms['projectid']==m)[0]])
									msb_time.append(obs_time_table['timeest'][j]*u.second)
									prog.append(m.lower())
							else:
								#then if present, check if the target has been observed in the current nights month/year combo yet
								if not any(dates_obs):
									targets.append(FixedTarget(coord=SkyCoord(ra=target_table['ra2000'][j]*u.rad,\
										dec=target_table['dec2000'][j]*u.rad),name=target_table['target'][j]))
									tc.append(TimeConstraint(time_range[0], time_range[1]))
									if m.lower() in FP:
										priority.append(2)
									else:
										priority.append(LAPprograms['tagpriority'][np.where(LAPprograms['projectid']==m)[0]])
										msb_time.append(obs_time_table['timeest'][j]*u.second)
										prog.append(m.lower())
		#check if at least one target has been added to our potential target list
		if len(targets)>0:
			#check at least one potential target is observable at some point in the night
			ever_observable = is_observable(constraints, jcmt, targets, time_range=time_range)
			#pick a HARP, SCUBA-2, and pointing calibrator for each half-night, and add to target list
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
						prog.append('Pointing/Focus cal')							
		else:
			ever_observable = False
		#As long as we have an observable target, we proceed to scheduling for the night.
		if np.any(ever_observable):
			#set the slew rate of telescope between sources
			slew_rate = 1.2*u.deg/u.second
			transitioner = Transitioner(slew_rate)#, {'program':{'default':30*u.second}})
			# set up the schedule for the night
			prior_scheduler = PriorityScheduler(constraints = constraints,observer = jcmt,\
				transitioner = transitioner)
			priority_schedule = Schedule(time_range[0], time_range[1])
			#create observing blocks for each target based on target list made above (these act just like normal MSBs at JCMT)
			bnew=[]
			for i in range(0,len(targets)):
				bnew.append(ObservingBlock(targets[i],msb_time[i], priority[i], configuration={'program' : prog[i]},\
							constraints=[tc[i]]))
			#run the astroplan priority scheduling tool
			prior_scheduler(bnew, priority_schedule)
			sched=priority_schedule.to_table(show_unused=True)
			#keep track of unused time
			unused=np.sum(np.array(sched['duration (minutes)'][np.where(sched['target']=='Unused Time')[0]]))/60.
			WBand=get_wband(tau_mjd[k])
			unused_tally[WBand].append(unused)
			#caltime=np.sum(np.array(sched['duration (minutes)'])[[p for p,n in enumerate(np.array(sched['target'])) if n in names]])/60.
			#cal_tally.append(caltime)
			
			#FOR TESTING ONLY--
			#
			#print unused,caltime
			#plt.figure(figsize = (14,6))
			#plot_schedule_airmass(priority_schedule)
			#plt.legend(loc = "upper right",ncol=3)
			#plt.show()
			#raw_input('stop')

			#record what targets have been observed, updating the MSB files and recording total time observed for each program
			for h in range(0,len(np.unique(sched['target']))):
				tar=np.unique(sched['target'])[h]
				if (tar not in ['TransitionBlock','Unused Time'] and 'CAL' not in tar):
					prog=sched['configuration'][np.where(sched['target']==tar)[0][0]]['program']
					tim_used=np.sum(np.array(sched['duration (minutes)'][np.where(sched['target']==tar)[0]]))/60.
					num_used=len(np.array(sched['duration (minutes)'][np.where(sched['target']==tar)[0]]))
					if tim_used >0:
						msbs=ascii.read(path_dir+'program_details_sim/'+prog+'-project-info.list')
						r0=msbs['remaining'][np.where(msbs['target']==tar)[0]]-num_used
						oc0=msbs['obscount'][np.where(msbs['target']==tar)[0]]+num_used
						msbs['remaining'][np.where(msbs['target']==tar)[0]]=r0
						msbs['obscount'][np.where(msbs['target']==tar)[0]]=oc0
						ascii.write([msbs['projectid'],msbs['msbid'],msbs['remaining'],msbs['obscount'],\
						msbs['timeest'],msbs['msb_total_hrs'],msbs['instrument'],
						msbs['type'],msbs['pol'],msbs['target'],msbs['ra2000'],msbs['dec2000'],\
						msbs['taumin'],msbs['taumax']],\
						path_dir+'program_details_sim/'+prog+'-project-info.list', names=msbs.colnames)
						total_observed[prog.upper()]=total_observed[prog.upper()]+tim_used
						if prog=='m16al001':
							 m16al001_tally[tar].append(Time(obs_mjd[k],format='mjd',scale='utc'))
		else:
			WBand=get_wband(tau_mjd[k])
			unused_tally[WBand].append(unused)
	return total_observed,m16al001_tally,unused_tally

###########################
#User Input
###########################
path_dir='/Users/atetarenko/Desktop/Support_Scientist_Work/LargeP_predict_model/'
#path_dir='/export/data2/atetarenko/LP_predict/'

sim_start='2019-01-01'
sim_end='2019-08-01'

flag='file'#'file' or 'fetch' if you are on EAO computer
wvmfile=path_dir+'wvmvalues_onepernight.csv'

LAPprograms_file=path_dir+'LP_priority.txt'
blocks_file=path_dir+'LAP-UT-blocks.txt'

#dates the instruments are unavailable in MJD
SCUBA_2_unavailable=[]
HARP_unavailable=[]
RU_unavailable=np.arange(58484,58574)
###########################


#read in large program details and correct MSBs
LAPprograms=ascii.read(LAPprograms_file)
LAPprograms.sort('tagpriority')
print 'Current Large programs:\n'
print LAPprograms, '\n'
RH=LAPprograms['projectid','remaining_hrs','allocated_hrs']
program_list=np.array(LAPprograms['projectid'])
#correct MSBs
correct_msbs(LAPprograms,path_dir)
#empty out simulations folder and add current program files
os.system('rm -rf '+path_dir+'program_details_sim/*.list')
os.system('cp -r '+path_dir+'program_details_fix/*.list '+path_dir+'program_details_sim')

print 'Predicting Large Program observations between '+sim_start+' and '+sim_end+' ...\n'

#calculate observing blocks within the selected simulation dates
Blocks=transform_blocks(blocks_file)
OurBlocks,firstprog=calc_blocks(Blocks,sim_start,sim_end)

#run observation simulateo for each observing block
total_observed = {k:v for k,v in zip(program_list,np.zeros(len(program_list)))}
m16al001_tally=defaultdict(list)
unused_tally=defaultdict(list)
for jj in range(0,len(OurBlocks)):
	FP=firstprog[jj]
	total_observed,m16al001_tally,unused_tally,cal_tally,tot_tally=predict_time(sim_start,sim_end,wvmfile,LAPprograms,OurBlocks[jj],path_dir,flag,total_observed,FP,m16al001_tally,SCUBA_2_unavailable,HARP_unavailable,RU_unavailable,unused_tally)

#calculate final results
obs_hrs=[]
remaining_new=[]
for i in range(0,len(program_list)):
	remaining_new.append(round(RH['remaining_hrs'][np.where(RH['projectid']==program_list[i].upper())[0][0]]-round(total_observed[program_list[i].upper()],2),2))
	obs_hrs.append(round(total_observed[program_list[i].upper()],2))

#write final results to a file and screen
ascii.write([RH['projectid'],remaining_new,RH['allocated_hrs'],obs_hrs],\
	path_dir+'sim_results/results.txt', names=['projectid','remaining_hrs','allocted_hrs','sim_obs_hrs'])
new=ascii.read(path_dir+'sim_results/results.txt')
print 'Final Prediction Results...\n'
print new

#optionally print out month/year combos that each source in m16al001 was observed
#print 'M16AL001 Tally:\n'
#for key in m16al001_tally.keys():
	#print key
	#print [(i.datetime.month,i.datetime.year) for i in m16al001_tally[key]]

#print unused time tally to file and screen
ascii.write([unused_tally.keys(),[round(np.sum(unused_tally[i]),2) for i in unused_tally.keys()]],path_dir+'sim_results/unused_tally.txt',\
	names=['WeatherBand','UnusedTime'])
unused_vals=ascii.read(path_dir+'sim_results/unused_tally.txt')
print 'Unused Time Tally:\n'
unused_vals.sort('WeatherBand')
print unused_vals
