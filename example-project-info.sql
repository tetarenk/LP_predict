use jcmt;

use omp;

SELECT m.projectid, m.msbid, m.remaining, m.obscount, m.timeest, (m.remaining*m.timeest/60/60) as msb_total_hrs,  o.instrument, o.type, o.pol, o.target, o.ra2000, 
o.dec2000,m.taumin,m.taumax FROM ompmsb as m JOIN ompobs as o ON m.msbid=o.msbid WHERE m.projectid="M16AL001" AND m.remaining >=1;

