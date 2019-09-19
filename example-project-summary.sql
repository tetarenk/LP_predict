use jcmt;
use omp;
SELECT p.projectid, p.semester, q.tagpriority, (p.remaining/60/60) as remaining_hrs, (p.allocated/60/60) as allocated_hrs,  p.taumin, p.taumax FROM ompproj  as p JOIN ompprojqueue as q ON p.projectid=q.projectid WHERE semester="LAP" AND  remaining >0 ORDER BY q.tagpriority;

