import rlss
inf = float('inf')
out = rlss.total_trip_time([[1,2,3,2,1],[1,3]],[2,1],[[0,2,4,inf,0],[0,0,3,0,inf],[0,0,0,2,4],[0,0,0,0,3],[0,0,0,0,0]], True)
print(out)
