import sys
import os
x = sys.argv[1]




y = float(x) ** 2

with open('sim_end_res.csv', 'w') as stream:
    stream.write('sim_end_res') # Column header is always simulation end result 'sim_end_res'
    stream.write('\n') # Need a new line to separate header from value
    stream.write(f'{y}') # Value for 'sim_end_res'

