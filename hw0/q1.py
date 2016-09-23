import sys

args = sys.argv

#column's index
index = int(args[1])

#read file
f = open(args[2], 'r').read().split('\n')

#output list
output_list = []

for i in range(0, len(f)):
	f[i] = f[i].split()
	if len(f[i]) > 0:
		output_list.append(float(f[i][index]))

#sort
output_list = sorted(output_list)
print output_list

#write file
o = open('ans1.txt', 'w')
for i in range(0, len(output_list)-1):
	o.write(str(output_list[i]) + ',')
o.write(str(output_list[len(output_list)-1]))
o.close()	
