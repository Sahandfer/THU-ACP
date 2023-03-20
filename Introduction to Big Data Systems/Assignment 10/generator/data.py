import random
import math

abc = [	'a','b','c','d','e','f','g',
		'h','i','j','k','l','m','n',
		'o','p','q','r','s','t',
		'u','v','w','x','y','z'
]

DICT_NUM = 1000
LOOP1 = 25
LOOP2 = 40
WORD_NUM = 1000

words = ['' for i in range(DICT_NUM)]
for i in range(LOOP1):
	for j in range(LOOP2):
		words[i*LOOP2+j] = ('%s%d' % (abc[i], j))

#print(words)
#print(words[50])
#print(len(words))

r = [random.gauss(2, 0.5) for i in range(WORD_NUM)]
#print(r)
#print(len(r))

r_max = -1000000
r_min = 1000000
for i in range(WORD_NUM):
	if r[i]>r_max:
		r_max = r[i]
	if r[i]<r_min:
		r_min = r[i]
r_range = r_max - r_min
r_range = 1.001 * r_range
#print(r_max, r_min, r_range)

#have = [0 for i in range(DICT_NUM)]
#print(have)
#print(len(have))

r_sum = 0
for i in range(WORD_NUM):
	r[i] = int(math.floor((r[i]-r_min)/r_range*DICT_NUM))
	if r[i]>WORD_NUM-1:
		print('r>%d' % WORD_NUM-1)
		r[i] = WORD_NUM-1
	r_sum += r[i]
	#have[r[i]] = 1
r_mean = r_sum / WORD_NUM
#print(r_sum, r_mean)
#print(words[r_mean])

'''
have_num = 0
for i in range(len(have)):
	if have[i]==1:
		have_num += 1
print(have_num)

r_max = -1000000
r_min = 1000000
for i in range(WORD_NUM):
	if r[i]>r_max:
		r_max = r[i]
	if r[i]<r_min:
		r_min = r[i]
r_range = r_max - r_min
print(r_max, r_min, r_range)
'''

fns = open('words.txt','w')
for i in range(200):
	fns.write(str(words[r[5*i]])+' '+str(words[r[5*i+1]])+' '+str(words[r[5*i+2]])+' '+str(words[r[5*i+3]])+' '+str(words[r[5*i+4]])+'\n')
fns.close()
