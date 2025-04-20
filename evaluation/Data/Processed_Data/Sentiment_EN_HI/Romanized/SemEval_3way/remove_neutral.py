def rem(filename):
	fn = '../SemEval_2way/'+filename+'_2.txt'
	print(fn)
	o1 = open(fn, 'w+')
	i = 0
	with open(filename+'.txt', 'r') as f:
		for l in f:
			# print(l)
			x = (l.strip()).split('\t')
			# print(x)
			if x[1]=="neutral":
				i+=1
			else:
				o1.write(l)
	o1.close()
	print(i)

def rem_test(filename1, filename2):
	fn1 = '../SemEval_2way/'+filename1+'_2.txt'
	fn2 = '../SemEval_2way/'+filename2+'_2.txt'
	print(fn1, fn2)
	o1 = open(fn1, 'w+')
	o2 = open(fn2, 'w+')
	i = 0
	with open(filename1+'.txt', 'r') as textfile1, open(filename2+'.txt', 'r') as textfile2: 
	    for x, y in zip(textfile1, textfile2):
	        if y.strip()=="neutral":
	        	i+=1
	        else:
	        	o1.write(x)
	        	o2.write(y)
	        # print("{0}\t{1}".format(x, y))
	print(i)
	o1.close()
	o2.close()

# rem('validation')
# rem('all')
# rem('train')
rem_test('test', 'testlabel')
