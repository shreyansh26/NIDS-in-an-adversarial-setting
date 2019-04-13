f = open("./copy_file.txt", "r")

a = []
for l in f:
	a.append(l.strip().replace('\n', '')

f = open("./copy_file.txt", "w")
f.write(''.join(a))
f.close()
