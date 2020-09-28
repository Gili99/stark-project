import scipy.io

mat = scipy.io.loadmat('CelltypeClassification.mat')
temp = {'1'}
temp.pop()

ret = []
f = open('dirs.txt', mode = 'w')

for file in mat['sPV'][0][0][0]:
    if file[0][0] not in temp:
        temp.add(file[0][0])
        f.write('Data\\' + file[0][0] + '\n')

f.close()
