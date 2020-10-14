import numpy as np
import time
from clusters import Spike, Cluster
import scipy.io

NUM_BYTES = 2

def get_next_spike(spkFile):
    data = np.zeros((8, 32))
    for i in range(32):
        for j in range(8):
            num = spkFile.read(NUM_BYTES) 
            if not num:
                return None

            data[j, i] = int.from_bytes(num, "little", signed = True) 
    spike = Spike()
    spike.data = data
    return spike   

def get_next_cluster_num(cluFile):
    num = cluFile.readline()
    assert num != ''
    return int(num)

def find_indices_in_filenames(targetName, cellClassMat):
    filenamesArr = cellClassMat['filename'][0][0]

    index = 0
    startIndex = 0
    for filenameArr in filenamesArr:
        filename = filenameArr[0][0] # everything is wrapped in arrays in the mat file for some reason
        if filename == targetName:
            startIndex = index
            break
        index += 1

    for i in range(startIndex, len(filenamesArr)):
        if filenamesArr[i][0][0] != targetName:
            return startIndex, i

    return startIndex, len(filenamesArr)

def find_cluster_index_in_shankclu_vector(startIndex, endIndex, shankNum, cluNum, cellClassMat):
    """print("printing args")
    print(startIndex)
    print(shankNum)
    print(cluNum)"""
    shankCluVec = cellClassMat['shankclu'][0][0]
    for i in range(startIndex, endIndex):
        shankCluEntry = shankCluVec[i]
        if shankCluEntry[0] == shankNum and shankCluEntry[1] == cluNum:
            return i
    return None

def determine_cluster_label(filename, shankNum, cluNum, cellClassMat):
    startIndex, endIndex = find_indices_in_filenames(filename, cellClassMat)
    cluIndex = find_cluster_index_in_shankclu_vector(startIndex, endIndex, shankNum, cluNum, cellClassMat)
    isAct = cellClassMat['act'][0][0][cluIndex][0]
    isExc = cellClassMat['exc'][0][0][cluIndex][0]
    isInh = cellClassMat['inh'][0][0][cluIndex][0]

    if cluIndex == None:
        return -2

    # 0 = PV
    # 1 = Pyramidal
    # -3 = both which means it will be discarded
    # -1 = untagged
    # -2 = clusters that appear in clu file but not in shankclu
    if isExc == 1: 
        if isAct == 1 or isInh == 1: # check if both conditions apply (will be discarded)
            return -3
        
        return 1

    if isAct == 1 or isInh == 1:
            return 0
    return -1


def create_cluster(name, cluNum, shankNum, cellClassMat):
    label = determine_cluster_label(name, shankNum, cluNum, cellClassMat)

    # Check if the cluster doesn't appear in shankclu
    if label == -2:
        return None

    cluster = Cluster()
    cluster.filename = name
    cluster.numWithinFile = cluNum
    cluster.shank = shankNum
    cluster.label = label
    return cluster

def read_directory(path, cellClassMat, i):
    clusters = dict()
    clusters_list = []
    name = path.split("\\")[-1]
    
    start = time.time()
    try:
        spkFile = open(path + "\\" + name + ".spk." + str(i), 'rb')
        cluFile = open(path + "\\" + name + ".clu." + str(i))
    except FileNotFoundError:
        print(path + "\\" + name + ".spk." + str(i) + ' and/or ' + path + "\\" + name + ".clu." + str(i) + ' not found')
        return clusters

    # Read the first line of the cluster file (contains num of clusters)
    get_next_cluster_num(cluFile)
    spike = get_next_spike(spkFile)    
    while(spike is not None):
        cluNum = get_next_cluster_num(cluFile)

        # clusters 0 and 1 are artefacts and noise by convention
        if cluNum == 0 or cluNum == 1:
            spike = get_next_spike(spkFile)  
            continue

        assert cluNum != None
        fullName = name + "_" + str(i) + "_" + str(cluNum) # the format is filename_shankNum_clusterNum

        # Check if cluster exists and create if not
        if fullName not in clusters:
            new_cluster = create_cluster(name, cluNum, (i), cellClassMat)

            # Check to see if the cluster we are trying to create is one that doesn't appear in shankclu (i.e has a label of -2)
            if new_cluster is None:
                spike = get_next_spike(spkFile) 
                continue

            clusters[fullName] = new_cluster
            clusters_list.append(new_cluster)

        clusters[fullName].add_spike(spike)
        spike = get_next_spike(spkFile)
        

    print("Finished File %s with index %d" % (name, i))
    spkFile.close()
    cluFile.close()

    end = time.time()
    print(str(end - start) + " total")

    return clusters_list

def read_all_directories(pathToDirsFile):
    cellClassMat = scipy.io.loadmat("Data\\CelltypeClassification.mat")['sPV']
    dirsFile = open(pathToDirsFile)
    for line in dirsFile:
        split_line = line.split()
        dataDir, remove_inds = split_line[0], split_line[1:] 
        print("reading " + str(dataDir.strip()))
        for i in range(1,5):
            if str(i) in remove_inds:
                print('Skipped shank %d in file %s' % (i, dataDir))
                continue
            dirClusters = read_directory(dataDir.strip(), cellClassMat, i)
            print("the number of clusters is: " + str(len(dirClusters))) 
            yield dirClusters

def main():
    """
    absPath = "E:\\code_david_michal\\2019_07_24_code\\Preprocessin_final\\data\\es25nov11_3"
    cellClassMat = scipy.io.loadmat("CelltypeClassification.mat")['sPV']
    read_directory(absPath, cellClassMat)"""
    clusters = read_all_directories("dirs.txt")
    goodClusters = [clusters[key] for key in clusters if clusters[key].label != -2]
    print(len(goodClusters))

if __name__ == "__main__":
    main()
