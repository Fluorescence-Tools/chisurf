import numpy as np
from scipy.cluster import hierarchy as hclust
from scipy.cluster.hierarchy import fcluster
from mfm.structure import rmsd, average, find_best


def findSmallestCluster(clusters):
    print("findSmallestCluster")
    minCl = list(clusters.keys())[0]
    for clName in clusters:
        if len(clusters[clName]) < len(clusters[minCl]):
            minCl = clName
    return minCl


def cluster(structures, threshold=5000, criterion='maxclust', Z=None, distances=None, directory=None):
    # http://www.mathworks.de/de/help/stats/hierarchical-clustering.html
    print("Performing cluster-analysis")
    k = 0
    #start_time = time.time()
    nStructures = len(structures)
    if distances is None:
        distances = np.empty(nStructures * (nStructures - 1) / 2)
        for i in range(nStructures):
            for j in range(i + 1, nStructures):
                distances[k] = rmsd(structures[j], structures[i])
                k += 1
            m = (nStructures * nStructures - 1) / 2
            print('RMSD computation %s/%s : %.1f%%' % (k, m, float(k) / m * 100.0))
        if directory is not None:
            print("Saving distance-matrix")
            np.save(directory + '/' + 'clDistances.npy', distances)

    print('mean pairwise distance ', np.mean(distances))
    print('stddev pairwise distance', np.std(distances))

    if Z is None:
        # run hierarchical clustering on the distance matrix
        print('\n\nRunning hierarchical clustering (UPGMA)...')
        Z = hclust.linkage(distances, method='average', preserve_input=True)
        # get flat clusters from the linkage matrix corresponding to states
        if directory is not None:
            print("Saving cluster-results")
            np.save(directory + '/' + 'clLinkage.npy', Z)

    print('\n\nFlattening the clusters...')
    assignments = fcluster(Z, t=threshold, criterion=criterion)
    cl = dict()
    for c in np.unique(assignments):
        cl[c] = []
    for i, a in enumerate(assignments):
        cl[a] += [i]
        #print "Needed time: %.3f seconds" % (time.time() - start_time)
    print('Number of clusters found', len(np.unique(assignments)))
    return Z, cl, assignments, distances


def find_representative(trajectory, cl):
    """
    :param trajectory: a list of structures
    :param c: a list of numbers (positions in structures) belonging to one cluster
    :return: index of representative structure of cluster
    """
    structuresInCluster = [trajectory[i] for i in cl]
    averageStructureInCluster = average(structuresInCluster)
    idx, representativeStructureInCluster = find_best(averageStructureInCluster, structuresInCluster)
    idxOfRepresentativeStructure = cl[idx]
    return idxOfRepresentativeStructure