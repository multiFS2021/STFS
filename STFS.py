#coding=UTF-8
import numpy as np
import skfeature.utility.entropy_estimators as ees

def formual(data,labels,n_features):
    """
    fi=candidate feature  Y=[l1,l2,..lm] f_selected=S(selected subset)
    J=sum(I(fi;lj)+sum[sum(I(f_selected;lj|fi))-sum[sum(I(fi;f_selected;lj))]
    :param data:  numpy.array [n_sample,n_feature]
    :param labels: numpy.array [n_sample,n_classes]
    :param n_features: the number of selected features
    :return:F list
    """
    F = []
    mm, f_nub = np.shape(data)
    nn, l_nub = np.shape(labels)
    n1=1/float(l_nub)
    # nl2=1/float(l_nub-1)
    rel = np.zeros((f_nub, 1))
    for i in range(f_nub):
        for j in range(l_nub):
            rel[i] += ees.midd(data[:, i], labels[:, j])
            for jj in range(l_nub):
                if jj !=j:
                    rel[i] += ees.cmidd(data[:, i],labels[:, jj], labels[:, j])
        rel[i] = n1*rel[i]

    # Selecting First Feature
    idx = np.argmax(rel[:, 0])
    F.append(idx)
    f_select = data[:, idx]
    ffmi1 = np.zeros((f_nub, 1))
    while len(F) < n_features:
        j_cmi = -100000000
        n2 = 1/float(len(F))
        for i in range(f_nub):
            if i not in F:
                for j in range(l_nub):
                    ffmi1[i] += ees.cmidd(data[:, i], labels[:, j], f_select)
                t = rel[i] + n1*n2*ffmi1[i]
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        f_select = data[:, idx]
    return F



