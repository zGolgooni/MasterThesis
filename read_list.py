__author__ = 'Zeynab'
import csv
import numpy as np


def load_file(path, file):
    ids = []
    paths = []
    names = []
    sampling_rates = []
    labels = []
    explanations = []
    partitions = []
    intervals = []
    with open(path + file) as csvfile:
        readCSV = csv.reader(csvfile)
        next(readCSV)
        for row in readCSV:
            id = row[0]
            path = row[1]
            name = row[2]
            sampling_rate = row[3]
            label = row[4]
            explanation = row[5]
            partition = row[6]
            interval = row[7]
            ids.append(id)
            paths.append(path)
            names.append(name)
            sampling_rates.append(int(sampling_rate))
            labels.append(label)
            partitions.append(partition)
            intervals.append(interval)
            if explanation.__contains__('Tachicardia'):
                explanations.append('Tachicardia')
            elif explanation.__contains__('Bradicardia'):
                explanations.append('Bradicardia')
            elif explanation.__contains__('Long QT'):
                explanations.append('Long QT')
            else:
                explanations.append(explanation)
    print(file + 'is read!')
    return ids, paths, names, sampling_rates, labels, explanations, partitions, intervals


def split_samples(file, path, fraction=0.15):
    #train = []
    #test = []
    Normal = []

    Bradi = []
    Tachi = []
    Poly = []
    LQT = []
    Arrest = []

    ids, paths, names, sampling_rates, labels, explanations,partitions, intervals = load_file(path, file)
    for i, l in enumerate(labels):
        if l == 'Normal':
            Normal.append(i)
        else:
            if explanations[i].__contains__('Long QT'):
                LQT.append(i)
            elif explanations[i].__contains__('Bradicardia'):
                Bradi.append(i)
            elif explanations[i].__contains__('Tachicardia'):
                Tachi.append(i)
            elif explanations[i].__contains__('Polymorphic'):
                Poly.append(i)
            elif explanations[i].__contains__('Arrest'):
                Arrest.append(i)
    total = list(range(0, len(labels)))
    print('total = %d , Normal=%d, LQT=%d, Tachi=%d, Bradi=%d, Poly=%d, Arrest=%d'%(len(total), len(Normal), len(LQT),len(Tachi),len(Bradi),len(Poly),len(Arrest)))

    test_Normal = np.random.choice(Normal, size=int(len(Normal) * fraction), replace=False)

    test_lQT = np.random.choice(LQT, size=max(1,int(len(LQT) * fraction)), replace=False)
    test_Tachi = np.random.choice(Tachi, size=max(1,int(len(Tachi) * fraction)), replace=False)
    test_Bradi = np.random.choice(Bradi, size=max(1,int(len(Bradi) * fraction)), replace=False)
    test_Poly = np.random.choice(Poly, size=max(1,int(len(Poly) * fraction)), replace=False)
    test_Arrest = np.random.choice(Arrest, size=max(1,int(len(Arrest) * fraction)), replace=False)


    test1 = np.append(test_lQT,test_Poly)
    test2 = np.append(test_Bradi,test_Tachi)
    test = np.append(test1,test2)
    test = np.append(test,test_Arrest)

    test = np.append(test,test_Normal)
    print('test = %d , Normal=%d, LQT=%d, Tachi=%d, Bradi=%d, Poly=%d, Arrest=%d'%(len(test), len(test_Normal), len(test_lQT),len(test_Tachi),len(test_Bradi),len(test_Poly),len(test_Arrest)))

    train = list(set(total) - set(test))
    return train, test


def load_partitions(file, path):
    ids, paths, names, sampling_rates, labels, explanations,partitions, intervals = load_file(path, file)
    train_ids = []
    test_ids = []
    for i, p in enumerate(partitions):
        if p == 'Train':
            train_ids.append(i)
        else:
            test_ids.append(i)
    return train_ids, test_ids
