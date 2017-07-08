__author__ = 'Zeynab'

from biosppy.signals import ecg
import numpy as np


def detect_r_points(signal, length, rate, algorithm='none'):
    ecg_out = ecg.ecg(signal=signal[:,0], sampling_rate=rate, show=False)
    filtered = ecg_out.__getitem__('filtered')
    temp_rpeaks = ecg_out.__getitem__('rpeaks')

    rpeaks = np.empty([temp_rpeaks.shape[0], 1], dtype=int)
    for i,r in enumerate(temp_rpeaks):
        #find maximum
        rpeaks[i] = r
        max_best_so_far = r
        end = False
        #while not end:
            #end = True
        for j in range(25):
            neighbor1 = max_best_so_far - j
            neighbor2 = max_best_so_far + j
            if abs(signal[neighbor1,0]) > abs(signal[max_best_so_far,0]):
                max_best_so_far = neighbor1
                    #end = False
            if abs(signal[neighbor2,0]) > abs(signal[max_best_so_far,0]):
                max_best_so_far = neighbor2
                    #end = False
        rpeaks[i] = max_best_so_far
    return rpeaks


def analyze_rpoints(y_signal, rpeaks, sampling_rate):
    #print('in analyse  %d'%rpeaks.shape[0])
    problems = np.empty([0, 1])
    #r-r interval
    rr_interval = np.empty([0, 1])
    for j in range(0, rpeaks.shape[0]-1):
        #rr_interval = simple_concatenate(rr_interval,np.array((rpeaks[j + 1] - rpeaks[j])/sampling_rate))
        rr_interval = np.append(rr_interval, np.array((rpeaks[j + 1] - rpeaks[j])/sampling_rate))
        #rr_interval.append((rpeaks[j + 1] - rpeaks[j])/sampling_rate)
    rr_average = np.average(rr_interval)
    rr_variance = np.var(rr_interval)
    #normal_mu, normal_std = norm.fit(normal_rmse)
    #rr_difference = rr_interval - rr_variance -> to calculate r-r avg from a clean and standard part not all of signal
    rr_problems = np.empty([0, 1])
    for j, r in enumerate(rr_interval):
        distance = abs(r - rr_average)
        if distance >= rr_average/2:
            type = 'R-R problem'
            #rr_problems = simple_concatenate(rr_problems, np.array(j))
            #rr_problems = simple_concatenate(rr_problems, np.array(j + 1))
            rr_problems = np.append(rr_problems, np.array([j]))
            #rr_problems = np.append(rr_problems, np.array([j+1]))

    #set threshold for r-r interval variance
    #r peaks amplitude
    r_values = y_signal[rpeaks]
    r_values_average = np.average(abs(r_values))
    r_values_var = np.var(abs(r_values))
    r_values_problems = np.empty([0, 1])
    for j, r in enumerate(r_values):
        distance = abs(abs(r)-abs(r_values_average))
        if distance >= r_values_average/2:
            type = 'R value problem'
            #r_values_problems = simple_concatenate(r_values_problems, np.array(j))
            r_values_problems = np.append(r_values_problems, np.array([j]))
    #problems = rr_problems.append(r_values_problems)

    #problems = simple_concatenate(problems, rr_problems)
    #problems = simple_concatenate(problems, r_values_problems)
    a=set(rr_problems)
    b=set(r_values_problems)
    diff= a-b
    total = list(b) + list(diff)
    problems =np.array(problems)
    #problems = np.append(problems, rr_problems)
    #problems = np.append(problems, r_values_problems)

    #print(problems)
    return problems


