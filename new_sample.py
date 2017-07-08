import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from Tools.ecg.r_peaks import detect_r_points
from Tools.ecg.Handy_analysis import classify_by_rpoints
from Tools.file.read_sample import read_sample
from Tools.file.read_list import load_file


def new_sample(path, name, sampling_rate,i):
    x_signal, y_signal =read_sample(path, name,sampling_rate, preprocess=False)
    rpeaks = detect_r_points(y_signal,y_signal.shape[0], sampling_rate)
    print('y_shape %d'%y_signal.shape[0])
    print('rpeaks_shape %d'%rpeaks.shape[0])
    label, type = classify_by_rpoints(y_signal, rpeaks, sampling_rate)
    '''
    plot_length = 40000
    trace1 = go.Scatter(y=y_signal[:plot_length], x=x_signal[:plot_length], name='Signal')
    trace2 = go.Scatter(y=y_signal[rpeaks], x=x_signal[rpeaks],mode='markers', name='rpeaks')
    figure = go.Figure(data=[trace1, trace2])
    py.plot(figure, filename=name)
    print('Plotting is done! :)')
    '''
    print('% s Suggested label by simple handy classification = %s' %(name,label))
    #if label == 'Arrhythmic':
    if 1==1:
        trace1 = go.Scatter(y=y_signal[:], x=x_signal[:], name='Signal')
        trace2 = go.Scatter(y=y_signal[rpeaks[:,0]], x=x_signal[rpeaks[:,0]],mode='markers', name='rpeaks')
        layout = go.Layout(title=name)
        figure = go.Figure(data=[trace1, trace2], layout=layout)
        py.plot(figure, filename=name)
        #print('Plotting is done! :)  sampling rate = %s' %sampling_rates[i])
    return label
