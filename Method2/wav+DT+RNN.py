__author__ = 'Zeynab'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LSTM, Conv1D, MaxPooling1D, Flatten,SimpleRNN
from keras.layers.core import Masking
from keras.preprocessing import sequence
import pywt
from sklearn.decomposition import PCA, FastICA
from sklearn import svm, tree
import numpy as np
from Tools.file.read_list import load_file, split_samples, load_partitions
from CNN.train_data import load_data_new, load_sample

num_experiments = 5

train_results = []
test_results = []

train_tp = []
train_tn = []
train_fp = []
train_fn = []
train_acc = []
train_sens = []
train_prec = []
#train_fn = np.empty([num_experiments,1])

test_tp = []
test_tn = []
test_fp = []
test_fn = []
test_acc = []
test_sens = []
test_prec = []

train_fractions = []
test_fractions = []

#part1
train_fractions.append(np.array([2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 60, 63, 64, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 120, 121, 122, 124, 125, 126, 127, 128, 130, 132, 134, 135, 136, 137, 138, 139, 140, 141, 143, 146, 147, 148, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 175, 176, 177, 178, 180, 181, 182, 184, 185, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 212, 214, 216, 217, 218, 219, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 250, 252, 253, 254, 256, 257, 260, 261, 262, 263, 265, 266, 268, 269, 270, 271, 272, 273, 275, 276, 277, 278, 279, 281, 284, 286, 288, 289, 291, 292, 293, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 307, 309, 310, 311, 312, 313, 314, 315, 316, 319, 320, 321, 322, 323, 324, 326, 327, 328, 329, 330, 332, 333, 334, 335, 337, 338, 339, 340, 341, 346, 347, 348, 351, 353, 354, 355, 356, 357, 358, 364, 365, 366, 367, 368, 369, 370, 373, 374, 375, 376, 378]))
test_fractions.append(np.array([ 40, 264, 363, 188,  46, 361, 345, 274, 285, 362, 267, 235, 144,
        61, 215, 149,  96,  23,  24, 142, 101,  89,  62,  59, 283,  37,
       306, 349, 318, 379, 317,  71, 210, 350, 336, 174, 308, 172, 331,
       179, 259, 133, 258, 241, 131, 129, 372, 255, 371, 248, 107, 249,
       251,  11, 119,  27, 287,  30, 344, 359,   9, 325, 352, 282, 280,
       152, 116, 360, 123, 233, 305, 221,  84, 145, 294, 343,   1,  65,
         8, 290,   0,  56, 220, 377,  70, 213,  20,  58, 342, 207,  82,
       223, 183, 211]))
#part2
train_fractions.append(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 41, 43, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 64, 67, 69, 71, 73, 74, 77, 78, 80, 81, 82, 84, 85, 87, 88, 90, 91, 92, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 129, 130, 131, 132, 133, 135, 136, 137, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 162, 163, 164, 167, 168, 170, 171, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 218, 219, 220, 221, 222, 223, 226, 227, 229, 230, 232, 234, 235, 237, 239, 240, 241, 243, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 259, 260, 262, 264, 266, 267, 268, 269, 271, 272, 273, 274, 277, 280, 281, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 302, 303, 304, 306, 307, 308, 309, 310, 312, 314, 315, 316, 318, 321, 322, 323, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 339, 340, 342, 344, 345, 346, 347, 348, 349, 350, 351, 354, 355, 356, 360, 362, 363, 365, 366, 367, 368, 369, 372, 373, 375, 376, 378, 379]))
test_fractions.append(np.array([40, 263, 270,  49,  44, 265, 361,  42, 187, 358, 364, 217, 139,
       216, 138,  72,  96,  89, 161,  86, 320, 144,  68,  61, 236, 317,
       338, 337,  66, 319, 327, 313, 172, 357, 169,  93, 166,  38, 311,
       159, 126, 371, 238, 370, 128, 242, 253, 198, 108, 261, 134, 107,
       244,  11, 305,  12,  79,  53, 201, 282, 231,  10, 203, 225, 106,
       324, 228,  63,  20, 165,  76, 224,  83, 341,  57, 374, 352, 278,
       233,  70, 377, 276, 301, 343,  28,  75, 296, 177, 279, 353,  65,
       359, 214, 275]))

#part3
train_fractions.append(np.array([0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 15, 17, 20, 23, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 70, 71, 72, 73, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 123, 124, 125, 127, 128, 131, 133, 134, 135, 136, 137, 138, 140, 141, 143, 144, 145, 147, 149, 150, 152, 153, 155, 156, 157, 158, 159, 161, 162, 163, 164, 165, 166, 167, 168, 170, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 192, 196, 197, 199, 200, 203, 204, 206, 207, 208, 211, 212, 214, 215, 216, 217, 219, 221, 223, 224, 225, 226, 227, 228, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 248, 249, 250, 251, 252, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 268, 269, 270, 272, 273, 275, 276, 277, 278, 280, 282, 284, 285, 286, 288, 289, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 308, 309, 311, 313, 315, 316, 318, 319, 321, 322, 323, 324, 326, 328, 329, 331, 334, 335, 338, 339, 340, 341, 342, 347, 348, 349, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 367, 368, 369, 370, 371, 372, 373, 374, 375, 377, 378, 379]))
test_fractions.append(np.array([366, 266, 271,  54, 274, 190, 267,  50, 345,  41, 346,  68,  60,
       160, 320,  25, 142,  21, 283, 148,  74,  22,  24, 139, 146, 355,
       210, 171, 327, 330, 169, 332, 350, 336, 317, 337, 314, 195, 310,
       312, 244,  18, 253, 246, 247, 130, 198, 245, 151, 132, 258, 129,
       126,  11, 290, 116,  13, 220, 193, 325, 229, 154, 376, 287, 222,
       291,  14, 209, 201, 202, 279, 218, 205, 105, 301, 344, 194,   3,
        99, 343, 333,  27, 231, 281,  16,  77,  69, 213, 177,   6, 122,
        53,  64,  19]))

#part4
train_fractions.append(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 18, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66, 67, 69, 70, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 87, 88, 90, 91, 92, 94, 96, 97, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 113, 114, 115, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 168, 169, 170, 171, 172, 174, 175, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 192, 194, 197, 198, 199, 200, 201, 203, 204, 205, 207, 208, 209, 210, 212, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 228, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 261, 263, 265, 266, 267, 268, 269, 270, 272, 273, 275, 276, 277, 278, 280, 281, 282, 284, 285, 286, 287, 288, 289, 290, 294, 295, 296, 298, 299, 300, 301, 302, 307, 308, 310, 312, 313, 314, 315, 316, 317, 318, 319, 320, 322, 323, 325, 326, 328, 329, 330, 331, 332, 335, 336, 337, 340, 342, 343, 344, 346, 347, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 363, 364, 366, 367, 368, 370, 371, 373, 374, 376, 378]))
test_fractions.append(np.array([ 50, 271, 365,  40, 264, 190, 274, 345,  44,  41,  54, 163, 293,
       141, 283,  72, 321,  60,  86,  39,  95,  68, 103, 191,  89, 195,
       311,  71, 348, 176, 349, 379, 350, 306, 327, 167,  93, 173, 334,
       338, 182, 262, 241, 260, 369, 240, 196, 244, 372, 133, 246, 255,
       245,  11, 225, 116, 206, 202,  16, 305, 309,  98, 110, 214,  20,
       375, 377,  78, 303, 154, 193,  17,  79,  73,  65, 211,  14, 304,
       360, 226, 229, 341, 111, 292, 291, 279, 118, 339, 213, 297,  27,
       324, 227, 333]))

#part5
train_fractions.append(np.array([0, 2, 3, 4, 5, 8, 9, 10, 11, 13, 15, 16, 17, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 37, 40, 42, 43, 44, 45, 47, 48, 49, 52, 55, 57, 58, 60, 61, 62, 64, 65, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 98, 100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 124, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137, 139, 140, 141, 142, 143, 144, 146, 147, 148, 150, 151, 154, 155, 156, 157, 158, 162, 163, 164, 166, 167, 168, 169, 170, 171, 172, 177, 179, 180, 181, 182, 184, 185, 186, 187, 189, 191, 193, 195, 196, 197, 198, 199, 201, 204, 205, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 231, 233, 234, 235, 236, 237, 239, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 257, 259, 260, 262, 263, 264, 265, 266, 268, 269, 271, 272, 273, 274, 275, 276, 279, 280, 281, 283, 284, 285, 287, 288, 289, 290, 291, 293, 294, 295, 296, 299, 301, 302, 303, 304, 305, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 322, 323, 324, 326, 329, 330, 332, 334, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 352, 353, 354, 355, 356, 359, 360, 362, 363, 364, 365, 366, 367, 369, 370, 373, 374, 375, 377, 379]))
test_fractions.append(np.array([361, 358,  50,  54,  46,  51, 270, 190, 267,  41, 188,  59, 138,
       368,  97, 321,  88,  23,  39, 351,  21, 161, 160, 378, 149, 176,
       174, 357, 192, 328,  71, 173, 331, 327, 336, 159,  35, 178,  38,
        66, 240,  18, 261, 371, 200, 238, 372, 126, 135, 256, 242, 107,
       258,  32, 152, 153, 277, 286,   1, 232, 165, 223, 202, 333, 123,
       300,  99,  53, 292,  63, 282, 212, 376, 183,   6,  12, 309,  56,
       175,  14, 194, 218, 145, 297, 325, 114, 278, 298, 230,  79, 206,
       203, 125,   7]))

'''
main_path = '/Users/Zeynab/'
#main_file = 'My data/In use/Data_v960213.csv'
main_file = 'My data/In use/Data_v960412.csv'
'''

main_path = '/home/mll/Golgooni/'
main_file = 'My data/In use/Data_v960412.csv'


ids, paths, names, sampling_rates, labels, explanations,partitions,intervals = load_file(main_path, main_file)
############################# Set parameters #################################
raw_dimension = 4000
mother_wavelet = 'db4'
wav_level = 8
num_coefficient = 4
wav_dimension = 0

#wav_dimension = 22 + 22 + 38 + 69


rnn_layer ='LSTM'
rnn_hidden_node = 3
rnn_dropout = 0.4

batch_size = 100
epochs = 20

#############################  #################################

for run in range(0, num_experiments):
    train_samples_id = train_fractions[run]
    test_samples_id = test_fractions[run]

############################# Pre step1 #################################
    i = 0
    index = 0
    sample_x, sample_y = load_sample(main_path + paths[i], names[i], labels[i], sampling_rates[i], explanations[i], intervals[i], dimension=raw_dimension, step=3, train=False)
    if wav_dimension == 0:
        coefficients = pywt.wavedec(sample_x[index,:], mother_wavelet, level=wav_level)
        wav_dimension = 0
        for c in range(0, num_coefficient):
            wav_dimension += coefficients[c].shape[0]
    ############################# Step1 #################################
    raw_train_x, raw_train_y = load_data_new(main_path, main_file, train_samples_id, raw_dimension, train=True)
    clf_train_x = np.empty((0, wav_dimension))
    clf_train_y = np.empty((0, 2))
    for i in range(0, raw_train_x.shape[0]):
        sample = raw_train_x[i, :]
        coefficients = pywt.wavedec(sample[:, 0], mother_wavelet, level=wav_level)
        wav_features = np.empty((0,0))
        for i in range(0, num_coefficient):
            wav_features = np.append(wav_features, coefficients[i])
        reshaped_feature = np.reshape(wav_features,[1,wav_dimension])
        clf_train_x = np.append(clf_train_x, reshaped_feature, axis=0)
    clf_train_y = raw_train_y

    clf = tree.DecisionTreeClassifier()
    clf.fit(clf_train_x, clf_train_y)
############################ Step 2 ################################
    rnn_train_x = []
    rnn_train_y = []

    for i in train_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i],explanations[i], intervals[i], dimension=raw_dimension,step =3, train=False)
        sample_features = np.empty((0, wav_dimension))
        for index in range(0,sample_x.shape[0]):
            coefficients = pywt.wavedec(sample_x[index,:], mother_wavelet, level=wav_level)
            wav_features = np.empty((0, 0))
            for i in range(0, num_coefficient):
                wav_features = np.append(wav_features, coefficients[i])
            reshaped_feature = np.reshape(wav_features, [1, wav_dimension])
            sample_features = np.append(sample_features, reshaped_feature, axis=0)
        predicted = clf.predict(sample_features)

        rnn_train_x.append(predicted)
        if labels[i] == 'Normal':
            rnn_train_y.append(0)
        else:
            rnn_train_y.append(1)

    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(rnn_hidden_node, input_shape=(None, 1)))
    rnn_model.add(Dropout(rnn_dropout))
    rnn_model.add(Dense(1))
    rnn_model.add(Activation('sigmoid'))
    rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    array_train_x = sequence.pad_sequences(rnn_train_x, maxlen=None, dtype='float64', padding='post', truncating='post',value=0.)
    #model.fit(array_train_x, np.array(rnn_train_y), batch_size=batch_size, nb_epoch=epochs, validation_split=0.15)

    for counter in range(0, epochs):
        for x, y in zip(rnn_train_x, rnn_train_y):
            x = np.reshape(x, [1, x.shape[0], 1])
            y = np.reshape(y, [1, 1])
            rnn_model.train_on_batch(x, y)

# ----------------  Test model  on test samples ---------------
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = 0

    fp_samples = []
    fn_samples = []
    for i in test_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i],explanations[i], intervals[i], dimension=raw_dimension,step =3, train=False)
        sample_features = np.empty((0,wav_dimension))
        for index in range(0,sample_x.shape[0]):
            coefficients = pywt.wavedec(sample_x[index,:], mother_wavelet, level=wav_level)
            wav_features = np.empty((0, 0))
            for i in range(0, num_coefficient):
                wav_features = np.append(wav_features, coefficients[i])
            reshaped_feature = np.reshape(wav_features, [1, wav_dimension])
            sample_features = np.append(sample_features, reshaped_feature, axis=0)

        step1_predicted = clf.predict(sample_features)
        reshaped_predict = np.reshape(step1_predicted,[1,step1_predicted.shape[0],1])
        final_predicted = rnn_model.predict(reshaped_predict)

        if final_predicted[0] < 0.5:
            predicted_label = 'Normal'
        else:
            predicted_label = 'Arrhythmic'
        #print('%d   %s  real label =%s    -> predicted = %s    %f'%(i,names[i], labels[i], predicted_label, final_predicted[0]))
        n += 1
        if labels[i] == 'Normal':
            if predicted_label == 'Normal':
                tn += 1
            else:
                fp += 1
                fp_samples.append(i)
        else:
            if predicted_label == 'Normal':
                fn += 1
                fn_samples.append(i)
            else:
                tp += 1
    print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))
    test_tp.append(tp)
    test_tn.append(tn)
    test_fp.append(fp)
    test_fn.append(fn)
    test_acc.append((tp + tn) / n)
    test_sens.append(tp / (tp + fn))
    test_prec.append(tp / (tp + fp))
    print('+++++++++++++   (Test) Pay attention ++++++++++++++')
    print('fn:')
    print(fn_samples)
    print('fp:')
    print(fp_samples)

# ----------------  Test model  on train samples ---------------
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = 0
    fp_samples = []
    fn_samples = []
    for i in train_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i],explanations[i], intervals[i], dimension=raw_dimension,step =3, train=False)
        sample_features = np.empty((0,wav_dimension))
        for index in range(0,sample_x.shape[0]):
            coefficients = pywt.wavedec(sample_x[index,:], mother_wavelet, level=wav_level)
            wav_features = np.empty((0, 0))
            for i in range(0, num_coefficient):
                wav_features = np.append(wav_features, coefficients[i])
            reshaped_feature = np.reshape(wav_features, [1, wav_dimension])
            sample_features = np.append(sample_features, reshaped_feature, axis=0)

        step1_predicted = clf.predict(sample_features)
        reshaped_predict = np.reshape(step1_predicted,[1,step1_predicted.shape[0],1])
        final_predicted = rnn_model.predict(reshaped_predict)

        if final_predicted[0] < 0.5:
            predicted_label = 'Normal'
        else:
            predicted_label = 'Arrhythmic'
        #print('%d   %s  real label =%s    -> predicted = %s    %f'%(i,names[i], labels[i], predicted_label, final_predicted[0]))
        n += 1
        if labels[i] == 'Normal':
            if predicted_label == 'Normal':
                tn += 1
            else:
                fp += 1
                fp_samples.append(i)
        else:
            if predicted_label == 'Normal':
                fn += 1
                fn_samples.append(i)
            else:
                tp += 1
    print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))
    train_tp.append(tp)
    train_tn.append(tn)
    train_fp.append(fp)
    train_fn.append(fn)
    train_acc.append((tp + tn) / n)
    train_sens.append(tp / (tp + fn))
    train_prec.append(tp / (tp + fp))
    print('+++++++++++++   (Train) Pay attention ++++++++++++++')
    print('fn:')
    print(fn_samples)
    print('fp:')
    print(fp_samples)

##################################### Total result #####################################

tp = np.average(np.array(train_tp))
tn = np.average(np.array(train_tn))
fp = np.average(np.array(train_fp))
fn = np.average(np.array(train_fn))
train_accuracy = np.average(np.array(train_acc))
train_sensitivity = np.average(np.array(train_sens))
train_precision = np.average(np.array(train_prec))
print('**** Total : Train samples: ****')
print('\n--->Result for data = train , samples (%d Arrhythmic, %d Normal)' % ((fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, Accuracy-> %f, recall-> %f,  precision-> %f\n\n' % (tp, tn, fp, fn, train_accuracy,train_sensitivity,train_precision))

tp = np.average(np.array(test_tp))
tn = np.average(np.array(test_tn))
fp = np.average(np.array(test_fp))
fn = np.average(np.array(test_fn))
test_accuracy = np.average(np.array(test_acc))
test_sensitivity = np.average(np.array(test_sens))
test_precision = np.average(np.array(test_prec))
print('**** Total : Test samples: ****')
print('\n--->Result for data = test , samples (%d Arrhythmic, %d Normal)' % ((fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, Accuracy-> %f, recall-> %f,  precision-> %f\n\n' % (tp, tn, fp, fn, test_accuracy,test_sensitivity,test_precision))

##################################### Save results & parameters in file #####################################
text_file = open("Result_wav+DT+RNN.txt", "a")
text_file.write("\t\t\tResult wavelet + Decision Tree + RNN\n")
text_file.write("\n################### Save results & parameters in file ####################\n")
text_file.write("Test: Accuracy: %f, Sensitivity: %f, Precision: %f\n" %(test_accuracy,test_sensitivity,test_precision))
text_file.write("Train: Accuracy: %f, Sensitivity: %f, Precision: %f\n" %(train_accuracy,train_sensitivity,train_precision))
text_file.write("Train fp =")
for ans in train_fp:
    text_file.write("%d,  " %ans)
text_file.write("\nTrain fn =")
for ans in train_fn:
    text_file.write("%d,  " %ans)
text_file.write("\nTest fp =")
for ans in test_fp:
    text_file.write("%d,  " % ans)
text_file.write("\nTest fn =")
for ans in test_fn:
    text_file.write("%d,  " % ans)

text_file.write("\n Wavelet : mother wavelet = %s, level = %d & %d coefficients are used as feature" %(mother_wavelet, wav_level, num_coefficient))
text_file.write("\n %d features from input of length %d\n" %(wav_dimension, raw_dimension))

text_file.write("\n RNN model with %d parameters, %s with %d node  (dropout = %f)\n" %(rnn_hidden_node,rnn_layer,rnn_dropout))
text_file.write("batch = %d, epochs = %d  train on %d samples\n" %(batch_size,epochs,len(rnn_train_y)))
text_file.write("#######################################################")
text_file.close()
