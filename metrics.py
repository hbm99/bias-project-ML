from sklearn import metrics
from itertools import combinations
import numpy as np
import operator
import matplotlib.pyplot as plt

EO_THRESHOLD = 0.1
DI_THRESHOLD = 0.8

TP= 'tp'
FP ='fp'
FN ='fn'
TN ='tn'


def macro_accuracy(y_true, y_predict):
    macro_acc= metrics.balanced_accuracy_score(y_true, y_predict)
    return macro_acc

def macro_f1(y_true, y_predict, labels):
    macro_f1= metrics.f1_score(y_true, y_predict,labels=labels, average='macro')
    return macro_f1

def confusion_matrix(y_true, y_predict, labels, plot=True):
    conf_matrix= metrics.confusion_matrix(y_true, y_predict, labels=labels, normalize='true')
    
    if plot:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=labels)
        # disp.ax_.set_title('Normalized Confusion matrix')
        
        disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        disp.ax_.set_title('Normalized Confusion Matrix')

        plt.show()

    return conf_matrix



def compute_TP_FP_FN_TN(y_true, y_predict, labels):
    rates = {lbl:{TP: 0, FP: 0, FN: 0, TN:0} for lbl in labels}
    conf_matrix = confusion_matrix(y_true, y_predict,labels, plot=False)

    false_positives = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    false_negatives = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    true_positives = np.diag(conf_matrix)
    true_negatives = conf_matrix.sum() - (false_positives + false_negatives + true_positives)

    # Filling dictionaries
    for i in range (len(labels)):
        rates[labels[i]][TP]= true_positives[i]
        rates[labels[i]][FP]= false_positives[i]
        rates[labels[i]][FN]= false_negatives[i]
        rates[labels[i]][TN]= true_negatives[i]
    
    return rates


def selection_rate(y_true, y_predict, labels):
    rates= compute_TP_FP_FN_TN(y_true, y_predict, labels)

    count_per_class={lbl:0 for lbl in labels}
    for item in y_true:
        count_per_class[item] += 1

    sr_per_class={lbl: rates[lbl][TP]/ count_per_class[lbl] for lbl in labels}

    return sr_per_class


def tpr(y_true, y_predict, labels):
    rates= compute_TP_FP_FN_TN(y_true, y_predict, labels)
    tpr_per_class={lbl: rates[lbl][TP]/ (rates[lbl][TP] + rates[lbl][FN]) for lbl in labels }
    
    return tpr_per_class

def fpr(y_true, y_predict, labels):
    rates= compute_TP_FP_FN_TN(y_true, y_predict, labels)
    fpr_per_class={lbl: rates[lbl][FP]/ (rates[lbl][FP] + rates[lbl][TN]) for lbl in labels}
    
    return fpr_per_class

def equalized_odds(y_true, y_predict, labels):
    print("EQUALIZED ODDS")
    tpr_values= tpr(y_true, y_predict, labels)

    fpr_values= fpr(y_true, y_predict, labels)

    equalized_odds = True

    for pair in combinations(labels, 2):
        
        first_class = pair[0]
        second_class = pair[1]
        if first_class == 'other' or second_class == 'other':
            continue

        tpr_eo_value= abs(tpr_values[first_class] - tpr_values[second_class])

        if tpr_eo_value >= EO_THRESHOLD:
            equalized_odds = False
            print(str(tpr_eo_value) +': Not equalized odds between ' + first_class + ' and ' + second_class + 
                  ' (tpr). Privileged group: ' + max(tpr_values.items(), key=operator.itemgetter(1))[0])
            
        
        fpr_eo_value= abs(fpr_values[first_class] - fpr_values[second_class])
        if fpr_eo_value >= EO_THRESHOLD:
            equalized_odds = False
            print(str(fpr_eo_value) +': Not equalized odds between ' + first_class + ' and ' + second_class + 
                  ' (fpr). Privileged group: ' + min(fpr_values.items(), key=operator.itemgetter(1))[0])

    if equalized_odds:
        print('Equalized odds') 
        return True
    return False


def disparate_impact(y_true, y_predict, labels):
    # disparate impact ratio = underprivileged group SR / privileged group SR

    print("DISPARATE IMPACT.")
    disparate_impact=False

    sr_values= selection_rate(y_true, y_predict, labels)

    for pair in combinations(labels, 2):
        
        first_class = pair[0]
        second_class = pair[1]

        if sr_values[first_class] > sr_values[second_class]:
                pg = first_class
                ug = second_class
        else: 
            pg = second_class
            ug = first_class

        disp_impact = sr_values[ug] / sr_values[pg]

        if disp_impact < DI_THRESHOLD:
            disparate_impact=True
            print('Disparate impact present in  ' + ug +  '/' +  pg + "\n"+
                    'Value:'+ str(disp_impact) )

    if not disparate_impact:
        print('No disparate impact present')
        return False
    return True