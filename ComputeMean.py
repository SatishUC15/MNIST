import numpy as np

def compute_mean(results):
    sensitivity = []
    specificity = []
    ppv = []
    npv = []
    for idx in range(len(results)):
        sensitivity.append(results[idx][0])
        specificity.append(results[idx][1])
        ppv.append(results[idx][2])
        npv.append(results[idx][3])
    print 'Mean sensitivity: ', np.mean(sensitivity), '+/-', np.std(sensitivity)
    print 'Mean specificity: ', np.mean(specificity), '+/-', np.std(specificity)
    print 'Mean PPV: ', np.mean(ppv), '+/-', np.std(ppv)
    print 'Mean NPV: ', np.mean(npv), '+/-', np.std(npv)

