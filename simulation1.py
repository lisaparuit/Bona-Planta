# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:43:27 2024

@author: lisap
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log10, floor
import progressbar
import time 

start = time.time()
diff = time.time() - start

### DATA MANAGEMENT

def import_data():
    # Importing data
    data = pd.read_csv('p433_Run171114_A_Count_Sintax_new - p433_Run171114_A_Count_Sintax_new.csv', header=[0], index_col=[0])
    data.index.name = None
    # getting the number of samples
    N = data.select_dtypes(include='number').shape[1]
    # getting the number of OTUS
    S = data.shape[0]
    # Removing the taxonomic info
    data_OTUs = data.loc[:, "NTC1_A": 'R1_16_A'].astype(int)
    # Splitting the samples
    data_R = data.loc[:, "R1_01_A":"R1_16_A"]
    data_RS = data.loc[:, "RS1_01_A":"RS1_08_A"]
    data_BS = data.loc[:, "BS1_01_A":"BH1_02_A"]
    return(data,N,S,data_OTUs, data_R, data_RS, data_BS)
#data,N,S,data_OTUs, data_R, data_RS, data_BS = import_data()

def relative_abundance(data):
    """
    Input = OTUs count table in pd.DataFrame type 
    Output = OTUs relative abundance in pd.DataFrame type
    """
    return data.div(data.sum(axis=0), axis = 1)

### FORMULAS ###

def Shannon(data_ra):
    """
    Input : relative abundace data of the sample on which the diversity is to be calculated 
            **data.type = pd.DataFrame od shape (S,) , index = OTUs_ID, columns = sample names**

    Output : Shannon's index
    """
    S = data_ra.shape[0]
    Pi = data_ra.to_numpy()
 
    # Remove zeros for log calculation
    Pi = Pi[Pi != 0]

    # Calculate Shannon's index
    H = -np.sum(Pi * np.log10(Pi))

    return H

def Evenness(data_ra):
    """
    Input : relative abundance data of the community or sub-community on which the evenness is to be calculated 
            **data.type = pd.DataFrame, index = OTUs_ID, columns = sample names**

    Output : evenness index calculated using Pielous's J formula
    """
    return Shannon(data_ra)/log10(S)

def BrayCurtis(data):
    """
    Input : data of the community or sub-community on which the BCd matrix is to be calculated 
            **data.type = pd.DataFrame, index = OTUs_ID, columns = sample names**

    Output : matrix of size ncol*ncol showing the value of BC dissimilarity between samples
    """
    N = data.shape[1]
    arr_BC = np.zeros((N,N))

    # Calculate the sum of each column once
    sum_data = data.sum()

    for j in range(N):
        for k in range(j, N):  # Only calculate the upper half of the matrix
            # Extract two samples
            df_z = data.iloc[:, [j, k]]

            # Calculate the sum od the individuals in each samples
            Sj = sum_data.iloc[j]
            Sk = sum_data.iloc[k]

            # Calculate C_jk
            C_jk = df_z.min(axis=1).sum()

            # Calculate BC dissimilarity
            arr_BC[j, k] = arr_BC[k, j] = 2*C_jk / (Sj+Sk) if (Sj+Sk) != 0 else 0
 
    return pd.DataFrame(1- arr_BC, index=data.columns, columns=data.columns)
#BC = BrayCurtis(data_R)

def K_calculator(data, taxa):
    """ 
    Input : data of the microbial community on which the filtering criterion is calculated; 
            **data.type = pd.DataFrame, index = OTUs_ID, columns = sample names**

    Output : filtering criterion K defined with BC dissimilarity and intra-sample evenness
    """
    N = data.shape[1]

    # if a taxa is given, it is removed from the local data variable
    if taxa is not None :
        data = data.drop(taxa)

    # Sum of BC dissimilarities
    BC = BrayCurtis(data)
    Sum_BC = BC.values[np.triu_indices(N,k=1)].sum()

    # Sum of intra-sample evenness
    data_ra = relative_abundance(data)
    Sum_E = 0
    for sample in data_ra.columns:
        Sum_E += Evenness(data_ra.loc[:,sample])

    return Sum_BC * Sum_E
#K_calculator(data_R, None)


### ASSESMENT AND FIGURES

def BC_heatmap(BC):
    # Data
    N = BC.shape[0]  
    # Figure   
    fig, ax = plt.subplots()
    img = ax.imshow(BC)
    # Label recuperation
    x_label_list = BC.columns
    y_label_list = BC.index
    # x-axis ticks and labelling
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(x_label_list)
    # y-axis ticks and labelling
    ax.set_yticks(np.arange(N), label = y_label_list)
    ax.set_yticklabels(y_label_list)
    # Rotate the x-tick labels and set their alignment
    ax.tick_params(axis = "x", top=True, bottom=False, labeltop=True, labelbottom=False, rotation=90)  
    # Colorbar and legend title
    fig.colorbar(img, label = 'BC dissimilarity')
    # Show
    plt.show()  
#BC_heatmap(BC)

def run(data, threshold):
    
    # getting the number of OTUS
    S = data.shape[0]
    
    # progressbar
    bar = progressbar.ProgressBar(maxval= S+1, 
                                  widgets=[progressbar.Bar('+', '[', ']'), ' ', 
                                           progressbar.Percentage()])
    bar.start()
    
    # calculation
    kr_list = [K_calculator(data)]
    i = 0
    for taxa in data.index:
        # modifying data
        data_copy = data.drop(taxa)
        data_copy.reset_index(drop=True, inplace=True)
        # calculating indicators
        kr_list.append(K_calculator(data_copy))
        # updating progressbar
        bar.update(i+1)
        i += 1
        sleep(0.1)
    
    # finishing bar
    bar.finish()
    
    # plotting
    X = [x for x in range(S+1)]
    thresh = kr_list[0]+threshold
    plt.figure
    plt.plot( X , kr_list, label = 'K' )
    plt.plot( X, [thresh]*(S+1), label = 'threshold' )
    plt.title('variariations of K for root microbiome community')
    plt.legend()
    plt.show() 

    # verification
    list_removal = []
    for i in range(1,len(kr_list)):
        if kr_list[i] > thresh:
            list_removal.append(data.iloc[i-1,0])
    return list_removal

def run_optimized(data, threshold):
    # getting the number of OTUS
    S = data.shape[0]
    
    # progressbar
    bar = progressbar.ProgressBar(maxval=S+1, 
                                  widgets=[progressbar.Bar('+', '[', ']'), ' ', 
                                           progressbar.Percentage()])
    bar.start()
    
    # calculation
    kr_list = [K_calculator(data, None)]
    for i, taxa in enumerate(data.index):
        # calculating indicators without dropping taxa
        kr_list.append(K_calculator(data, taxa))
        
        # updating progressbar
        bar.update(i+1)
        time.sleep(0.1)
    
    # finishing bar
    bar.finish()
    
    # plotting
    X = list(range(S+1))
    thresh = kr_list[0] + threshold
    plt.figure()
    plt.plot(X, kr_list, label='K')
    plt.plot(X, [thresh]*(S+1), label='threshold')
    plt.title('Variations of K for root microbiome community')
    plt.legend()
    plt.show()

    # verification
    list_removal = [ (data.iloc[i, 0] , kr_list[i]-kr_list[0] ) 
                    for i in range(1, len(kr_list)) 
                    if kr_list[i] > thresh ]
    return list_removal 


#np.random.seed(1)
list_removal = run_optimized(data_R, 0) 
#df_test = data_R.iloc[[x for x in np.random.randint(0,S,int(S/10))],:]

