import scipy.io
import numpy as np
import pandas as pd
import mat73
import argparse

## For new matlab version 
def pred2tsv_new(matlab_pred_path, output_path):
    mat = mat73.loadmat(matlab_pred_path)
    
    proteins = mat['pred']['object']
    proteins  = [protein[0] for protein in proteins]
    # Get the GO terms in the ontology
    terms = []
    for i in range(np.size(mat['pred']['score'],1)):
        terms.append(mat['pred']['ontology']['term'][i]['id'][0])

    GO_index = list(range(len(terms))) 
    prediction = []
    for i, protein in enumerate(proteins):
        non_zero_mask = mat['pred']['score'][i].toarray()[0]!=0
        score_indices = [GO_index[j] for j in range(len(non_zero_mask)) if non_zero_mask[j]]
        selected_terms = [terms[index] for index in score_indices]
        selected_score = mat['pred']['score'][i].toarray()[0][non_zero_mask]
        prediction.append(pd.DataFrame([[protein]*len(selected_terms), selected_terms, list(selected_score)]).T)

    prediction_df = pd.concat(prediction)
    prediction_df.to_csv(output_path, sep = '\t', index = False, header = None)
    
## For older matlab versions
def pred2tsv_old(matlab_pred_path, output_path):
    mat = scipy.io.loadmat(matlab_pred_path)
    
    # Get the names of the proteins predicted for
    proteins = mat['pred']['object'][0, 0].flatten()
    proteins = [item[0] for item in proteins]
    
    # Get the GO terms in the ontology
    terms = []
    for i in range(np.size(mat['pred']['score'][0, 0],1)):
        terms.append(mat['pred']['ontology'][0, 0][0][0][0][i][0][0][0])
        
    GO_index = list(range(len(terms))) 
    prediction = []
    for i, protein in enumerate(proteins):
        non_zero_mask = mat['pred']['score'][0, 0].toarray()[i]!=0
        score_indices = [index for index, condition in zip(GO_index, non_zero_mask) if condition]
        selected_terms = [terms[index] for index in score_indices]
        selected_score = mat['pred']['score'][0, 0].toarray()[i] [non_zero_mask]
        prediction.append(pd.DataFrame([[protein]*len(selected_terms), selected_terms, list(selected_score)]).T)
        
    prediction_df = pd.concat(prediction)
    prediction_df.to_csv(output_path, sep = '\t', index = False, header = None)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATLAB predictions to TSV format.")
    parser.add_argument("method", choices=['old', 'new'], help="Choose 'old' method or 'new' method")
    parser.add_argument("matlab_pred_path", type=str, help="Path to the MATLAB file with predictions")
    parser.add_argument("output_path", type=str, help="Output path for the TSV file")
    
    args = parser.parse_args()
    
    if args.method == 'old':
        pred2tsv_old(args.matlab_pred_path, args.output_path)
    elif args.method == 'new':
        pred2tsv_new(args.matlab_pred_path, args.output_path)

