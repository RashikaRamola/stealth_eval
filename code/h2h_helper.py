import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_plots_h2_h(results_path, metric, cols,out_path='/home/rashika/CAFA4/eval/plots/', n_curves = None, names_file = None, S_min_coord = None):
    dir_list = os.listdir(results_path)
    
    cumulate = True
    add_extreme_points = True
    coverage_threshold = 0.3
    axis_title_dict = {'pr': 'Precision', 'rc': 'Recall', 'f': 'F-score', 'pr_w': 'Weighted Precision', 'rc_w': 'Weighted Recall', 'f_w': 'Weighted F-score', 'mi': 'Misinformation (Unweighted)', 'ru': 'Remaining Uncertainty (Unweighted)', 'mi_w': 'Misinformation', 'ru_w': 'Remaining Uncertainty', 's': 'S-score', 'pr_micro': 'Precision (Micro)', 'rc_micro': 'Recall (Micro)', 'f_micro': 'F-score (Micro)', 'pr_micro_w': 'Weighted Precision (Micro)', 'rc_micro_w': 'Weighted Recall (Micro)', 'f_micro_w': 'Weighted F-score (Micro)'}
    ontology_dict = {'biological_process': 'BPO', 'molecular_function': 'MFO', 'cellular_component': 'CCO'}
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    for file in dir_list:
        df_file = results_path + file +"/evaluation_all.tsv"
        df = pd.read_csv(df_file, sep="\t")
        out_folder = out_path + file
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
            
        
        df = pd.read_csv(df_file, sep="\t")
        
        # Set method information (optional)
        if names_file is None:
            df['group'] = df['cafa_filename']
            df['label'] = df['cafa_filename']
            df['is_baseline'] = False
        else:
            methods = pd.read_csv(names_file, sep = "\t", header=0)
            df = pd.merge(df, methods, on='cafa_filename', how='left')
            df['group'].fillna(df['cafa_filename'], inplace=True)
            df['label'].fillna(df['cafa_filename'], inplace=True)
            if 'is_baseline' not in df:
                df['is_baseline'] = False
            else:
                df['is_baseline'].fillna(False, inplace=True)
            # print(methods)
        #df = df.drop(columns='cafa_filename').set_index(['group', 'label', 'ns', 'tau'])
        df = df.set_index(['group_unique', 'label', 'ns', 'cafa_filename','tau'])
        
        # Filter by coverage
        df = df[df['cov'] >= coverage_threshold]
        
        # Assign colors based on group
        if 'colors' not in df.columns:
            cmap = plt.get_cmap('tab20')
            df['colors'] = df.index.get_level_values('group_unique')
            df['colors'] = pd.factorize(df['colors'])[0]
            df['colors'] = df['colors'].apply(lambda x: cmap.colors[x % len(cmap.colors)])

        index_best = df.groupby(level=['group_unique', 'ns'])[metric].idxmax() if metric in ['f', 'f_w', 'f_micro', 'f_micro_w'] else df.groupby(['group_unique', 'ns'])[metric].idxmin()
        
        # Filter the dataframe for the best methods
        df_methods = df.reset_index('tau').loc[[ele[:-1] for ele in index_best], ['tau', 'cov', 'colors'] + cols + [metric]].sort_index()

        # Makes the curves monotonic. Cumulative max on the last column of the cols variable, e.g. "pr" --> precision
        if cumulate:
            if metric in ['f', 'f_w', 'f_micro', 'f_micro_w']:
                df_methods[cols[-1]] = df_methods.groupby(level=['label', 'ns'])[cols[-1]].cummax()
            else:
                df_methods[cols[-1]] = df_methods.groupby(level=['label', 'ns'])[cols[-1]].cummin()


        # Save to file
        df_methods.drop(columns=['colors']).to_csv('{}/fig_{}.tsv'.format(out_folder, metric), float_format="%.3f", sep="\t")
        
        # Add first last points to precision and recall curves to improve APS calculation
        #def add_points(df_):
        #    df_ = pd.concat([df_.iloc[0:1], df_])
        #    df_.iloc[0, df_.columns.get_indexer(['tau', cols[0], cols[1]])] = [0, 1, 0]  # tau, rc, pr
        #    df_ = pd.concat([df_, df_.iloc[-1:]])
        #    df_.iloc[-1, df_.columns.get_indexer(['tau', cols[0], cols[1]])] = [1.1, 0, 1]
        #    return df_

        #if metric.startswith('f') and add_extreme_points:
        #    df_methods = df_methods.reset_index().groupby(['group_unique', 'label', 'ns'], as_index=False).apply(add_points).set_index(['group_unique', 'label', 'ns'])
        
        # Filter the dataframe for the best method and threshold
        df_best = df.loc[index_best, ['cov', 'colors'] + cols + [metric]]
        
        # Calculate average precision score 
        #if metric.startswith('f'):
        #    df_best['aps'] = df_methods.groupby(level=['group_unique', 'label', 'ns'])[[cols[0], cols[1]]].apply(lambda x: (x[cols[0]].diff(-1).shift(1) * x[cols[1]]).sum())

        # Calculate the max coverage across all thresholds
        df_best['max_cov'] = df_methods.groupby(level=['group_unique', 'label', 'ns'])['cov'].max()
        
        # Set a label column for the plot legend
        df_best['label'] = df_best.index.get_level_values('group_unique')
        
        # Define a function to create the label
        def create_label(row):
            cafa_part = row['label'].split("_", 1)[0].upper() + " - " + row['label'].split("_", 1)[1]
            metric_part = f"{metric.upper().split('_')[0]}={row[metric]:.3f}"
            max_cov_part = f"C={row['max_cov']:.3f}"
            return f"{cafa_part} ({metric_part} {max_cov_part})"

        # Apply the function row-wise to create the label column
        df_best['label'] = df_best.apply(create_label, axis=1)
        #df_best['label'] = df_best.agg(lambda x: f"{x['label'].split("_",1)[0].upper() + x['label'].split("_",1)[1]} ({metric.upper().split('_')[0]}={x[metric]:.3f} C={x['max_cov']:.3f})", axis=1)
#         if 'aps' not in df_best.columns:
#             df_best['label'] = df_best.agg(lambda x: f"{x['label']} ({metric.upper()}={x[metric]:.3f} C={x['max_cov']:.3f})", axis=1)
#         else:
#             df_best['label'] = df_best.agg(lambda x: f"{x['label']} ({metric.upper()}={x[metric]:.3f} APS={x['aps']:.3f} C={x['max_cov']:.3f})", axis=1)
        
        # Generate the figures
        plt.rcParams.update({'font.size': 12, 'legend.fontsize': 8})

        # F-score contour lines
        x = np.arange(0.01, 1, 0.01)
        y = np.arange(0.01, 1, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X * Y / (X + Y)
        
        for ns, df_g in df_best.groupby(level='ns'):
            fig, ax = plt.subplots(figsize=(4, 4))

             # Contour lines. At the moment they are provided only for the F-score
            if metric.startswith('f'):
                CS = ax.contour(X, Y, Z, np.arange(0.1, 1.0, 0.1), colors='gray', linewidths=0.5)
                ax.clabel(CS, inline=True) #, fontsize=10)

            cnt = -1
            # Iterate methods
            for i, (index, row) in enumerate(df_g.sort_values(by=[metric, 'max_cov'], ascending=[False if metric.startswith('f') else True, False]).iterrows()):
                
                cnt+=1
                #print(row)
                if (n_curves and cnt <= n_curves) or ('BLAST' in row['label']) or ('Naive' in row['label']):
                
                    #data = df_methods.loc[index[:-1]]

                    data = df_methods.loc[index[:-2]]
    
    
                    # Precision-recall or mi-ru curves
                    # Determine linestyle based on label or condition
                    linestyle = '-'  # Default linestyle
                    if 'BLAST' in row['label'] or 'Naive' in row['label']:
                        linestyle = 'dotted'
                    elif 'cafa3' in row['label']:
                        linestyle = 'dotted'
                    elif 'cafa2' in row['label']:
                        linestyle = 'dashed'
                    
                    ax.plot(data[cols[0]], data[cols[1]], color=row['colors'], linestyle = linestyle, label=row['label'], lw=1.5, zorder=500-i)


                    
                    # F-max or S-min dots
                    ax.plot(row[cols[0]], row[cols[1]], color=row['colors'], marker='o', markersize=12, mfc='none', zorder=1000-i)
                    ax.plot(row[cols[0]], row[cols[1]], color=row['colors'], marker='o', markersize=6, zorder=1000-i)
                    
                    
                
                
            # Set axes limit
            
            if metric.startswith('s') and S_min_coord:
                plt.xlim(S_min_coord[file][0])
                plt.ylim(S_min_coord[file][1])
            
            #Set axes limit
            #if metric.startswith('s'):
            #    plt.xlim(23, 28)
            #    plt.ylim(0, 50)
                #plt.xlim(0.4*df_best.loc[:,:,ns,:][cols[0]].max(), df_best.loc[:,:,ns,:][cols[0]].max())
                #plt.ylim(0, 1)

            # plt.xlim(0, max(1, df_best.loc[:,:,ns,:][cols[0]].max()))
            # plt.ylim(0, max(1, df_best.loc[:,:,ns,:][cols[1]].max()))

            # Set titles
            type_dict = {}
            type_dict['type1'] = 'NK'
            type_dict['type2'] = 'LK'
            type_dict['type3'] = 'PK'
            type_dict['type12'] = 'NK + LK'
            ax.set_title(file.split('_')[0].upper() + " (" + type_dict[file.split('_')[2]] + ")", fontsize = 25)
            #ax.set_xlabel(axis_title_dict[cols[0]], labelpad=20, fontsize=36)
            #ax.set_ylabel(axis_title_dict[cols[1]], labelpad=20, fontsize=36)
            ax.set_xlabel("Precision", fontsize = 18)
            ax.set_ylabel("Recall", fontsize = 18)
            fig.set_linewidth(5)
        

            # Legend
            #ax.legend(loc='center right', bbox_to_anchor=(1, 1.5))
            leg = ax.legend(markerscale=6, title=file, loc='upper center')
            for legobj in leg.get_lines():
                legobj.set_linewidth(10.0)
             
            #leg.set_bbox_to_anchor((0.5, -1))  
            ax.legend(title='', fontsize=8, bbox_to_anchor=(1.025, -0.2))
            # Save figure on disk
            plt.savefig("{}/fig_{}_{}.png".format(out_folder, metric, ns), bbox_inches='tight', dpi=300, transparent=True)
            # plt.clf()



