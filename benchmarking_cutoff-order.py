#
#    MiMiC Helper Scripts:  a collection of python scripts to make working with MiMiC easier
#
#    MIT License
#    Copyright (C) 2025 Andrea Levy, Emiliano Ippoliti
#

import sys
import os.path
import stat
import re
from shutil import copyfile
import numpy as np

#help function accessible via python benchmarking_cutoff-order.py --help (or -h) 
import argparse
parser = argparse.ArgumentParser(description='Set up folders and files to run benchmarks on long range cutoff and multipole order for MiMiC.')
parser.add_argument('BENCHMARKS_PATH', type=str,
                    help='path where to create benchmarks folder')
parser.add_argument('RESTART_PATH', type=str, 
                    help='path to RESTART file')
parser.add_argument('CPMDINP_PATH', type=str, 
                    help='path to CPMD input file used to obtain the RESTART file')
parser.add_argument('TPR_PATH', type=str, 
                    help='path to .tpr file')
parser.add_argument('RUNINPUT_PATH', type=str, 
                    help='path to input script to run MiMiC')
parser.add_argument('-a', '--analysis', action='store_true',
                    help='analysis of the results performed')
parser.add_argument('-p', '--plot', action='store_true',
                    help='plot results of analysis')
parser.add_argument('-mo', '--multipole_order', action='store', nargs='+', type=int,
        help='multipole orders to test (default 0 1 2 3 4 5 6 7 8)')
parser.add_argument('-cr', '--cutoff_radius', action='store', nargs='+', type=int,
                    help='cutoff radius to test (default evaluated from box size)')
args = parser.parse_args()


#input files check
BENCHMARKS = args.BENCHMARKS_PATH
if not (os.path.isdir(BENCHMARKS)):
    print('\nERROR: Path where to set up benchmarks does not exist.')
    print('Please check path: '+BENCHMARKS)
    exit()
if(BENCHMARKS[-1]!='/'):
    BENCHMARKS += '/'

RESTART = args.RESTART_PATH
if not (os.path.isfile(RESTART)):
    print('\nERROR: Path to RESTART file does not exist.')
    print('Please check input path: '+RESTART)
    exit()

CPMDINP = args.CPMDINP_PATH
if not (os.path.isfile(CPMDINP)):
    print('\nERROR: Path to CPMD input file does not exist.')
    print('Please check input path: '+CPMDINP)
    exit()

TPR = args.TPR_PATH
if not (os.path.isfile(TPR)):
    print('\nERROR: Path to tpr file does not exist.')
    print('Please check input path: '+TPR)
    exit()

INPUT = args.RUNINPUT_PATH
if not (os.path.isfile(INPUT)):
    print('\nERROR: Path to input script does not exist.')
    print('Please check input path: '+INPUT)
    exit()
runinp_linestochange = ['cpmd_input=','--time=','--nodes=']

#If analysis option is not provided, the script set up folders and files
#After having run all inp, script can be rerun adding --analysis option
if(args.analysis==False):  
    if (os.path.isdir(BENCHMARKS+'benchmarking_cutoff-order')):
        print('\nERROR: benchmarking_cutoff-order folder already exists,')
        print('probably from a previous execution of benchmarking_cutoff-order.')
        print('Please check: '+BENCHMARKS+'benchmarking_cutoff-order and what RESTART file was used')
        print('and consider renaming the folder to be able to execute benchmarking_cutoff-order.py.')
        exit()
    
    # set up reference for benchmarks
    # and extract BOX information 
    
    # reference.inp file
    os.mkdir(BENCHMARKS+'benchmarking_cutoff-order')
    os.mkdir(BENCHMARKS+'benchmarking_cutoff-order/reference')
    
    copyfile(RESTART, BENCHMARKS+'benchmarking_cutoff-order/reference/RESTART') 
    copyfile(TPR, BENCHMARKS+'benchmarking_cutoff-order/reference/mimic.tpr') 
    inpfile = open(BENCHMARKS+'benchmarking_cutoff-order/reference/reference.inp', 'w')
    inpfile.write('''&CPMD
    MIMIC
    OPTIMIZE WAVEFUNCTION
    RESTART WAVEFUNCTION COORDINATES
    MEMORY BIG
    REAL SPACE WFN KEEP
    CONVERGENCE ORBITALS
     1e-6
    NEW CONSTRAINTS
    ISOLATED MOLECULE
    TRAJECTORY OFF
    PRINT FORCES ON
&END
    
    ''')

    previnp = open(CPMDINP, 'r')
    copy_inp = False
    sections = ['&CPMD', '&SYSTEM', '&DFT', '&MIMIC', '&ATOMS']
    sections_to_copy = sections[1:]
    prev_line = ''
    MIMIC_to_skip = ['LONG-RANGE COUPLING', 'FRAGMENT SORTING ATOM-WISE UPDATE', 'CUTOFF DISTANCE', 'MULTIPOLE ORDER']
    current_sec = ''
    #to store
    MM_BOX=[0,0,0]
    QM_BOX=[0,0,0]

    for line in previnp:
        for sec in sections:
            if(sec in line):
                current_sec = sec
                if(sec in sections_to_copy):
                    copy_inp = True
        if ('&END' in line and current_sec in sections_to_copy): 
            inpfile.write(line+'\n')
            copy_inp = False
        elif(copy_inp and current_sec != '&MIMIC'):
            inpfile.write(line)
        elif(copy_inp and current_sec == '&MIMIC'):
            if any(s in line for s in MIMIC_to_skip):
                pass
            elif any(s in prev_line for s in MIMIC_to_skip):
                pass
            elif (line[0]=='/'):
                inpfile.write(os.path.abspath(BENCHMARKS+'benchmarking_cutoff-order/reference')+'\n')
            else:
                inpfile.write(line)
        #store BOX information 
        if(current_sec == '&MIMIC' and 'BOX' in prev_line):
            MM_BOX = [float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] 
        if(current_sec == '&SYSTEM' and 'CELL' in prev_line):
            QM_BOX = [float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] 
            if not('ABSOLUTE' in prev_line):
                QM_BOX[1] *= QM_BOX[0]
                QM_BOX[2] *= QM_BOX[0]
        prev_line = line
    
    previnp.close()
    inpfile.close()
    
    # run file for reference.inp 
    copyfile(INPUT, BENCHMARKS+'benchmarking_cutoff-order/reference/run.sh') 
    with open(BENCHMARKS+'benchmarking_cutoff-order/reference/run.sh', 'r') as input:
        with open('temp', 'w') as output:
            for line in input:
                #runinp_linestochange = ['cpmd_input=','--time=','--nodes=']
                if not any(s in line for s in runinp_linestochange):
                    output.write(line)
                elif('cpmd_input=' in line):
                    output.write('cpmd_input=reference\n')
                elif('--time=' in line):
                    output.write('#SBATCH --time=00:03:00\n')
                elif('--nodes=' in line):
                    output.write('#SBATCH --nodes=1\n')
    
    os.replace('temp', BENCHMARKS+'benchmarking_cutoff-order/reference/run.sh') 
        
    # set up different folders for different runs
    # varying multiple order and long range cutoff
    
    #multipole orders to test
    #default [0,1,2,3,4,5,6,7,8]
    if(args.multipole_order==None):
        orders = range(9)
    #otherwise can be provided as input
    else:
        orders = args.multipole_order
    
    #long range radii to test
    #default evaluated from quantum and classical box size
    if(args.cutoff_radius==None):
        min_cut = 0.5*np.sqrt(3)*max(QM_BOX)*.7    #70% of maximum distance between center of QM box and edge
        max_cut = 0.5*np.sqrt(3)*max(MM_BOX)*.7    #70% of maximum distance between center of MM box and edge 
        cutoffs = np.linspace(min_cut, max_cut, 10)
        cutoffs = [int(c) for c in cutoffs]
        cutoffs = set(cutoffs)
    #otherwise can be provided as input
    else:
        cutoffs = args.cutoff_radius
    
    # loop over all cutoffs and orders
    for CUTOFF in cutoffs:
        for ORDER in orders:
            step = 'cutoff-'+str(CUTOFF)+'_order-'+str(ORDER)
            os.mkdir(BENCHMARKS+'benchmarking_cutoff-order/'+step)
            copyfile(BENCHMARKS+'benchmarking_cutoff-order/reference/RESTART', BENCHMARKS+'benchmarking_cutoff-order/'+step+'/RESTART') 
            copyfile(BENCHMARKS+'benchmarking_cutoff-order/reference/mimic.tpr', BENCHMARKS+'benchmarking_cutoff-order/'+step+'/mimic.tpr') 
            copyfile(BENCHMARKS+'benchmarking_cutoff-order/reference/reference.inp', BENCHMARKS+'benchmarking_cutoff-order/'+step+'/'+step+'.inp') 
    
            #optimize wavefunction input file
            inpfile = BENCHMARKS+'benchmarking_cutoff-order/'+step+'/'+step+'.inp'
            with open(inpfile, 'r') as input:
                with open('temp', 'w') as output:
                    for line in input:
                        if 'benchmarking_cutoff-order/reference' not in line.strip('\n'):
                            output.write(line)
                        else:   
                            output.write(os.path.abspath(BENCHMARKS+'benchmarking_cutoff-order/'+step)+'\n')
                            output.write('LONG-RANGE COUPLING\nFRAGMENT SORTING ATOM-WISE UPDATE\n 1\nCUTOFF DISTANCE\n')
                            output.write(str(CUTOFF)+'\n')
                            output.write('MULTIPOLE ORDER\n')
                            output.write(str(ORDER)+'\n')
            os.replace('temp', inpfile)
    
    
            # slurm file for cpmd.inp 
            copyfile(INPUT, BENCHMARKS+'benchmarking_cutoff-order/'+step+'/run.sh') 
            with open(BENCHMARKS+'benchmarking_cutoff-order/'+step+'/run.sh', 'r') as input:
                with open('temp', 'w') as output:
                    for line in input:
                        #runinp_linestochange = ['cpmd_input','--output=','--error=','--time=','--nodes=']
                        if not any(s in line for s in runinp_linestochange):
                            output.write(line)
                        elif('cpmd_input=' in line):
                            output.write('cpmd_input='+step+'\n')
                        elif('--time=' in line):
                            output.write('#SBATCH --time=00:03:00\n')
                        elif('--nodes=' in line):
                            output.write('#SBATCH --nodes=1\n')
    
            os.replace('temp', BENCHMARKS+'benchmarking_cutoff-order/'+step+'/run.sh') 
    
    # example bash script to run single point calculations 
    bash_example = open(BENCHMARKS+'/run_benchmarks.sh','w')
    bash_example.write('''#!/bin/bash
    BENCHMARKING_DIR=$(pwd)'/benchmarking_cutoff-order'
    benchmarks=$BENCHMARKING_DIR/*
    
    for dir in $benchmarks;
    do
        cd $dir
        echo -e '\\nRunning '$dir'\\n...'
        sbatch -W run.sh
        wait
        echo -e 'Done\\n'
    done''')
        
    bash_example.close()
    st = os.stat('./run_benchmarks.sh')
    os.chmod('./run_benchmarks.sh', st.st_mode | stat.S_IEXEC)
    
if(args.analysis==True):
    import pandas as pd
    
    results_file = open(BENCHMARKS+'benchmarking_cutoff-order/reference/benchmarking.out', 'r')
    ReadLine=False
    prev_line = ''
    for line in results_file:
        if('FINAL RESULTS' in line):
            ReadLine=True
        if(ReadLine==True):
            if('TOTAL ENERGY' in line):
                EN_ref = float(line.split('=')[-1].replace('A.U.',''))
            if('NUCLEAR GRADIENT:' in prev_line):
                F_ref = float(line.split('=')[1].replace('NORM',''))
        prev_line = line

    df = pd.DataFrame(columns=['cutoff','order','EnergyDiff', 'ForceDiff'])
    for file in os.listdir(BENCHMARKS+'benchmarking_cutoff-order/'):
        if(file!='reference'):
            results_file = open(BENCHMARKS+'benchmarking_cutoff-order/'+file+'/benchmarking.out', 'r')
            ReadLine=False
            prev_line = ''
            force = 0.
            for line in results_file:
                if('FINAL RESULTS' in line):
                    ReadLine=True
                if(ReadLine==True):
                    if('NUCLEAR GRADIENT:' in prev_line):
                       force  = float(line.split('=')[1].replace('NORM',''))
                    if('TOTAL ENERGY' in line):
                        cutoff = re.search(r"cutoff-(\d+)", file)
                        row_to_add = pd.Series({'cutoff':float(cutoff.group(1)), 'order':int(file.split('-')[-1]), 'EnergyDiff':abs(float(line.split('=')[-1].replace('A.U.',''))-EN_ref), 'ForceDiff':abs(force-F_ref)})
                        #df = df.append(row_to_add, ignore_index=True)
                        df.loc[len(df)] = row_to_add                        
                prev_line = line

    df = df.sort_values(by=['cutoff', 'order'])    


    # output file energy comparison
    energy_output = open('benchmarking_cutoff-order_energy.dat', 'w')
    
    energy_output.write('cut/ord')
    for i in sorted(set(df['order'])):
        energy_output.write('\t\t\t'+str(int(i)))
    for i in sorted(set(df['cutoff'])):
        energy_output.write('\n'+str(i))
        for j in df.loc[df['cutoff']==i]['EnergyDiff']:
            energy_output.write('\t'+str(j))            

    energy_output.close()
    
    #output file force comparison
    force_output = open('benchmarking_cutoff-order_force.dat', 'w')
    
    force_output.write('cut/ord')
    for i in sorted(set(df['order'])):
        force_output.write('\t\t\t'+str(int(i)))
    for i in sorted(set(df['cutoff'])):
        force_output.write('\n'+str(i))
        for j in df.loc[df['cutoff']==i]['ForceDiff']:
            force_output.write('\t'+str(j))            
    force_output.close()

    #plot (optional)
    if(args.plot==True):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2)
        fig.suptitle('Benchmarking multiple order - long range cutoff')
        
        ax[0].set_yscale('log')
        for i in sorted(set(df['order'])):
            ax[0].plot(df.loc[df['order']==i]['cutoff'],df.loc[df['order']==i]['EnergyDiff'], 'o-', label=str(int(i)))
        ax[0].set_xlabel('Cutoff [a.u]')
        ax[0].set_ylabel('|$\Delta$E| [a.u]')
        ax[0].legend(loc='center left',title='Multiple Order')
        
        ax[1].set_yscale('log')
        for i in sorted(set(df['order'])):
            ax[1].plot(df.loc[df['order']==i]['cutoff'],df.loc[df['order']==i]['ForceDiff'], 'o-', label=str(int(i)))
        ax[1].set_xlabel('Cutoff [a.u]')
        ax[1].set_ylabel('|$\Delta$g| [a.u]')
        ax[1].axhline(y = 1E-6, color='black', linestyle = '--', label = 'Numerical Precision')
        ax[1].legend(loc='center left',title='Multiple Order')
        
        plt.show()
