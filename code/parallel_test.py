import subprocess
import multiprocessing
from joblib import Parallel, delayed

def run_process(gaf_file, out_path, log_file):
    cmd = [
    "python3",                 # Command to execute Python 3
    "preprocess_gaf.py",       # Script to run
    gaf_file,  # Path to input file
    #"--highTP",
    "--out_path", out_path,        # Output path parameter
    #"--evidence_codes", "EXP", "IDA",   # Evidence codes parameter
    #"--extract_col_list", "DB Object ID", "Qualifier"  # Extract column list parameter
    #" > ", log_t0,
]
    test_cmd = "bash test.sh"
    with open(log_file, "w") as f:
        print(" ".join(cmd))
        result = subprocess.run(" ".join(cmd), shell=True, stdout=f, stderr=subprocess.STDOUT)
        #result = subprocess.run(["ls", "-l"], capture_output=True)
        
if __name__ == "__main__":
    # Define commands and log file names
    work_dir = "/data/rashika/CAFA4/"
    
    t0_gaf_file = work_dir + "uniprot/raw_goa/sample_t0.gz"
    #t0_gaf_file = work_dir + "uniprot/raw_goa/t0/goa_uniprot_all.gaf.195.gz"
    #t0_out_path = work_dir + "extracted_goa/t0_preprocessed.csv"
    t0_out_path = work_dir + "extracted_goa/t0_sample.csv"
    #log_t0 =  work_dir + "log/log_preprocess_t0.txt"
    log_t0 =  work_dir + "log/log_preprocess_t0_sample.txt"
    
    t1_gaf_file = work_dir + "uniprot/raw_goa/sample_t1.gz"
    #t1_gaf_file = work_dir + "uniprot/raw_goa/t1/goa_uniprot_all.gaf.gz"
    #t1_out_path = work_dir + "extracted_goa/t1_preprocessed.csv"
    t1_out_path = work_dir + "extracted_goa/t1_sample.csv"
    #log_t1 = work_dir + "log/log_preprocess_t1.txt"
    log_t1 =  work_dir + "log/log_preprocess_t1_sample.txt"
    
    Inputs = [[t0_gaf_file, t0_out_path, log_t0], [t1_gaf_file, t1_out_path, log_t1]]
    your_outputs = Parallel(n_jobs=2, verbose=10)(delayed(run_process)(*inputs_i) for inputs_i in Inputs)
    # Create processes for each command
    #process1 = multiprocessing.Process(target=run_process, args=(cmd_preprocess_t0, log_t0))
    #process2 = multiprocessing.Process(target=run_process, args=(cmd_preprocess_t1, log_t1))
    #run_process(cmd_preprocess_t0, log_t0)
    #run_process(cmd_preprocess_t1, log_t0)
    # Start the processes
    #process1.start()
    #process2.start()

    # Wait for both processes to finish
    #process1.join()
    #process2.join()

    #print("Both processes have finished.")

