import subprocess

datasets = ['searchsnippets', 'stackoverflow', 'biomedical']
dim_reds = ['UMAP']

for dataset in datasets:
    for dim_red in dim_reds:
        cmd = f"python stc.py --dataset {dataset} --dim_red {dim_red}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(cmd)
        stdout, stderr = process.communicate()
        print(stdout.decode())
        if stderr:
            print(stderr.decode())
        # break
    # break