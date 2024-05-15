import subprocess

datasets = ['searchsnippets', 'stackoverflow', 'biomedical']

for dataset in datasets:
    cmd = f"python embedding.py --dataset {dataset}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(cmd)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(stderr.decode())
        # break
    # break