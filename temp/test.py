import subprocess
aa = subprocess.run("python ./test2.py", capture_output=True, shell=True)
print(aa.stderr)
