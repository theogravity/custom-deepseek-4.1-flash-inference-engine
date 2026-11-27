from pathlib import Path
import shutil,subprocess,sys
p=Path('/tmp/ds4-cpu-eval')
folder=p/'image';(folder/'target/release').mkdir(parents=True,exist_ok=True)
shutil.copyfile('/home/benchmark/llm/ds4-flash-rs/Dockerfile',folder/'Dockerfile')
shutil.copy2(p/'target/release/ds4-flash-rs',folder/'target/release')
subprocess.run(['docker','build','-t','ds4-flash-rs:'+sys.argv[1],str(folder)],check=True)
