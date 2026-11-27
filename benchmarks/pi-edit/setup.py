from pathlib import Path
import json, os, subprocess as sp
p=Path(__file__).resolve().parent
release=Path(os.environ.get('PI_RELEASE_DIR', Path.home()/'.pi/agent/install/releases/1.4.0'))
def box():
 return ['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/home','--dir','/home/agent','--ro-bind','/etc/passwd','/etc/passwd','--ro-bind','/etc/group','/etc/group','--setenv','PATH','/usr/bin:/bin','--setenv','HOME','/home/agent','--setenv','LANG','C.UTF-8','--setenv','GOTOOLCHAIN','local','--setenv','GOPROXY','off','--setenv','CGO_ENABLED','0','--setenv','GOMAXPROCS','2','--setenv','GOCACHE','/cache','--chdir','/work']
def launcher():
 config=json.loads((p/'provider.json').read_text())
 return config['config'],config['thinking'],(p/'provider.mjs').read_bytes()

def node_runtime():
 import shutil
 selected=os.environ.get('PI_NODE_BINARY')
 candidates=[selected] if selected else ['/usr/bin/node',shutil.which('node')]
 for candidate in candidates:
  if not candidate or not Path(candidate).is_file():continue
  binary=Path(candidate).resolve()
  check=sp.run(box()+['--dir','/work','--ro-bind',str(binary),'/node','--','/node','--version'],
               capture_output=True,text=True)
  if check.returncode==0 and check.stdout.strip().startswith('v'):return binary
 raise RuntimeError('Pi requires a Node executable runnable in the sandbox. Set PI_NODE_BINARY to the official Linux x64 Node binary (tested v22.23.3).')

if __name__=='__main__':
 node_runtime()
 cmd=box()+['--bind',str(p/'cache'),'/cache','--ro-bind',str(p/'fixture'),'/work','--','go','test','-count=1','./...']
 r=sp.run(cmd,capture_output=True,text=True);print(r.stdout[-550:],r.stderr[-100:]);assert r.returncode==1 and 'want 8010' in r.stdout
