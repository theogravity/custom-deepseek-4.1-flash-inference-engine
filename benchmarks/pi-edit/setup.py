from pathlib import Path
import json, os, subprocess as sp
p=Path(__file__).resolve().parent
release=Path(os.environ.get('PI_RELEASE_DIR', Path.home()/'.pi/agent/install/releases/1.4.0'))
def box():
 return ['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/home','--dir','/home/agent','--ro-bind','/etc/passwd','/etc/passwd','--ro-bind','/etc/group','/etc/group','--setenv','PATH','/usr/bin:/bin','--setenv','HOME','/home/agent','--setenv','LANG','C.UTF-8','--setenv','GOTOOLCHAIN','local','--setenv','GOPROXY','off','--setenv','CGO_ENABLED','0','--setenv','GOMAXPROCS','2','--setenv','GOCACHE','/cache','--chdir','/work']
def launcher():
 import os,shlex,tempfile
 with tempfile.TemporaryDirectory() as td:
  cmd=['model-cli','pi','dsv41-flash-exl3','--direct']
  raw=sp.check_output(cmd,text=True,env=os.environ|{'XDG_CACHE_HOME':td,'MODEL_CLI_NO_UPDATE':'1'})
  args=shlex.split(raw.replace('\\\n',' '))
  cfg=json.loads(next(x.split('=',1)[1] for x in args if x.startswith('MODEL_CLI_PI_CONFIG=')))
  cfg['connection']['apiKey']='$MODEL_CLI_PI_API_KEY'
  cfg['connection']['baseUrl']='http://127.0.0.1:8180/v1'
  extension=(Path(td)/'model-cli/pi-provider.mjs').read_bytes()
  return cfg,args[args.index('--thinking')+1],extension
if __name__=='__main__':
 cmd=box()+['--bind',str(p/'cache'),'/cache','--ro-bind',str(p/'fixture'),'/work','--','go','test','-count=1','./...']
 r=sp.run(cmd,capture_output=True,text=True);print(r.stdout[-550:],r.stderr[-100:]);assert r.returncode==1 and 'want 8010' in r.stdout
