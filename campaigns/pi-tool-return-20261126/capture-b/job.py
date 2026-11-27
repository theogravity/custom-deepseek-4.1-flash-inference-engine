import json,fcntl,sys
from pathlib import Path
from trial import one,p
sys.path.insert(0,'/tmp/ds4-agentic-live-1126')
from sweep import b,watch,flags,images
import threading
with open('/tmp/ds4-native-bench.lock','a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 a=b.idle('native');b.verify('native',flags(1),images[1])
 stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
 try:r=one(b.bench.key,1,thinking_override='high')
 finally:stop.set();t.join()
 z=b.idle('native')
 print(r.get('graded_correct'),r.get('error'),len(r['turns']),flush=True)
 r['isolated']=z['requests_started']-a['requests_started']==len(r['turns'])
 (p/'task.json').write_text(json.dumps(r))
