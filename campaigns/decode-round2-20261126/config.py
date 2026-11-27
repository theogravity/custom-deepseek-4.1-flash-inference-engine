from pathlib import Path
import json,sys
p=Path('/tmp/ds4-cycle2-1126')
name,flag=sys.argv[1:3];inherited=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
(p/name/'config.json').write_text(json.dumps([[k,'ds4-flash-rs:c1-'+name,inherited|{flag:v},['edit','long'],6] for k,v in [('off-a','0'),('on','1'),('off-b','0')]],indent=2))
