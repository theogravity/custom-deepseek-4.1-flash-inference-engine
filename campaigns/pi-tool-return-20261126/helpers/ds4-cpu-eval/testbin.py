import sys,json
from pathlib import Path
for l in Path(sys.argv[1]).read_text().splitlines():
 try:a=json.loads(l)
 except ValueError:continue
 if a.get('profile',{}).get('test') and a.get('executable'):
  print(a['executable']);break
else:raise Exception('no test binary')
