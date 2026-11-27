from pathlib import Path
import json,sys
folder=Path(__file__).resolve().parent/sys.argv[1]
for r in json.loads((folder/'config.json').read_text()):
 assert (folder/(r[0]+'.json')).is_file()
data=[json.loads((folder/(r[0]+'.json')).read_text()) for r in json.loads((folder/'config.json').read_text())]
wins=[data[1][k]['median']/max(data[0][k]['median'],data[2][k]['median']) for k in ('edit','long')]
losses=[data[1][k]['median']/((data[0][k]['median']+data[2][k]['median'])/2) for k in ('edit','long')]
assert max(wins)>1.005 and min(losses)>0.97,(wins,losses)
