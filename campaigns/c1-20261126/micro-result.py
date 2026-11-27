from pathlib import Path
import statistics as st,json,hashlib,sys
p=Path(__file__).resolve().parent/sys.argv[1];raw=p.joinpath('micro.txt').read_bytes();rows=[l.split() for l in raw.splitlines()]
assert len(rows)==2*8*2*9
cells={}
for proj,jobs,nrows,mode,rep,us in rows:
 k=(int(proj),int(jobs),int(nrows),int(mode));c=cells.setdefault(k,{});assert int(rep) not in c;c[int(rep)]=float(us)
assert all(len(c)==9 for c in cells.values())
out=[]
for pr,j,n,mode in sorted(cells):
 if mode:continue
 a,z=[st.median(cells[pr,j,n,k].values()) for k in (0,1)]
 out.append(dict(projection='gate+up' if pr==0 else 'down',jobs=j,rows=n,control_us=a,shuffle_us=z,change_pct=round((z/a-1)*100,2)))
(p/'micro-summary.json').write_text(json.dumps(dict(method="CUDA events on GPU 1, alternating mode order, nine samples of 40 sequential launches, median microseconds per launch. Full FP32 partial buffers compared bitwise for baseline vs shuffle before each shape; five-job fixture covers 1,8,9,15,16 routed rows per expert. Synthetic repeated weights, not serving throughput.",raw_sha256=hashlib.sha256(raw).hexdigest(),cells=out),indent=2)+'\n')
