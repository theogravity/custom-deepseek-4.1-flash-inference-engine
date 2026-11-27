from collections import defaultdict, Counter
from pathlib import Path
from statistics import median
import hashlib, json, sys, sqlite3
p=Path(__file__).resolve().parent
repo=Path('/home/benchmark/llm/ds4-flash-rs')
def load(path):
    return json.loads(path.read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def profile(n):
    q=p/'timeline'
    db=q/f'profile-c{n}.sqlite'
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    start,end=c.execute('select min(start),max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    pad=(end-start)//20
    a,b=start+pad,end-pad
    span=b-a
    books=defaultdict(lambda:defaultdict(int))
    shapes=defaultdict(Counter)
    edges=defaultdict(Counter)
    family_edges=defaultdict(Counter)
    for start,end,device,name,gx,gy,gz in c.execute('select k.start,k.end,k.deviceId,s.value,k.gridX,k.gridY,k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where k.start < ? and k.end > ?',(b,a)):
        l,r=max(a,start),min(b,end)
        edges[l][device]+=1;edges[r][device]-=1
        family_edges[l][name]+=1;family_edges[r][name]-=1
        row=books[name];row['calls']+=1;row['ns']+=r-l
        row[device]+=r-l
        shapes[name][(gx,gy,gz)]+=r-l
    ids=sorted({i for v in edges.values() for i in v})
    assert len(ids)==2
    state=Counter()
    occupancy=Counter()
    last=a
    for t,change in sorted(edges.items()):
        active=sum(1 for device in ids if state[device]>0)
        occupancy[active]+=t-last
        for device in ids:
            if state[device]>0:occupancy[f'device_{device}']+=t-last
        state.update(change)
        assert min(state.values())>=0
        last=t
    occupancy[0]+=b-last
    assert all(state[i]==0 for i in ids) and sum(occupancy[i] for i in [0,1,2])==span
    families=Counter()
    exclusively=Counter()
    solo=Counter()
    overlap_with_project=Counter()
    last=a
    active_by_device=Counter()
    for t,change in sorted(family_edges.items()):
        if len(families)==1:
            family=next(iter(families))
            exclusively[family]+=t-last
            if sum(v>0 for v in active_by_device.values())==1:
                solo[family]+=t-last
        if 'project' in families:
            for family in families:
                if family!='project':
                    overlap_with_project[family]+=t-last
        families.update(change); families=+families
        active_by_device.update(edges[t])
        last=t
    tot=sum(x['ns'] for x in books.values())
    summary={}
    for name,v in sorted(books.items(),key=lambda item:-item[1]['ns']):
        summary[name]=dict(calls=v['calls'],gpu_ms=v['ns']/1e6,pct_sum_kernel=100*v['ns']/tot,exclusive_family_wall_pct=100*exclusively[name]/span,exclusive_one_gpu_wall_pct=100*solo[name]/span,overlap_with_expert_wall_pct=100*overlap_with_project[name]/span,per_device_gpu_ms={str(i):v[i]/1e6 for i in ids})
    apis={}
    for name,count,ns in c.execute('select s.value,count(*),sum(min(r.end,?)-max(r.start,?)) as dur from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.start < ? and r.end > ? group by 1 order by dur desc limit 12',(b,a,b,a)):
        apis[name]=dict(calls=count,host_ms=ns/1e6)
    copies={}
    for label,count,byte,ns in c.execute('select e.label,count(*),sum(m.bytes),sum(m.end-m.start) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id where m.start>=? and m.end<=? group by 1',(a,b)):
        copies[label]=dict(complete_operations=count,mib=byte/2**20,sum_gpu_ms=ns/1e6)
    expert_transfers=[]
    sizes=(13271040,44544)
    for size in sizes:
        count,total,elapsed=c.execute('select count(*),coalesce(sum(bytes),0),coalesce(sum(end-start),0) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and start>=? and end<=? and bytes=?',(a,b,size)).fetchone()
        expert_transfers.append(dict(operation_bytes=size,count=count,total_bytes=total,sum_gpu_ms=elapsed/1e6))
    transfer_edges=Counter()
    for l,r in c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and start>=? and end<=? and bytes in (?,?)',(a,b,*sizes)):
        transfer_edges[l]+=1; transfer_edges[r]-=1
    copy_active=kernel_active=copy_any=copy_only=0
    last=a
    for t in sorted(edges.keys() | transfer_edges.keys()):
        if copy_active:
            copy_any+=t-last
            if not kernel_active: copy_only+=t-last
        copy_active+=transfer_edges[t]
        kernel_active+=sum(edges[t].values())
        last=t
    diag=[r[0] for r in c.execute("select text from DIAGNOSTIC_EVENT where (text like '%incomplete CUPTI%' or text like '%not all CUDA events%')")] if 'text' in [r[1] for r in c.execute("pragma table_info(DIAGNOSTIC_EVENT)")] else []
    scope=load(q/f'profile-c{n}-scope.json')
    clock_ns=c.execute('select systemClockNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    bounds=[(clock_ns+v)/1e9 for v in [a,b]]
    monitored=[scope['first']['t'],scope['last']['t']]
    print('GPU trace and scheduler bounds',n,bounds,monitored,file=sys.stderr)
    assert monitored[0]<=bounds[0] and bounds[1]<=monitored[1]
    assert all(x['active']==n and x['decoding']==n for x in scope['observations'])
    max_poll_gap=max(y['t']-x['t'] for x,y in zip(scope['observations'],scope['observations'][1:]))
    control=[json.loads(l) for l in (p/'rust-a.jsonl').read_text().splitlines()][1]
    def hashes(w):
        assert all(r['first_function_present'] and r['usage']['completion_tokens']==1500 for r in w['rows'])
        return [[r['content_sha256'],r['reasoning_sha256']] for r in w['rows']]
    got=hashes(control)
    traced=[json.loads(l) for l in (q/'profile-c6.out').read_text().splitlines()]
    assert len(traced)==2 and all(hashes(w)==got for w in traced)
    return dict(raw_report_sha256=digest(q/f'profile-c{n}.nsys-rep'),window_ms=span/1e6,window_rule='Central 90% of the first-to-last kernel interval. Clip intersecting activity durations. Copy counts and bytes use complete operations.',output_sha256=got,monotonic_window_s=bounds,monotonic_scope_s=monitored,max_scheduler_sample_gap_s=max_poll_gap,workers_observed=n,output_budget=1500,scope_scheduler_samples=scope['sample_n'],scope_first=scope['first'],scope_last=scope['last'],device_ids=ids,kernel_wall_percent=dict(neither=100*occupancy[0]/span,exactly_one=100*occupancy[1]/span,both=100*occupancy[2]/span,per_device={str(i):100*occupancy[f'device_{i}']/span for i in ids}),sum_kernel_ms=tot/1e6,kernels=summary,runtime=apis,copies=copies,expert_size_h2d=expert_transfers,expert_size_h2d_wall_ms=copy_any/1e6,expert_size_h2d_without_recorded_kernel_wall_ms=copy_only/1e6,diagnostics=diag,top_kernel_grids={name:[dict(grid=list(k),sum_gpu_ms=v/1e6) for k,v in shapes[name].most_common(5)] for name in list(summary)[:5]})

if __name__=='__main__':
    d=profile(6)
    (p/'edit-profile.json').write_text(json.dumps(d,indent=2))
    print(d['window_ms'],d['kernel_wall_percent'])
    for k,v in list(d['kernels'].items())[:6]:print(k,v['pct_sum_kernel'])
