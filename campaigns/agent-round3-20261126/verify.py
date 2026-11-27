"""Check three legs and record service-level medians without mixing decode and prefill."""
from pathlib import Path
import json,sys
from statistics import median
p=Path(__file__).resolve().parent
def analyze(name):
    folder=p/(name+'-receipts')
    d=[json.loads((folder/(k+'.json')).read_text()) for k in ['a','on','b']]
    config=json.loads((p/(name+'.json')).read_text())
    assert d[0]['image']==d[2]['image'] and all(w['flags']==v[1] for w,v in zip(d,config))
    for r in d:
        assert len(r['pi']['rows'])==7 and len(r['long']['samples'])==6
    assert len({(r['hashes']['reasoning_content'],r['hashes']['content'],r['output_tokens']) for w in d for r in w['pi']['rows']})==1
    fields=[('pi','median',True),('pi','ttft',False),('long','decode',True),('long','ttft',False),('long','wall',True)]
    if name=='release':
        fields.extend([('six','decode',True),('six','wall',True)])
    if name=='text':
        assert all(r['count_tokens']['input_tokens']==d[0]['count_tokens']['input_tokens'] for r in d)
        fields.append(('count_tokens','median',False))
    out={}
    for group,metric,higher in fields:
        vals=[w[group][metric] for w in d]
        gain=[100*(vals[1]/v-1 if higher else 1-vals[1]/v) for v in [vals[0],vals[2]]]
        out[group+'_'+metric]=dict(aba=vals,gain_vs_controls_pct=gain)
    if name=='release':
        from attest import attest
        actual=attest()
        assert d[1]['image']==actual['image']
        assert min(out['long_ttft']['gain_vs_controls_pct'])>=3, out['long_ttft']
        for metric in ('pi_median','pi_ttft','long_decode','long_wall','six_decode','six_wall'):
            assert min(out[metric]['gain_vs_controls_pct'])>=-3,(metric,out[metric])
    (folder/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    return out
if __name__=='__main__':print(json.dumps(analyze(sys.argv[1]),indent=2))
