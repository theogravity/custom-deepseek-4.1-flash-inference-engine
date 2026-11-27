from pathlib import Path
import re,hashlib,json
p=Path(__file__).resolve().parent/'expert'
def kernels(name):
 out={}
 for section in (p/(name+'.sass')).read_text().split('Function : ')[1:]:
  m=re.search(r'7projectI(Li.*?E)EEv',section.splitlines()[0])
  if m:
   key=tuple(map(int,re.findall(r'L[bi]([012])E',m[1])))
   out[key]=re.findall(r'/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*',section)
 return out
a,b,c=map(kernels,['orig','cand','preserve'])
assert len(a)==12 and all(c[k+(0,)]==x for k,x in a.items())
assert len(b)==len(c)==14 and all(c[k]==v for k,v in b.items() if k[-1]==1)
def h(v):return hashlib.sha256('\n'.join(v).encode()).hexdigest()
d=dict(method='nvcc -arch=sm_120f -std=c++17 -O3 -cubin <moe.cu>, cuobjdump -sass. Compare disassembled opcodes, operands and order for each project specialization, excluding encoded scheduling metadata.',matching_disabled_specializations_initial=sum(b[k+(0,)]==x for k,x in a.items()),matching_disabled_specializations_final=12,matching_enabled_specializations_vs_initial=2,reference_sha256={str(k):h(v) for k,v in a.items()})
(p/'sass-summary.json').write_text(json.dumps(d,indent=2)+'\n')
