from pathlib import Path
import subprocess as sp,json
p=Path(__file__).resolve().parent
wanted={'text':[1,0,0],'markov':[1,1,0],'hash':[1,0,1]}
out={}
for name,goal in wanted.items():
    image='ds4-flash-rs:agent3-'+name
    ident=sp.check_output(['docker','image','inspect','-f','{{.Id}}',image],text=True).strip()
    arms=['on'] if name=='text' else ['a','on','b']
    assert all(json.loads((p/(name+'-receipts')/(a+'.json')).read_text())['image']==ident for a in arms)
    if name=='text':
        assert all(json.loads((p/'changing-receipts'/(a+'.json')).read_text())['image']==ident for a in ['a','on','b'])
    found=[]
    for control in ['DS4_AGENT_TEXT_CACHE','DS4_MARKOV_PAIRS','DS4_BULK_PREFIX_KEYS']:
        status=sp.run(['docker','run','--rm','--entrypoint','/usr/bin/grep',ident,'-aq',control,'/usr/local/bin/ds4-flash-rs']).returncode
        assert status in (0,1),(ident,control,status)
        found.append(status==0)
    assert found==goal,(name,found)
    out[name]={'image':ident,
               'elf_sha256':sp.check_output(['docker','run','--rm','--entrypoint','/usr/bin/sha256sum',ident,'/usr/local/bin/ds4-flash-rs'],text=True).split()[0],
               'text_markov_hash_flags_present':found}
(p/'variant-identity.json').write_text(json.dumps(out,indent=2)+'\n')
