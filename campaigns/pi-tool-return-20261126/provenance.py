from pathlib import Path
import hashlib,json,subprocess
p=Path(__file__).resolve().parent
original=p/'idle-tested.elf'
built=Path('/tmp/ds4-cpu-eval/target/release/ds4-flash-rs')
def sections(path):
    # ELF64 little-endian. Compare every allocatable section by name, size,
    # flags and bytes, excluding the compiler's generated build ID.
    import struct
    d=path.read_bytes(); assert d[:6]==b'\x7fELF\x02\x01'
    off=struct.unpack_from('<Q',d,40)[0]
    ents,n,names=struct.unpack_from('<HHH',d,58)
    def header(i):return struct.unpack_from('<IIQQQQIIQQ',d,off+ents*i)
    tab=header(names);strings=d[tab[4]:tab[4]+tab[5]];result={}
    for i in range(n):
        name,typ,flags,addr,offset,size,_,_,align,_=header(i)
        if not flags&2:continue
        key=strings[name:strings.index(b'\0',name)].decode()
        if key=='.note.gnu.build-id':continue
        result[key]=(addr,size,align,flags,None if typ==8 else hashlib.sha256(d[offset:offset+size]).hexdigest())
    return result
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
a,b=sections(original),sections(built)
diff=[k for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)]
result={'tested_elf_sha256':sha(original),'rebuilt_elf_sha256':sha(built),'allocatable_sections_except_build_id':len(a),'different_loadable_sections':diff,'runtime_source':subprocess.check_output(['git','-C',p/'idle','rev-parse','HEAD'],text=True).strip()}
(p/'provenance.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
if diff: raise SystemExit('rebuild has differing loadable sections')
