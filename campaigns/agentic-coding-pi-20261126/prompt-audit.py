"""Compare the exact Pi tool-return prompt with the pack's reference encoder, on CPU."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
from tempfile import TemporaryDirectory
here=Path(__file__).resolve().parent
engine=Path('/home/benchmark/llm/ds4-flash-rs')
pack=Path('/mnt/models/dsv41-flash-exl3')
sys.path.insert(0,str(pack/'encoding'))
import encoding
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    source=here/'frozen-red-turn.json'
    request=json.loads(source.read_bytes())
    wire=sha(json.dumps(request).encode())
    messages=request['messages']
    messages[0]['tools']=request['tools']
    reference=encoding.encode_messages(messages,thinking_mode='thinking',reasoning_effort=request['chat_template_kwargs']['reasoning_effort'])
    outdirs=list((engine/'target/release/build').glob('*/*/out'))
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/'chat.rs').write_text((engine/'src/chat.rs').read_text()+'\npub fn text_for_audit(p: &Prompt) -> String { p.0.iter().map(|(s,_)|s.as_str()).collect() }\n')
        shutil.copyfile(engine/'src/tool_prompt.txt',root/'tool_prompt.txt')
        ref=root/'ref.txt';ref.write_text(reference)
        args=root/'audit.rs'
        args.write_text('''mod chat;
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let check_order: serde_json::Value=serde_json::from_str(r#"{"z":0,"a":1}"#)?;
    anyhow::ensure!(check_order.as_object().unwrap().keys().next().unwrap()=="z","helper lacks engine's preserve_order feature");
    let req=serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let full=tokenizers::Tokenizer::from_file(&args[2]).map_err(|e|anyhow::anyhow!("{e}"))?;
    let mut plain=full.clone(); plain.with_added_vocabulary(tokenizers::AddedVocabulary::new());
    let (prompt,_,_)=chat::render(&req)?;
    std::fs::write(&args[3],chat::text_for_audit(&prompt))?;
    std::fs::write(&args[4],serde_json::to_vec(&prompt.encode(&full,&plain)?)?)?;
    let reference=std::fs::read_to_string(&args[5])?;
    let tokens=full.encode(reference,false).map_err(|e|anyhow::anyhow!("{e}"))?;
    std::fs::write(&args[6],serde_json::to_vec(tokens.get_ids())?)?;
    Ok(())
}
''')
        cmd=['rustc','--edition=2021','-A','dead_code','-o',str(root/'audit'),str(args)]
        for d in outdirs:cmd+=['-L','dependency='+str(d),'-L','native='+str(d)]
        for name in ['serde_json','anyhow','tokenizers']:
            lib=sorted((engine/'target/release/build'/name).glob('*/out/lib'+name+'-*.rlib'))[-1]
            cmd+=['--extern',name+'='+str(lib)]
        subprocess.run(cmd,check=True)
        native=root/'native.txt';ids=root/'native.ids';refids=root/'ref.ids'
        subprocess.run([root/'audit',source,pack/'tokenizer.json',native,ids,ref,refids],check=True)
        got=json.loads(ids.read_bytes());expected=json.loads(refids.read_bytes())
        fmt=lambda seq:sha(b''.join(i.to_bytes(4,'little') for i in seq))
        result={'original_wire_sha256':wire,'tokenizer_sha256':sha((pack/'tokenizer.json').read_bytes()),'tokens_equal':got==expected,'reference_encoder_sha256':sha((pack/'encoding/encoding.py').read_bytes()),'text_equal':native.read_bytes()==ref.read_bytes(),'native_token_count':len(got),'reference_token_count':len(expected),'native_ids_sha256':fmt(got),'reference_ids_sha256':fmt(expected)}
        (here/'prompt-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
        if not result['text_equal'] or got!=expected:
            import difflib
            print('Divergent lines / opcodes:',[(o,a,b,c,d) for o,a,b,c,d in difflib.SequenceMatcher(None,reference.splitlines(),native.read_text().splitlines(),autojunk=False).get_opcodes() if o!='equal'])
            raise SystemExit(1)
if __name__=='__main__':main()
