from pathlib import Path
import json,hashlib,sys
p=Path(__file__).resolve().parent
def bodies():
    initial=json.loads(Path('/tmp/ds4-six.pFsc06/404k-histories.json').read_text())['sessions'][0]['messages']
    asks=[
        'Inspect the test.',
        'Explain the parser bug and suggest a fix.',
        'List the file names and give a proposed test case.',
        'Return a small Python example containing an escaped option followed by an argument.',
        'Explain why the test failed. Suggest a minimal fix and give two edge cases for the reviewer.',
        'Summarize the earlier debugging history and identify the remaining bug. Name the failing case, the relevant parser branch, and one way to test the fix.',
        'Review the proposed shell argument parser. Describe in four numbered steps how to handle escaped quotes, adjacent arguments, empty strings, and an unset environment variable. Include one concrete command and its expected result.',
    ]
    return [{'model':'DeepSeek V4.1 Flash','reasoning_effort':'none','temperature':0,'max_tokens':16,
             'stream':True,'stream_options':{'include_usage':True},'messages':initial+[{'role':'user','content':q}]} for q in asks]
def sha(body):return hashlib.sha256(json.dumps(body).encode()).hexdigest()
if __name__=='__main__':
    sys.path.insert(0,'/home/benchmark/llm/ds4-flash-rs/tools')
    from six_streams import load_tokenizer,chat_ids,id_digest
    tok=load_tokenizer(Path('/mnt/models/dsv41-flash-exl3/tokenizer.json'))
    rows=[]
    for q in bodies():
        ids=chat_ids(tok,q['messages'])
        rows.append(dict(wire=sha(q),tokens=len(ids),ids=id_digest(ids)))
    assert len({r['tokens'] for r in rows})==7
    (p/'changing-reference.json').write_text(json.dumps(rows,indent=2)+'\n')
