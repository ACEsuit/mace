"""Check existing runner access; never print credentials, URLs, or weight data."""
import json
from pathlib import Path
import sys
from huggingface_hub import get_hf_file_metadata, hf_hub_url, try_to_load_from_cache

out={}
for filename in ['uma-s-1p1.pt','uma-m-1p1.pt']:
    cached=try_to_load_from_cache('facebook/UMA',filename)
    row={'already_cached':isinstance(cached,str) and Path(cached).is_file()}
    try:
        meta=get_hf_file_metadata(hf_hub_url('facebook/UMA',filename))
        row.update(download_access=True,bytes=meta.size,commit=meta.commit_hash)
    except Exception as exc:
        response=getattr(exc,'response',None)
        row.update(download_access=False,error_type=type(exc).__name__,
                   http_status=getattr(response,'status_code',None))
    out[filename]=row
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
