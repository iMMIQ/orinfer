"""Read original pinned safetensors by bounded HTTP ranges or local files."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import time
import urllib.parse
import urllib.request


SIZES = {'BF16':2,'F16':2,'F32':4,'F64':8,'I8':1,'U8':1,'I16':2,'U16':2,
         'I32':4,'U32':4,'I64':8,'U64':8,'BOOL':1}


def relative_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Invalid source filename')
    return name


def validate_header(header):
    intervals = []
    for name,t in header.items():
        if name == '__metadata__':continue
        shape,offsets,dtype = t.get('shape'),t.get('data_offsets'),t.get('dtype')
        if not isinstance(shape,list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError('Invalid source tensor shape')
        if not isinstance(offsets,list) or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets):
            raise ValueError('Invalid source tensor offsets')
        if dtype not in SIZES or offsets[1]-offsets[0] != math.prod(shape)*SIZES[dtype]:
            raise ValueError('Invalid source tensor extent/dtype')
        intervals.append(tuple(offsets))
    end = 0
    for start,stop in sorted(intervals):
        if start != end:raise ValueError('Overlapping or gapped source tensor payload')
        end = stop


class Source:
    def __init__(self, index, *, directory=None, repo=None, revision=None, cache=None):
        self.weight_map = json.loads(Path(index).read_text())['weight_map']
        self.directory = Path(directory) if directory else None
        self.repo,self.revision = repo,revision
        self.cache = Path(cache) if cache else None
        if self.directory is None:
            if not repo or len(repo.split('/')) != 2 or not re.fullmatch(r'[0-9a-f]{40}',revision or ''):
                raise ValueError('Remote weights require repository and pinned commit')
        for name in self.weight_map.values():relative_name(name)
        self.headers = {}

    def read(self, filename, offset, count):
        relative_name(filename)
        if type(offset) is not int or type(count) is not int or offset < 0 or count <= 0:
            raise ValueError('Invalid byte range')
        if self.directory is not None:
            with (self.directory/filename).open('rb') as f:
                f.seek(offset);data = f.read(count)
            if len(data) != count:raise EOFError('Truncated local source')
            return data
        encoded = urllib.parse.quote(filename,safe='/')
        # Distinct query prevents a CDN from reusing a different cached range.
        url = f'https://huggingface.co/{self.repo}/resolve/{self.revision}/{encoded}?download=true&orinfer_range={offset}-{count}'
        for attempt in range(5):
            try:
                req = urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+count-1}',
                                                         'User-Agent':'orinfer-weight-converter'})
                with urllib.request.urlopen(req,timeout=90) as response:
                    cr = response.headers.get('Content-Range','')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                    if response.status != 206 or not match or tuple(map(int,match.groups()[:2])) != (offset,offset+count-1):
                        raise ValueError('Server did not return the exact requested range')
                    data = response.read(count+1)
                if len(data) != count:raise EOFError('Truncated HTTP source')
                return data
            except (OSError,ValueError):
                if attempt == 4:raise
                time.sleep(min(2**attempt,8))

    def header(self, filename):
        if filename in self.headers:return self.headers[filename]
        key = hashlib.sha256(json.dumps([self.repo,self.revision,str(self.directory),filename]).encode()).hexdigest()
        cached = self.cache/f'{key}.json' if self.cache else None
        if cached and cached.exists():
            item = json.loads(cached.read_text());begin,header = item['begin'],item['header']
        else:
            size, = struct.unpack('<Q',self.read(filename,0,8))
            if not 2 <= size <= 64*1024**2:raise ValueError('Invalid safetensors header length')
            raw = self.read(filename,8,size)
            begin,header = 8+size,json.loads(raw)
            if cached:
                cached.parent.mkdir(parents=True,exist_ok=True)
                temporary = cached.with_suffix('.tmp')
                temporary.write_text(json.dumps({'begin':begin,'header':header}))
                temporary.replace(cached)
        validate_header(header)
        self.headers[filename] = begin,header
        return begin,header

    def tensor(self, name):
        filename = self.weight_map[name]
        begin,header = self.header(filename)
        return filename,begin,header[name]

    def rows(self, name, start, count):
        filename,begin,t = self.tensor(name)
        shape = t['shape']
        if not shape or type(start) is not int or type(count) is not int or count <= 0 or start < 0 or start+count > shape[0]:
            raise ValueError('Invalid tensor row range')
        row_bytes = math.prod(shape[1:])*SIZES[t['dtype']]
        return self.read(filename,begin+t['data_offsets'][0]+start*row_bytes,count*row_bytes)
