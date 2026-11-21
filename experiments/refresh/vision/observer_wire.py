"""Bounded JSON + NPZ framing; never unpickle remote data."""
import json,struct
VERSION=1
MAX_JSON=1024*1024
MAX_DATA=64*1024*1024

def read_exact(stream,n):
    out=bytearray()
    while len(out)<n:
        chunk=stream.read(n-len(out))
        if not chunk:raise EOFError('observer worker stream closed')
        out.extend(chunk)
    return bytes(out)

def read_frame(stream):
    jn,dn=struct.unpack('!II',read_exact(stream,8))
    if not 0<jn<=MAX_JSON or not 0<=dn<=MAX_DATA:raise ValueError('frame length outside protocol limits')
    meta=json.loads(read_exact(stream,jn).decode('utf-8'))
    if not isinstance(meta,dict):raise ValueError('metadata must be object')
    return meta,read_exact(stream,dn)

def write_frame(stream,meta,data=b''):
    js=json.dumps(meta,allow_nan=False,separators=(',',':')).encode('utf-8')
    if not 0<len(js)<=MAX_JSON or len(data)>MAX_DATA:raise ValueError('frame too large')
    for block in (struct.pack('!II',len(js),len(data)),js,data):
        view=memoryview(block)
        while view:
            n=stream.write(view)
            if not n:raise BrokenPipeError('observer worker short write')
            view=view[n:]
    stream.flush()

