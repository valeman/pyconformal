"""Stream the SGSC interval readings out of the 7z archive without writing the full CSV to disk.
Keeps general-supply readings between two dates."""
import sys, py7zr, io, time
from py7zr import Py7zIO, WriterFactory
LO, HI = sys.argv[1], sys.argv[2]            # e.g. 2012-12-20 2013-04-02
class Sink(Py7zIO):
    def __init__(self, out): self.out = out; self.tail = b""; self.n = 0; self.kept = 0; self.header = None; self.t = time.time()
    def write(self, b):
        data = self.tail + bytes(b); cut = data.rfind(b"\n"); self.tail = data[cut + 1:]; block = data[:cut]
        lines = block.split(b"\n")
        if self.header is None:
            self.header = lines[0].decode().strip().split(","); print("header:", self.header, flush=True); print("first row:", lines[1][:200], flush=True)
            self.i_date = self.header.index("READING_DATETIME"); self.out.write(lines[0] + b"\n"); lines = lines[1:]
        lo, hi = LO.encode(), HI.encode(); keep = []
        for ln in lines:
            f = ln.split(b",", self.i_date + 1)
            d = f[self.i_date][:10]
            if lo <= d <= hi: keep.append(ln)
        self.n += len(lines); self.kept += len(keep)
        if keep: self.out.write(b"\n".join(keep) + b"\n")
        if self.n % 5_000_000 < len(lines): print(f"{self.n:,} rows read, {self.kept:,} kept, {time.time() - self.t:.0f} s", flush=True)
        return len(b)
    def read(self, size=None): return b""
    def seek(self, offset, whence=0): return 0
    def flush(self): pass
    def size(self): return 0
class F(WriterFactory):
    def __init__(self, s): self.s = s
    def create(self, filename): print("member:", filename, flush=True); return self.s
with open("window.csv", "wb") as out, py7zr.SevenZipFile("readings.7z") as z:
    print([(i.filename, i.uncompressed) for i in z.list()][:3], flush=True)
    s = Sink(out); z.extractall(factory=F(s)); print("done", s.n, s.kept)
