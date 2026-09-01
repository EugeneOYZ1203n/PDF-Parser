import sys, math, random, tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\BackedUp\LaptopProjects\2608_DPAIntern\PDF-Parser")
import pymupdf as fitz
from rastervec.Evaluation.Conversion.conversion import convert_page_to_vector_text
from rastervec.pipeline import run_page_context
from rastervec.Reader.reader import Reader
from rastervec.renderer import cluster_frame_size, render_vector_cluster
from rastervec.OCR.Paddle_OCR.render_ocr import MIN_RENDER_SIDE_PX
from rastervec.models import ClusterOcrResult, TextVectorResult
from rastervec.helpers.geometry import union_bbox

tmp = Path(tempfile.mkdtemp())
src = tmp/"s.pdf"; d=fitz.open(); p=d.new_page(width=400,height=300)
for i in range(6): p.insert_text((30, 40+i*35), f"Row number {i}", fontsize=13)
d.save(str(src)); d.close()
conv = tmp/"c.pdf"; conv.write_bytes(convert_page_to_vector_text(str(src),0))
with Reader(str(conv)) as r:
    ctx = run_page_context(r, 0, final_stage="text_candidates")
clusters = [c for c in (ctx.text_clusters or []) if c]
print("clusters:", len(clusters))

# fake ClusterOcrResults: alternate pass/blank
pool=[]
for i,c in enumerate(clusters):
    txt = f"Row number {i}" if i%2==0 else ""
    res = TextVectorResult(paths=c, text=txt, confidence=0.9, bbox=union_bbox([x.bbox for x in c]),
                           ocr_bbox=None, rotation_used=0, page_index=0)
    pool.append(ClusterOcrResult(cluster=c, resolved=res, ocr_seconds=0.0))

SHOWCASE_N=20; SHOWCASE_SEED=0
def render_ocr_input(cluster, dpi=300):
    w,h=cluster_frame_size(cluster); m=min(w,h)
    if m>0: dpi=max(dpi, math.ceil(MIN_RENDER_SIDE_PX*72.0/m))
    return render_vector_cluster(cluster, dpi)

passed=[r for r in pool if r.resolved.text.strip()]
failed=[r for r in pool if not r.resolved.text.strip()]
rng=random.Random(SHOWCASE_SEED); half=SHOWCASE_N//2
pick_pass=rng.sample(passed,min(half,len(passed)))
pick_fail=rng.sample(failed,min(SHOWCASE_N-len(pick_pass),len(failed)))
chosen_ids={id(r) for r in pick_pass+pick_fail}
remaining=[r for r in pool if id(r) not in chosen_ids]; rng.shuffle(remaining)
pick=pick_pass+pick_fail+remaining[:max(0,SHOWCASE_N-len(chosen_ids))]; rng.shuffle(pick)
cols=4; rows=math.ceil(len(pick)/cols)
fig,axes=plt.subplots(rows,cols,figsize=(cols*3.2,rows*3.0),squeeze=False)
for ax,result in zip(axes.flat,pick):
    ax.imshow(render_ocr_input(result.cluster))
    ax.set_xticks([]);ax.set_yticks([])
for ax in axes.flat[len(pick):]: ax.axis("off")
plt.tight_layout(); fig.savefig(str(tmp/"out.png"))
print("rendered", len(pick), "->", (tmp/"out.png").stat().st_size, "bytes  OK")
