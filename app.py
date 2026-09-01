import os
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
from sobol import draw_letter, run_ga
import time
import base64
from io import BytesIO

IMG_W, IMG_H = 400, 400

CSS = """
footer{display:none !important}
#preview-col{position:sticky;top:16px;align-self:start;z-index:5;background:var(--background-fill-primary)}
#evo-canvas{width:100% !important;max-width:400px !important;height:auto !important;aspect-ratio:1/1;border-radius:12px;background:#1e1e1e;display:block;margin:auto;image-rendering:auto}
#delta-json{display:none !important} /* keep in DOM for JS observer but hide visually — visible=False breaks observer */
@media(max-width:768px){
  .gradio-container{padding:8px !important;max-width:100% !important}
  #preview-col{position:static !important;top:auto !important}
  .gradio-container h1{font-size:1.4rem !important;margin:0 0 4px !important}
  .gradio-container .prose{margin-bottom:4px !important}
  .gr-form,.gr-box,.gr-panel{gap:8px !important}
  .gr-accordion{margin-bottom:8px !important}
}
"""

JS = r"""
async () => {
  const waitFor = (sel, timeout=300) => new Promise((res, rej)=>{
    const start=Date.now();
    const iv=setInterval(()=>{
      const el=document.querySelector(sel);
      if(el){clearInterval(iv);res(el);}
      else if(Date.now()-start>timeout){clearInterval(iv);rej();}
    },10);
  });
  try{
    const canvas = await waitFor('#evo-canvas');
    const ctx = canvas.getContext('2d');
    if(!ctx){ console.error("Canvas 2d context not found"); return; }
    console.log("done_ctx");
    // verify canvas is working - draw test pixel and check
    ctx.fillStyle="#1e1e1e"; ctx.fillRect(0,0,canvas.width,canvas.height);
    console.log("done_background");
    // test draw - small red dot at 5,5 to prove canvas works
    ctx.fillStyle="rgba(255,0,0,1)"; ctx.fillRect(5,5,10,10);
    console.log("done_drawing");
    const testPixel = ctx.getImageData(5,5,1,1).data;
    const canvasOk = testPixel[0]===255;
    console.log("Canvas verify:", canvasOk ? "OK - canvas is drawing" : "FAIL", testPixel);
    // update status box to show verify
    const statusEl = document.getElementById('evo-status');
    if(statusEl){
      const orig = statusEl.value || "";
      statusEl.value = canvasOk ? "Canvas ready ✓" : "Canvas error ✗";
      setTimeout(()=>{ statusEl.value = orig; }, 1500);
    }
    // expose download helper
    window._evoDownload = (ext)=>{
      const a=document.createElement('a');
      const mime = ext==='jpg' ? 'image/jpeg' : 'image/png';
      const fname = ext==='jpg' ? 'genetic-davinci.jpg' : 'genetic-davinci.png';
      a.download=fname;
      try{
        a.href=canvas.toDataURL(mime, ext==='jpg'?0.92:1.0);
      }catch(e){
        console.error("toDataURL failed",e);
        return;
      }
      document.body.appendChild(a);
      a.click();
      setTimeout(()=>a.remove(), 100);
      console.log("Downloaded",fname);
    };
    // also expose for manual console test: _evoTest()
    window._evoTest = ()=>{
      ctx.fillStyle="rgba(0,255,0,0.8)";
      ctx.beginPath(); ctx.arc(200,200,30,0,Math.PI*2); ctx.fill();
      console.log("Test circle drawn at 200,200");
      return "test drawn";
    };

    let deltaEl = null;
    try{ deltaEl = await waitFor('#delta-json'); }catch(e){}
    if(!deltaEl) deltaEl = document.querySelector('[data-testid="json"]');
    if(!deltaEl) deltaEl = document.getElementById('delta-json');
    // fallback: Gradio JSON with visible=False may be hidden but still in DOM as display:none
    // try broader search
    if(!deltaEl){
      const all = document.querySelectorAll('div');
      for(const d of all){
        if(d.id && d.id.includes('delta')){ deltaEl=d; break; }
      }
    }
    const getDeltaContent = () => {
      if(!deltaEl) return null;
      let txt = deltaEl.textContent || deltaEl.innerText || "";
      const pre = deltaEl.querySelector('pre');
      if(pre) txt = pre.textContent;
      const ta = deltaEl.querySelector('textarea');
      if(ta) txt = ta.value;
      // also check code element
      const code = deltaEl.querySelector('code');
      if(code) txt = code.textContent;
      txt = (txt||"").trim();
      if(!txt || txt==='null' || txt==='""' || txt==='{}') return null;
      return txt;
    };

    window._evoReset = (bg)=>{
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.fillStyle=bg||"#1e1e1e";
      ctx.fillRect(0,0,canvas.width,canvas.height);
      // verify not empty after reset
      console.log("Canvas reset to",bg);
    };
    window._evoDraw = (s)=>{
      if(!s || !s.shape_type) return;
      const a = (s.a!=null ? s.a : 120)/255;
      ctx.globalAlpha = a;
      ctx.fillStyle = `rgb(${s.r|0},${s.g|0},${s.b|0})`;
      ctx.strokeStyle = ctx.fillStyle;
      if(s.shape_type==='circle'){
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.R, 0, Math.PI*2);
        ctx.fill();
      } else if(s.shape_type==='line'){
        ctx.beginPath();
        ctx.moveTo(s.x1, s.y1);
        ctx.lineTo(s.x2, s.y2);
        ctx.lineWidth = Math.max(1, s.thickness||1);
        ctx.lineCap='round';
        ctx.stroke();
      } else if(s.shape_type==='triangle'){
        ctx.beginPath();
        ctx.moveTo(s.x1, s.y1);
        ctx.lineTo(s.x2, s.y2);
        ctx.lineTo(s.x3, s.y3);
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1.0;
      // verify pixel at shape center changed (for debugging)
      // console.log("Drew", s.shape_type, "at", s.x||s.x1, s.y||s.y1);
    };

    window._evoReset("rgb(30,30,30)");
    console.log("Canvas ready, deltaEl:", !!deltaEl, deltaEl ? deltaEl.id || deltaEl.className : "none");

    if(deltaEl){
      let last = "";
      let drawCount = 0;
      const obs = new MutationObserver(()=>{
        const txt = getDeltaContent();
        if(!txt || txt===last) return;
        last = txt;
        try{
          const j = JSON.parse(txt);
          if(j && j._type==='init'){
            window._evoReset(j.bg);
            drawCount=0;
            console.log("Init bg", j.bg);
          } else if(j && j.shape_type){
            window._evoDraw(j);
            drawCount++;
            if(drawCount % 10 === 0) console.log(`Drew ${drawCount} shapes, last ${j.shape_type}`);
            // verify canvas not empty every 20 shapes
            if(drawCount % 20 === 0){
              const d = ctx.getImageData(200,200,1,1).data;
              console.log(`Verify canvas pixel @200,200: [${d[0]},${d[1]},${d[2]},${d[3]}]`);
            }
          } else if(j && j._type==='clear'){
            window._evoReset(j.bg);
            drawCount=0;
          }
        }catch(e){
          // ignore
        }
      });
      obs.observe(deltaEl, {childList:true, characterData:true, subtree:true, attributes:true, attributeOldValue:true});
      setInterval(()=>{
        const txt = getDeltaContent();
        if(txt && txt!==last){
          last=txt;
          try{
            const j=JSON.parse(txt);
            if(j && j.shape_type) window._evoDraw(j);
            else if(j && j._type==='init') window._evoReset(j.bg);
          }catch(e){}
        }
      }, 80);
      console.log("Observer attached to", deltaEl);
    } else {
      console.error("Delta JSON element not found - canvas will stay empty! Check elem_id");
      // fallback: show error in status
      const st = document.querySelector('#evo-status textarea') || document.getElementById('evo-status');
      if(st) st.value = "Error: canvas bridge not found";
    }
  }catch(e){
    console.error("evo canvas init failed", e);
  }
}
"""

def get_default_target():
    # Script-relative so the app works regardless of the current working dir.
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "assets", "sample_target.jpg")
    img = Image.open(img_path).convert("RGBA")
    return img.resize((IMG_W, IMG_H), Image.LANCZOS)


def run_optimization(target_img, use_max_shapes, max_shapes, population_size,
                     alpha, shape_types, progress=gr.Progress(track_tqdm=False)):
    if target_img is None:
        target_img = get_default_target()

    target = target_img.resize((IMG_W, IMG_H), Image.LANCZOS).convert("RGBA")
    population_size = int(population_size)
    max_shapes = int(max_shapes) if use_max_shapes and max_shapes is not None else None
    alpha = int(alpha)
    if not shape_types:
        shape_types = ["circle", "triangle", "line"]

    # compute avg bg for canvas init (same as sobol.py avg_color)
    try:
        arr = np.array(target.convert("RGB"))
        avg = arr.mean(axis=(0,1)).astype(int)
        bg_css = f"rgb({avg[0]},{avg[1]},{avg[2]})"
    except:
        bg_css = "rgb(30,30,30)"

    # instant init — single small JSON, not PNG
    progress(0, desc="Starting...")
    yield {"_type": "init", "bg": bg_css}, "Ready — evolving..."

    t0 = time.perf_counter()
    for best_img, loss, shape_count, shapes, _ in run_ga(
        target, "arial.ttf", 64, max_shapes, population_size, IMG_W, IMG_H, alpha=alpha, shape_types=shape_types,verbose=False
    ):
        if not shapes:
            continue
        elapsed = time.perf_counter() - t0
        status = f"Shape {shape_count}/{max_shapes if max_shapes else '∞'} | loss={loss:.4f} | {elapsed:.1f}s"
        progress(shape_count / (max_shapes or 100), desc=status)
        # shapes[-1] is already the delta we need — ~80 bytes JSON vs 60KB PNG
        s = shapes[-1].copy()
        s["_status"] = status
        s["_type"] = "delta"
        # ensure required keys for canvas
        # s already has r,g,b,a,shape_type,x,y,x1,y1,x2,y2,x3,y3,R,thickness
        yield s, status


with gr.Blocks(fill_width=True, css=CSS) as demo:
    gr.Markdown("# Genetic Da Vinci")
    gr.Markdown("Upload an image and evolve it with circles, triangles & lines. Canvas draws live — 750× less data than PNG.")
    with gr.Row():
        with gr.Column(elem_id="controls-col"):
            target_input = gr.Image(label="Target Image", type="pil", value=get_default_target(), height=320)
            with gr.Accordion("Settings", open=False):
                use_max_shapes = gr.Checkbox(label="Limit max shapes", value=True)
                max_shapes = gr.Number(value=2500, label="Max shapes", precision=0, minimum=1, maximum=20000)
                population_size = gr.Slider(4, 80, value=24, step=2, label="Population", info="12 fast / 24 balanced / 50 slow")
                alpha = gr.Slider(40, 180, value=120, step=5, label="Opacity", info="60 transparent → 120 opaque")
                shape_types = gr.CheckboxGroup(choices=["circle", "triangle", "line"], value=["circle", "triangle", "line"], label="Shape types")
                with gr.Row():
                    btn_fast = gr.Button("⚡ Fast (500 shapes, pop 12)", size="sm")
                    btn_bal = gr.Button("Balanced (150 shapes, pop 24)", size="sm")
                    btn_qual = gr.Button("High (500 shapes, pop 50)", size="sm")
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary", size="lg")
                stop_btn = gr.Button("Stop", variant="stop", size="lg")
            status = gr.Textbox(label="Status", value="Ready", interactive=False, lines=1, elem_id="evo-status")

        with gr.Column(elem_id="preview-col"):
            # Live canvas — draws deltas via JS, not gr.Image src
            canvas_html = gr.HTML(value='<canvas id="evo-canvas" width="400" height="400" style="width:100%;max-width:400px;height:auto;border-radius:12px;background:#1e1e1e;display:block;margin:auto;"></canvas>', label="Live Canvas")
            # Hidden JSON bridge — observed by JS (must be visible=True to stay in DOM, hidden via CSS)
            delta = gr.JSON(value=None, visible=True, elem_id="delta-json")
            with gr.Row():
                dl_png = gr.Button("Download PNG", size="sm", variant="secondary")
                dl_jpg = gr.Button("Download JPG", size="sm", variant="secondary")
                test_btn = gr.Button("Test Canvas", size="sm")
            # Fallback hidden image for download (throttled, not live)
            # We keep no gr.Image live to avoid PNG flicker

    # presets
    btn_fast.click(lambda: (500, 12, 80), outputs=[max_shapes, population_size, alpha])
    btn_bal.click(lambda: (1000, 24, 120), outputs=[max_shapes, population_size, alpha])
    btn_qual.click(lambda: (2500, 50, 140), outputs=[max_shapes, population_size, alpha])

    run_event = run_btn.click(
        run_optimization,
        inputs=[target_input, use_max_shapes, max_shapes, population_size, alpha, shape_types],
        outputs=[delta, status],
        show_progress="hidden",
    )
    stop_btn.click(fn=lambda: ({"_type":"clear","bg":"rgb(30,30,30)"}, "Stopped"), inputs=None, outputs=[delta, status], cancels=[run_event])
    # verification + download — pure JS, no Python roundtrip
    dl_png.click(None, js="() => window._evoDownload('png')")
    dl_jpg.click(None, js="() => window._evoDownload('jpg')")
    test_btn.click(None, js="() => window._evoTest()")
    demo.load(None, None, None, js=JS)

# Runs locally on the default localhost address; no tunnel / share / hardcoded port.
demo.queue().launch(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", spacing_size="sm", radius_size="md", text_size="sm"))
