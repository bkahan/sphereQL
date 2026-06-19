// SphereQL viewer runtime — extracted from template.html so the baked offline
// HTML and the WASM studio share one implementation. Inlined verbatim into the
// emitted page at the /*__SPHEREQL_VIEWER__*/ placeholder, immediately after
// the module-scope `const D` scene payload. Plain browser script (no modules):
// it reads the globals `D` and `THREE` plus the page DOM.
//
// Architecture: the persistent THREE objects (renderer / scene / camera /
// controls / lights / raycaster) and every static DOM event binding are
// created ONCE. Everything that depends on the scene *data* — points, overlays,
// legend, labels, stats — lives in module-level `let`s that `rebuild(sc)`
// reassigns. `teardown()` disposes the previous scene's GPU buffers and clears
// its generated DOM first, so the viewer can swap to any Scene at runtime (a
// dropped file, or a live WASM pipeline) without leaking or reloading the page.
// The baked page simply calls `rebuild(D)` once at boot.

// ── Constants & helpers (persistent) ─────────────────────────────────────
const escHtml=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const fmin=a=>a.reduce((m,v)=>v<m?v:m,Infinity);
const fmax=a=>a.reduce((m,v)=>v>m?v:m,-Infinity);
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
const reduceMotion=matchMedia("(prefers-reduced-motion:reduce)").matches;
const DEF={scale:12,radial:1,spread:1,size:3.5,globe:true,autorot:false,ui:1,palette:"aurora",zoom:0.5,density:false};
const PALETTES={
  aurora:["#5cc8ff","#ff8a65","#86e0a8","#c79bff","#ffd95c","#ff7fa8","#4dd0e1","#bfa07a","#9fb2d4","#b6e07a","#8aa0ff","#ffb454","#ff6f6f","#74b9ff","#dce07a","#a98bff"],
  spectral:["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2","#f1a340","#998ec3"],
  viridis:["#440154","#472d7b","#3b528b","#2c728e","#21918c","#28ae80","#5ec962","#addc30","#fde725","#a0da39","#35b779","#31688e"],
  sunset:["#f9c74f","#f9844a","#f8961e","#f3722c","#f94144","#ff6f91","#c9457e","#9b5de5","#f15bb5","#fee440","#ff9b54","#e07a5f"],
  ice:["#caf0f8","#90e0ef","#48cae4","#00b4d8","#0096c7","#5e60ce","#6930c3","#7400b8","#80ffdb","#56cfe1","#64dfdf","#4ea8de"]
};
const classColor={Genuine:0xeaf2ff,Weak:0x7a86a8,OverlapArtifact:0xff7a7a};
const OVERLAY_LABELS={centroid:"Centroids",bridge:"Bridges",geodesic_path:"Geodesic paths",voronoi_cap:"Voronoi caps",antipode:"Antipodes",coverage_void:"Coverage / void",domain_group:"Domain groups",glob:"Globs",manifold_slice:"Manifold slice"};
const LABEL_KIND_NAMES={centroid:"Centroids",antipode:"Antipodes",domain_group:"Domain groups"};
const overlayDefaultOff=new Set(["voronoi_cap","coverage_void","glob","manifold_slice","bridge"]);

// Vertex-shader transform shared by the points material (and, later, the
// id-buffer pick material): the GPU equivalent of `curPos(i)`. Declares the
// per-point transform attributes + uniforms and defines sphTransform(origPos) →
// displayed position — spread/radial, or the morph slerp when uHasMorph/uMorphT
// are set. KEEP THIS IN LOCKSTEP WITH curPos(): they must compute the same
// position, or CPU features (selection/minimap/ruler) drift from what's drawn.
const VERTEX_TRANSFORM=`
attribute vec3 aCatDir;attribute vec3 aMorphDir;attribute float aMorphR;attribute float aMorphHas;
uniform float uSpread;uniform float uRadial;uniform float uSR;uniform float uMorphT;uniform float uHasMorph;
vec3 sphTransform(vec3 o){
  float mag=length(o);
  if(mag<1e-9) return o;
  vec3 d=o/mag;
  if(uHasMorph>0.5 && uMorphT>0.0){
    if(aMorphHas<0.5) return o;
    vec3 bd=aMorphDir;
    float om=acos(clamp(dot(d,bd),-1.0,1.0));
    vec3 n;
    if(om<1e-5){ n=d; }
    else if(om>3.141592653589793-1e-4){
      vec3 h=abs(d.x)<0.9?vec3(1.0,0.0,0.0):vec3(0.0,1.0,0.0);
      float hd=dot(h,d);vec3 p=normalize(h-hd*d);
      float th=uMorphT*om;n=d*cos(th)+p*sin(th);
    } else {
      float s=sin(om);float w1=sin((1.0-uMorphT)*om)/s;float w2=sin(uMorphT*om)/s;n=d*w1+bd*w2;
    }
    return n*(mag+(aMorphR-mag)*uMorphT);
  }
  if(uSpread!=1.0){
    float dt=clamp(dot(aCatDir,d),-1.0,1.0);float om=acos(dt);
    if(om>=1e-4){float s=sin(om);float w1=sin((1.0-uSpread)*om)/s;float w2=sin(uSpread*om)/s;d=normalize(aCatDir*w1+d*w2);}
  }
  return d*max(0.02,uSR+(mag-uSR)*uRadial);
}
`;

// ── DOM refs (persistent) ────────────────────────────────────────────────
const canvas=document.getElementById("c");
const tooltip=document.getElementById("tooltip"),reticle=document.getElementById("reticle"),sellabel=document.getElementById("sellabel");
const legendDiv=document.getElementById("legend-items");
const oi=document.getElementById("overlay-items");
const statsDiv=document.getElementById("stats-content");
const labelsDiv=document.getElementById("labels");
const labelTogglesDiv=document.getElementById("label-toggles");
const searchInput=document.getElementById("search-input");
const spread=document.getElementById("spread"),spreadVal=document.getElementById("spread-val");
const radial=document.getElementById("radial"),radialVal=document.getElementById("radial-val");
const scaleS=document.getElementById("scale"),scaleVal=document.getElementById("scale-val");
const zoomS=document.getElementById("zoom"),zoomVal=document.getElementById("zoom-val");
const psize=document.getElementById("psize"),sizeVal=document.getElementById("size-val");
const globeCb=document.getElementById("globe");
const autorot=document.getElementById("autorot");
const uiS=document.getElementById("ui"),uiVal=document.getElementById("ui-val");
const schemeSel=document.getElementById("scheme");
const densityCb=document.getElementById("density");
const cfgFile=document.getElementById("cfg-file");
const sceneFile=document.getElementById("scene-file");
const dropzone=document.getElementById("dropzone");
const pinsDiv=document.getElementById("pins");
const mini=document.getElementById("mini"),mctx=mini.getContext("2d");
let MW=mini.width,MH=mini.height; // minimap buffer size — tracks its (resizable) CSS size
const miniBase=document.createElement("canvas");miniBase.width=MW;miniBase.height=MH;const mbctx=miniBase.getContext("2d");
// Drag-to-resize the minimap (CSS `resize:both` on #mini): keep the drawing
// buffer matched to the element's size so the sky chart stays crisp.
if(typeof ResizeObserver!=="undefined"){
  new ResizeObserver(()=>{
    const w=Math.max(60,Math.round(mini.clientWidth)),h=Math.max(30,Math.round(mini.clientHeight));
    if(w===MW&&h===MH)return;
    MW=w;MH=h;mini.width=MW;mini.height=MH;miniBase.width=MW;miniBase.height=MH;
    drawMinimapBase();
  }).observe(mini);
}

// ── THREE setup (persistent) ─────────────────────────────────────────────
// preserveDrawingBuffer keeps the last frame readable so PNG export
// (canvas.toDataURL) is reliable across browsers without an off-screen
// render-target; the modest VRAM cost is acceptable for an explorer UI.
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x06060f);
const camera=new THREE.PerspectiveCamera(54,innerWidth/innerHeight,0.01,200000);
const controls=new THREE.OrbitControls(camera,canvas);
controls.enableDamping=true;controls.dampingFactor=0.07;
controls.autoRotate=false;controls.autoRotateSpeed=0.35;controls.zoomSpeed=DEF.zoom;
scene.add(new THREE.AmbientLight(0x44557a,2.1));
const dl=new THREE.DirectionalLight(0xcfe2ff,0.55);dl.position.set(3,5,4);scene.add(dl);
const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=0.045;
const mouse=new THREE.Vector2();
// Offscreen 1×1 target for GPU id-buffer picking: render only the points (each
// carrying its index baked into color) to the cursor pixel and read it back —
// O(1) picking at any N vs the raycaster's O(N) intersect. Null (→ CPU fallback)
// when render targets aren't available (e.g. the headless test stub).
const pickRT=typeof THREE.WebGLRenderTarget==="function"?new THREE.WebGLRenderTarget(1,1):null;
// Persistent tool layer: the great-circle ruler draws here. Scaled with the
// scene (added to `scalables` in rebuild); its content is scene-scoped and
// cleared on teardown.
const rulerGroup=new THREE.Group();scene.add(rulerGroup);
const rulerReadout=document.getElementById("ruler-readout");
// Persistent query layer: semantic-query neighbor geodesics draw here (scaled
// with the scene; content is scene-scoped, cleared on teardown / new query).
const queryGroup=new THREE.Group();scene.add(queryGroup);
// Persistent pin layer: user-dropped (θ,φ) annotation markers on the shell.
const pinGroup=new THREE.Group();scene.add(pinGroup);
const _pv=new THREE.Vector3(),_fwd=new THREE.Vector3(),_tmp=new THREE.Vector3();
const _zc=new THREE.Vector3(),_zd=new THREE.Vector3(),_zn=new THREE.Vector3();
const _av=new THREE.Vector3(),_cd=new THREE.Vector3();

// ── Per-scene state (reassigned by rebuild) ──────────────────────────────
let pts=[],N=0,overlays=[],SR=1.0,maxR=1,showAxes=false;
let catSet=[],catColor={},catVisible={},catCounts={},catDir={},catDirArr=[],posIndex=new Map();
let idToIndex=new Map(); // stable id → point index, for semantic-query highlight
let morphTarget=null,morphT=0; // id → {d:unit dir, r} of a second scene, + slider t
let origPos=new Float32Array(0);
let pointsGeo=null,pointsMat=null,pickMat=null,pointsMesh=null,globeGroup=null,linesGroup=null;
let overlayGroups={},overlayKinds=new Set(),bridgeLines=[],bridgesByPoint={};
let labelData=[],labelEls=[],labelKindOn={},labelKindsPresent=[],soloCat=null,labelRefDist=1;
let legendRows={};
let scalables=[];
let baseSize=DEF.size,curScale=DEF.scale,spreadF=DEF.spread,radialG=DEF.radial,uiScale=DEF.ui;
let selectedIdx=-1,hoveredIdx=-1;
let tgtTween=null,pendingTransform=false;
let rulerOn=false,rulerPicks=[],rulerLast=null;
let zoomLocked=false; // compare-mode wheel-zoom lock (set via #embed sphereql-lock)
let pins=[],pinEls=[],pinOn=false; // (θ,φ) annotation markers + their DOM labels

// ── Module functions (operate on the current scene state) ─────────────────
function buildCatColor(name){const pal=PALETTES[name]||PALETTES.aurora;catSet.forEach((c,i)=>catColor[c]=pal[i%pal.length]);}
// Smooth orbit-target tween — recenters the view on a selected sphere and
// reverts to origin on deselect.
function frameCamera(){const d=DEF.scale*maxR*2.6;camera.position.set(d*0.12,d*0.3,d);controls.target.set(0,0,0);controls.update();}
function tweenTarget(to){tgtTween={from:controls.target.clone(),to:to.clone(),t:0,dur:reduceMotion?1:22};}
// Apply the same angular(domain)+radial transform to an arbitrary position,
// using the nearest domain centroid as the spread pivot. Used to keep bridge
// endpoints attached as the point cloud spreads.
function transformPos(p){
  const x=p[0],y=p[1],z=p[2],mag=Math.hypot(x,y,z);
  if(mag<1e-9)return [x,y,z];
  let dx=x/mag,dy=y/mag,dz=z/mag;
  if(spreadF!==1){
    let bc=null,bd=-2;for(const c of catDirArr){const dt=c[0]*dx+c[1]*dy+c[2]*dz;if(dt>bd){bd=dt;bc=c;}}
    const om=Math.acos(clamp(bd,-1,1));
    if(bc&&om>=1e-4){const s=Math.sin(om),w1=Math.sin((1-spreadF)*om)/s,w2=Math.sin(spreadF*om)/s;
      const nx=bc[0]*w1+dx*w2,ny=bc[1]*w1+dy*w2,nz=bc[2]*w1+dz*w2,nm=Math.hypot(nx,ny,nz)||1;dx=nx/nm;dy=ny/nm;dz=nz/nm;}
  }
  const nmag=Math.max(0.02,SR+(mag-SR)*radialG);
  return [dx*nmag,dy*nmag,dz*nmag];
}
// Combined angular (domain spread) + radial + morph transform. The per-point
// math now runs in the vertex shader (see `sphTransform`); this just pushes the
// current parameters as uniforms and refreshes the CPU-side bits that read
// displayed positions — bridge endpoints, the minimap, and any active
// selection. O(bridges) instead of the old O(N) buffer rewrite per slider tick.
function applyTransform(){
  if(!pointsMat)return;
  const u=pointsMat.uniforms;
  u.uSpread.value=spreadF;u.uRadial.value=radialG;u.uSR.value=SR;
  u.uMorphT.value=morphTarget?morphT:0;u.uHasMorph.value=morphTarget?1:0;
  for(const b of bridgeLines){const a=b.fromIndex>=0?curPos(b.fromIndex):transformPos(b.from),c=transformPos(b.to),pos=b.line.geometry.getAttribute("position");pos.setXYZ(0,a[0],a[1],a[2]);pos.setXYZ(1,c[0],c[1],c[2]);pos.needsUpdate=true;}
  if(selectedIdx>=0)deselectPoint();drawMinimapBase();
}
function applySize(sz){baseSize=sz;const sa=pointsGeo.getAttribute("size").array;for(let i=0;i<N;i++)sa[i]=catVisible[pts[i].cat]?sz:0;pointsGeo.getAttribute("size").needsUpdate=true;if(selectedIdx>=0)deselectPoint();}

function v3(a){return new THREE.Vector3(a[0],a[1],a[2]);}
function capRing(dir,ha,col){const d=v3(dir).normalize(),rr=SR*Math.sin(ha),off=SR*Math.cos(ha);
  const ring=new THREE.Mesh(new THREE.RingGeometry(rr*0.99,rr,64),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.45,side:THREE.DoubleSide}));
  ring.position.copy(d.clone().multiplyScalar(off));ring.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),d);return ring;}
function lineBetween(a,b,col,op){return new THREE.Line(new THREE.BufferGeometry().setFromPoints([v3(a),v3(b)]),new THREE.LineBasicMaterial({color:col,transparent:true,opacity:op}));}
function marker(pos,col,rad){const m=new THREE.Mesh(new THREE.SphereGeometry(rad,14,14),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.6,roughness:0.35}));m.position.copy(v3(pos));return m;}
function groupFor(k){if(!overlayGroups[k]){const g=new THREE.Group();overlayGroups[k]=g;scene.add(g);}return overlayGroups[k];}

// Uniform world scale applied to each object's own transform (keeps point
// picking correct — THREE scales the Points pick threshold by object scale).
function applyScale(s){curScale=s;scalables.forEach(o=>o.scale.setScalar(s));raycaster.params.Points.threshold=maxR*0.07*s;}

function projectToScreen(p){
  _pv.set(p[0]*curScale,p[1]*curScale,p[2]*curScale);
  camera.getWorldDirection(_fwd);
  const inFront=_tmp.copy(_pv).sub(camera.position).dot(_fwd)>0; // robust behind-camera reject
  _pv.project(camera);
  return{x:(_pv.x*0.5+0.5)*innerWidth,y:(-_pv.y*0.5+0.5)*innerHeight,vis:inFront&&_pv.z<=1};
}
// Zoom to cursor: keep the world point under the pointer fixed on screen while
// dollying. OrbitControls r128 has no zoomToCursor, so we intercept the wheel
// in the capture phase and stop it reaching OrbitControls; touch pinch still
// uses OrbitControls' built-in zoom.
function worldUnderCursor(mx,my){
  _zc.set((mx/innerWidth)*2-1,-(my/innerHeight)*2+1,0.5).unproject(camera);
  _zd.copy(_zc).sub(camera.position).normalize();
  camera.getWorldDirection(_zn);
  const denom=_zd.dot(_zn);
  if(Math.abs(denom)<1e-6)return controls.target.clone();
  const tt=(_zn.dot(controls.target)-_zn.dot(camera.position))/denom;
  return camera.position.clone().add(_zd.multiplyScalar(tt));
}
// Bake point index i into an RGB color (id = i+1, so 0 = background/no point)
// and recover it from the read-back bytes. 24-bit id space (≤16M points).
function pickEncode(i){const id=i+1;return[(id&255)/255,((id>>8)&255)/255,((id>>16)&255)/255];}
function pickDecode(r,g,b){return((r&255)|((g&255)<<8)|((b&255)<<16))-1;} // byte args; -1 = background
// GPU id-buffer pick: render ONLY the points (with pickMat baking each id into
// color) to the 1px target focused on the cursor, read it back, decode. Returns
// the point index, -1 for empty, or -2 when the pick path is unavailable (so
// the caller falls back to the CPU pick). Restores all render state in finally.
function pickGPU(e){
  if(!pickRT||!pickMat||!pointsMesh||!renderer.readRenderTargetPixels)return -2;
  const prev=renderer.getRenderTarget?renderer.getRenderTarget():null,vis=[];
  // Capture the clear color/alpha too — we set them for the pick pass and must
  // put them back (the main loop relies on scene.background today, but a
  // transparent embed would clear to black/transparent if we leaked this).
  const pcol=renderer.getClearColor?renderer.getClearColor(new THREE.Color()):null,palpha=renderer.getClearAlpha?renderer.getClearAlpha():1;
  scene.children.forEach(c=>{vis.push(c.visible);if(c!==pointsMesh)c.visible=false;});
  try{
    pointsMesh.material=pickMat;
    camera.setViewOffset(innerWidth,innerHeight,e.clientX,e.clientY,1,1);
    renderer.setRenderTarget(pickRT);
    if(renderer.setClearColor)renderer.setClearColor(0x000000,0);
    renderer.clear();renderer.render(scene,camera);
    const buf=new Uint8Array(4);renderer.readRenderTargetPixels(pickRT,0,0,1,1,buf);
    return pickDecode(buf[0],buf[1],buf[2]);
  }finally{
    camera.clearViewOffset();renderer.setRenderTarget(prev);
    if(pcol&&renderer.setClearColor)renderer.setClearColor(pcol,palpha);
    pointsMesh.material=pointsMat;scene.children.forEach((c,i)=>{c.visible=vis[i];});
  }
}
// CPU fallback: project every visible point through the SAME transform the GPU
// uses (curPos) and pick the nearest to the cursor within a small radius. O(N),
// but transform-correct (the raycaster would intersect untransformed origPos).
function pickCPU(e){
  let best=-1,bestD=14*14;
  for(let i=0;i<N;i++){if(!catVisible[pts[i].cat])continue;const sp=projectToScreen(curPos(i));if(!sp.vis)continue;
    const dx=sp.x-e.clientX,dy=sp.y-e.clientY,d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=i;}}
  return best;
}
// Streaming-mode pick: project every loaded tile point and return the GLOBAL
// row nearest the cursor (within a small radius), stashing its position for the
// reticle. O(loaded points), coalesced to one pick/frame by updateHover.
function pickStreamCPU(e){
  _streamHoverPos=null;
  if(!streamGroup)return -1;
  let best=-1,bestD=14*14;
  for(const mesh of streamGroup.children){
    const pos=mesh.geometry&&mesh.geometry.getAttribute&&mesh.geometry.getAttribute("position"),rows=mesh.userData&&mesh.userData.rows;
    if(!pos||!rows)continue;const a=pos.array;
    for(let i=0;i<rows.length;i++){const x=a[i*3],y=a[i*3+1],z=a[i*3+2],sp=projectToScreen([x,y,z]);if(!sp.vis)continue;
      const dx=sp.x-e.clientX,dy=sp.y-e.clientY,d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=rows[i];_streamHoverPos=[x,y,z];}}
  }
  return best;
}
function getHovered(e){
  if(streamGroup&&streamStreamer)return pickStreamCPU(e); // streaming → global row
  if(!pointsMesh)return -1;
  let idx=-2;
  try{idx=pickGPU(e);}catch(err){idx=-2;}
  if(idx===-2)return pickCPU(e); // GPU path unavailable/failed
  if(idx>=0&&idx<N&&catVisible[pts[idx].cat])return idx;
  return -1;
}
// Current (displayed) position of point i, computed from its ORIGINAL position
// by the same spread/radial (or morph) transform the vertex shader applies on
// the GPU. This is the single CPU-side source of truth for displayed positions
// (selection, minimap, ruler, query/bridge geodesics) now that the per-frame
// O(N) position-buffer rewrite is gone — the GLSL `sphTransform` in the points
// shader is a line-for-line transcription of this, so CPU features agree with
// what's drawn. Morph and spread/radial are mutually exclusive (matching the
// old applyTransform/applyMorph): while morphing, matched points morph and
// unmatched points hold at the original position.
function curPos(i){
  const ox=origPos[i*3],oy=origPos[i*3+1],oz=origPos[i*3+2],mag=Math.hypot(ox,oy,oz);
  if(mag<1e-9)return[ox,oy,oz];
  let dx=ox/mag,dy=oy/mag,dz=oz/mag;
  if(morphTarget&&morphT>0){
    const tgt=pts[i].id!=null?morphTarget.get(String(pts[i].id)):undefined;
    if(!tgt)return[ox,oy,oz];
    const bd=tgt.d,om=Math.acos(clamp(dx*bd[0]+dy*bd[1]+dz*bd[2],-1,1));
    let nx,ny,nz;
    if(om<1e-5){nx=dx;ny=dy;nz=dz;}
    else if(om>Math.PI-1e-4){const h=Math.abs(dx)<0.9?[1,0,0]:[0,1,0];const hd=h[0]*dx+h[1]*dy+h[2]*dz;let px=h[0]-hd*dx,py=h[1]-hd*dy,pz=h[2]-hd*dz;const pm=Math.hypot(px,py,pz)||1;px/=pm;py/=pm;pz/=pm;const th=morphT*om,c2=Math.cos(th),s2=Math.sin(th);nx=dx*c2+px*s2;ny=dy*c2+py*s2;nz=dz*c2+pz*s2;}
    else{const s=Math.sin(om),w1=Math.sin((1-morphT)*om)/s,w2=Math.sin(morphT*om)/s;nx=dx*w1+bd[0]*w2;ny=dy*w1+bd[1]*w2;nz=dz*w1+bd[2]*w2;}
    const rr=mag+(tgt.r-mag)*morphT;
    return[nx*rr,ny*rr,nz*rr];
  }
  if(spreadF!==1){const c=catDir[pts[i].cat];const dot=clamp(c[0]*dx+c[1]*dy+c[2]*dz,-1,1),om=Math.acos(dot);
    if(om>=1e-4){const s=Math.sin(om),w1=Math.sin((1-spreadF)*om)/s,w2=Math.sin(spreadF*om)/s;
      const nx=c[0]*w1+dx*w2,ny=c[1]*w1+dy*w2,nz=c[2]*w1+dz*w2,nm=Math.hypot(nx,ny,nz)||1;dx=nx/nm;dy=ny/nm;dz=nz/nm;}}
  const nmag=Math.max(0.02,SR+(mag-SR)*radialG);
  return[dx*nmag,dy*nmag,dz*nmag];
}

function selectPoint(idx){
  selectedIdx=idx;hoveredIdx=-1;const P=curPos(idx);
  tweenTarget(new THREE.Vector3(P[0]*curScale,P[1]*curScale,P[2]*curScale)); // auto-center on this sphere
  const dists=pts.map((q,i)=>{if(i===idx||!catVisible[q.cat])return{i,d:Infinity};const c=curPos(i),dx=P[0]-c[0],dy=P[1]-c[1],dz=P[2]-c[2];return{i,d:Math.sqrt(dx*dx+dy*dy+dz*dz)};}).filter(d=>d.d<Infinity).sort((a,b)=>a.d-b.d).slice(0,5);
  const near=new Set([idx,...dists.map(d=>d.i)]);
  const sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
  for(let i=0;i<N;i++){const base=new THREE.Color(catColor[pts[i].cat]);
    if(near.has(i)){sa[i]=i===idx?baseSize*1.7:baseSize*1.4;ca[i*3]=base.r;ca[i*3+1]=base.g;ca[i*3+2]=base.b;}
    else{sa[i]=baseSize*0.5;ca[i*3]=base.r*0.28;ca[i*3+1]=base.g*0.28;ca[i*3+2]=base.b*0.28;}}
  pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;pointsMat.uniforms.opacity.value=0.4;
  while(linesGroup.children.length)linesGroup.remove(linesGroup.children[0]);
  const lm=new THREE.LineBasicMaterial({color:0x5cc8ff,transparent:true,opacity:0.5});
  for(const d of dists){const c=curPos(d.i);linesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(P[0],P[1],P[2]),new THREE.Vector3(c[0],c[1],c[2])]),lm));}
  // This sphere's bridges (item → target domain), drawn from the sphere itself
  // so highlighting always reveals its bridges regardless of the global toggle.
  const myBridges=bridgesByPoint[idx];
  if(myBridges)for(const br of myBridges){const c=transformPos(br.to);
    linesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(P[0],P[1],P[2]),new THREE.Vector3(c[0],c[1],c[2])]),new THREE.LineBasicMaterial({color:br.color,transparent:true,opacity:0.9})));}
  const p=pts[idx],info=document.getElementById("info");
  sellabel.innerHTML=`<span class="sl-dot" style="background:${catColor[p.cat]};color:${catColor[p.cat]}"></span>${escHtml(p.label||"Point "+idx)}`;
  document.getElementById("info-label").textContent=p.label||"Point "+idx;
  const tag=document.getElementById("info-cat");tag.textContent=p.cat;tag.style.color=catColor[p.cat];tag.style.background=catColor[p.cat]+"18";
  document.getElementById("info-coords").innerHTML=`<span>θ</span><b>${p.theta.toFixed(4)}</b><span>φ</span><b>${p.phi.toFixed(4)}</b><span>r</span><b>${p.r.toFixed(4)}</b>${myBridges?`<span>bridges</span><b>${myBridges.length}</b>`:""}`;
  const nb=document.getElementById("info-neighbors");
  nb.innerHTML=dists.map(d=>{const dc=catColor[pts[d.i].cat];return `<div class="nb" data-idx="${d.i}" style="background:${dc}22;border-left:2px solid ${dc}"><span>${escHtml(pts[d.i].label||"Point "+d.i)}</span><span class="dist">${d.d.toFixed(3)}</span></div>`;}).join("");
  nb.querySelectorAll(".nb").forEach(el=>el.addEventListener("click",()=>selectPoint(parseInt(el.dataset.idx))));
  info.classList.add("visible");
}
function deselectPoint(revertCam){
  selectedIdx=-1;sellabel.style.display="none";
  const sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
  for(let i=0;i<N;i++){sa[i]=catVisible[pts[i].cat]?baseSize:0;const c=new THREE.Color(catColor[pts[i].cat]);ca[i*3]=c.r;ca[i*3+1]=c.g;ca[i*3+2]=c.b;}
  pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;pointsMat.uniforms.opacity.value=1.0;
  while(linesGroup.children.length)linesGroup.remove(linesGroup.children[0]);
  document.getElementById("info").classList.remove("visible");
  if(revertCam)tweenTarget(new THREE.Vector3(0,0,0)); // return to the main sphere
}

function setAll(v){catSet.forEach(c=>{catVisible[c]=v;legendRows[c].classList.toggle("dim",!v);});updateVisibility();}
function updateVisibility(){const sa=pointsGeo.getAttribute("size").array;for(let i=0;i<N;i++)sa[i]=catVisible[pts[i].cat]?baseSize:0;pointsGeo.getAttribute("size").needsUpdate=true;if(selectedIdx>=0)deselectPoint();drawMinimapBase();}

function applyPalette(name){
  buildCatColor(name);
  const ca=pointsGeo.getAttribute("color").array;
  for(let i=0;i<N;i++){const c=new THREE.Color(catColor[pts[i].cat]);ca[i*3]=c.r;ca[i*3+1]=c.g;ca[i*3+2]=c.b;}
  pointsGeo.getAttribute("color").needsUpdate=true;
  catSet.forEach(c=>{const dot=legendRows[c]&&legendRows[c].querySelector(".ldot");if(dot){dot.style.background=catColor[c];dot.style.color=catColor[c];}});
  pointsMat.uniforms.opacity.value=1.0;drawMinimapBase();
  if(selectedIdx>=0)selectPoint(selectedIdx); // re-apply highlight in place (no info-panel flicker)
}

// ── Config (TOML) save / load ────────────────────────────────────────────
function currentSettings(){return{scale:curScale,zoom_speed:controls.zoomSpeed,radial:radialG,domain_spread:spreadF,point_size:baseSize,ui_scale:uiScale,color_scheme:schemeSel.value,reference_globe:globeCb.checked,auto_rotate:autorot.checked,density:densityCb.checked,pins:btoa(encodeURIComponent(JSON.stringify(pins)))};}
function toToml(o){return"# SphereQL view settings\n"+Object.entries(o).map(([k,v])=>{const val=typeof v==="string"?`"${v}"`:(typeof v==="boolean"?(v?"true":"false"):v);return `${k} = ${val}`;}).join("\n")+"\n";}
function parseToml(text){const o={};text.split(/\r?\n/).forEach(line=>{line=line.trim();if(!line||line[0]==="#")return;const i=line.indexOf("=");if(i<0)return;const k=line.slice(0,i).trim();let v=line.slice(i+1).trim();if(/^".*"$/.test(v))v=v.slice(1,-1);else if(v==="true")v=true;else if(v==="false")v=false;else{const n=parseFloat(v);if(!isNaN(n))v=n;}o[k]=v;});return o;}
function applySettings(o){
  if("scale"in o){const s=+o.scale;scaleS.value=s;scaleVal.textContent=s.toFixed(1)+"×";applyScale(s);}
  if("zoom_speed"in o){const v=+o.zoom_speed;zoomS.value=v;zoomVal.textContent=v.toFixed(2)+"×";controls.zoomSpeed=v;}
  if("radial"in o){radialG=+o.radial;radial.value=radialG;radialVal.textContent=radialG.toFixed(1)+"×";}
  if("domain_spread"in o){spreadF=+o.domain_spread;spread.value=spreadF;spreadVal.textContent=spreadF.toFixed(1)+"×";}
  if("point_size"in o){const v=+o.point_size;psize.value=v;sizeVal.textContent=v.toFixed(1);applySize(v);}
  if("ui_scale"in o){uiScale=+o.ui_scale;uiS.value=uiScale;uiVal.textContent=uiScale.toFixed(2)+"×";document.documentElement.style.setProperty("--ui",uiScale);}
  if("color_scheme"in o&&PALETTES[o.color_scheme]){schemeSel.value=o.color_scheme;applyPalette(o.color_scheme);}
  if("reference_globe"in o){globeCb.checked=!!o.reference_globe;globeGroup.visible=!!o.reference_globe;}
  if("auto_rotate"in o){autorot.checked=!!o.auto_rotate;controls.autoRotate=!!o.auto_rotate;}
  if("density"in o){densityCb.checked=!!o.density;if(pointsMat)pointsMat.uniforms.densityOn.value=o.density?1:0;}
  if("pins"in o){try{const arr=JSON.parse(decodeURIComponent(atob(o.pins)));if(Array.isArray(arr)){pins=arr.filter(p=>p&&isFinite(+p.theta)&&isFinite(+p.phi)).map(p=>({theta:+p.theta,phi:+p.phi,label:String(p.label||"")}));renderPins();}}catch(err){console.warn("SphereQL: ignoring bad pins in settings");}}
  applyTransform();
}
function resetDefaults(){
  scaleS.value=DEF.scale;scaleVal.textContent=DEF.scale.toFixed(1)+"×";applyScale(DEF.scale);
  zoomS.value=DEF.zoom;zoomVal.textContent=DEF.zoom.toFixed(2)+"×";controls.zoomSpeed=DEF.zoom;
  radial.value=DEF.radial;radialVal.textContent=DEF.radial.toFixed(1)+"×";radialG=DEF.radial;
  spread.value=DEF.spread;spreadVal.textContent=DEF.spread.toFixed(1)+"×";spreadF=DEF.spread;
  psize.value=DEF.size;sizeVal.textContent=DEF.size.toFixed(1);applySize(DEF.size);
  uiScale=DEF.ui;uiS.value=DEF.ui;uiVal.textContent=DEF.ui.toFixed(2)+"×";document.documentElement.style.setProperty("--ui",DEF.ui);
  schemeSel.value=DEF.palette;applyPalette(DEF.palette);
  globeCb.checked=DEF.globe;globeGroup.visible=DEF.globe;
  autorot.checked=DEF.autorot;controls.autoRotate=DEF.autorot;
  densityCb.checked=DEF.density;if(pointsMat)pointsMat.uniforms.densityOn.value=DEF.density?1:0;
  setPinMode(false);clearPins();
  labelKindsPresent.forEach(k => (labelKindOn[k] = true));
  labelTogglesDiv.querySelectorAll("input").forEach(cb => {
    cb.checked = true;
  });
  soloCat = null;
  setAll(true);
  searchInput.value="";applyTransform();frameCamera();
}

// ── Minimap (equirectangular θ→φ sky chart) ──────────────────────────────
function tpOf(x,y,z){const r=Math.hypot(x,y,z)||1;let th=Math.atan2(y,x);if(th<0)th+=2*Math.PI;return[th,Math.acos(clamp(z/r,-1,1))];}
function drawMinimapBase(){
  mbctx.clearRect(0,0,MW,MH);mbctx.strokeStyle="rgba(120,160,255,0.10)";mbctx.lineWidth=1;
  for(let i=1;i<3;i++){const y=MH*i/3;mbctx.beginPath();mbctx.moveTo(0,y);mbctx.lineTo(MW,y);mbctx.stroke();}
  for(let i=1;i<6;i++){const x=MW*i/6;mbctx.beginPath();mbctx.moveTo(x,0);mbctx.lineTo(x,MH);mbctx.stroke();}
  if(!pointsGeo)return;
  for(let i=0;i<N;i++){if(!catVisible[pts[i].cat])continue;const c=curPos(i),[th,ph]=tpOf(c[0],c[1],c[2]);
    mbctx.fillStyle=catColor[pts[i].cat];mbctx.fillRect(th/(2*Math.PI)*MW-0.6,ph/Math.PI*MH-0.6,1.7,1.7);}}
function drawMinimap(){
  mctx.clearRect(0,0,MW,MH);mctx.drawImage(miniBase,0,0);
  const[th,ph]=tpOf(camera.position.x,camera.position.y,camera.position.z);
  const x=th/(2*Math.PI)*MW,y=ph/Math.PI*MH;
  mctx.strokeStyle="#ffb454";mctx.lineWidth=1.3;
  mctx.beginPath();mctx.arc(x,y,5,0,2*Math.PI);mctx.stroke();
  mctx.beginPath();mctx.moveTo(x-9,y);mctx.lineTo(x+9,y);mctx.moveTo(x,y-9);mctx.lineTo(x,y+9);mctx.stroke();}

// ── Floating overlay labels (centroids / antipodes / domain groups) ───────
// DOM labels anchored to the 3D markers: projected each frame, scaled by
// camera distance, hidden when behind the globe, and clickable (focus +
// solo-domain). Tied to the matching overlay-group visibility.
function focusLabel(ld){
  if(ld.kind==="centroid"){
    if(soloCat===ld.cat){soloCat=null;setAll(true);}
    else{soloCat=ld.cat;catSet.forEach(c=>{catVisible[c]=c===ld.cat;if(legendRows[c])legendRows[c].classList.toggle("dim",c!==ld.cat);});updateVisibility();}
  }
  tweenTarget(new THREE.Vector3(ld.anchor[0]*curScale,ld.anchor[1]*curScale,ld.anchor[2]*curScale));
}
function updateLabels(){
  _cd.copy(camera.position).normalize();
  for(const {el,ld} of labelEls){
    const grp=overlayGroups[ld.kind];
    if(!labelKindOn[ld.kind]||!grp||!grp.visible){el.style.display="none";continue;}
    _av.set(ld.anchor[0]*curScale,ld.anchor[1]*curScale,ld.anchor[2]*curScale);
    const al=Math.hypot(_av.x,_av.y,_av.z)||1;
    const facing=(_av.x*_cd.x+_av.y*_cd.y+_av.z*_cd.z)/al>-0.15; // front hemisphere
    const sp=projectToScreen(ld.anchor);
    if(sp.vis&&facing){
      const dist=camera.position.distanceTo(_av),s=clamp(labelRefDist/Math.max(dist,1e-3),0.7,1.7);
      el.style.display="flex";el.style.left=sp.x+"px";el.style.top=sp.y+"px";el.style.fontSize=(10.5*s).toFixed(1)+"px";
      el.classList.toggle("solo",ld.kind==="centroid"&&soloCat===ld.cat);
    }else el.style.display="none";
  }
}

// Dispose every geometry/material under an object (frees GPU memory on swap).
function disposeObject(o){if(!o)return;o.traverse(c=>{if(c.geometry)c.geometry.dispose();if(c.material){const m=c.material;(Array.isArray(m)?m:[m]).forEach(x=>{if(x&&x.dispose)x.dispose();});}});}

// ── Static event bindings (attached once; act on the current scene) ───────
canvas.addEventListener("wheel",e=>{
  e.preventDefault();e.stopImmediatePropagation();
  if(zoomLocked)return; // compare-mode zoom lock
  const f=worldUnderCursor(e.clientX,e.clientY);
  const s=Math.exp(Math.sign(e.deltaY)*Math.min(Math.abs(e.deltaY),120)/120*controls.zoomSpeed*0.2);
  camera.position.sub(f).multiplyScalar(s).add(f);
  controls.target.sub(f).multiplyScalar(s).add(f);
  const d=camera.position.distanceTo(controls.target),cd=clamp(d,controls.minDistance,controls.maxDistance);
  if(d!==cd)camera.position.copy(controls.target).addScaledVector(_tmp.copy(camera.position).sub(controls.target).normalize(),cd);
  controls.update();
},{capture:true,passive:false});
// Hover picking is coalesced to one pick per frame (run by animate via
// updateHover): a synchronous GPU readback per raw mousemove could fire several
// times a frame on a high-frequency mouse. We just stash the latest event.
let _hoverEv=null;
canvas.addEventListener("mousemove",e=>{_hoverEv=e;});
canvas.addEventListener("mouseleave",()=>{_hoverEv=null;hoveredIdx=-1;tooltip.style.display="none";});
// One hover pick for the latest pointer position → tooltip + cursor. Called
// once per frame from animate(), so at most one GPU readback per frame.
function updateHover(){
  if(!_hoverEv)return;const e=_hoverEv;_hoverEv=null;
  const idx=getHovered(e);hoveredIdx=idx;
  if(streamStreamer){ // streaming: idx is a global row; pts[] is empty, so no
    // per-hover metadata fetch — a minimal tooltip, full detail on click.
    if(idx>=0){tooltip.innerHTML=`<div class="tt-lbl">point #${idx}</div><div class="tt-meta">click to inspect</div>`;tooltip.style.display="block";tooltip.style.left=(e.clientX+16)+"px";tooltip.style.top=(e.clientY+14)+"px";canvas.style.cursor="crosshair";}
    else{tooltip.style.display="none";canvas.style.cursor="grab";}
    return;
  }
  if(idx>=0){const p=pts[idx];
    tooltip.innerHTML=`<div class="tt-lbl">${escHtml(p.label||"Point "+idx)}</div><div class="tt-meta">${escHtml(p.cat)} · θ ${p.theta.toFixed(2)}  φ ${p.phi.toFixed(2)}  r ${p.r.toFixed(2)}</div>`;
    tooltip.style.display="block";tooltip.style.left=(e.clientX+16)+"px";tooltip.style.top=(e.clientY+14)+"px";canvas.style.cursor="crosshair";
  }else{tooltip.style.display="none";canvas.style.cursor="grab";}
}
window.addEventListener("keydown",e=>{if(e.key!=="Escape")return;if(pinOn){setPinMode(false);return;}if(rulerOn&&rulerPicks.length){clearRuler();return;}if(selectedIdx>=0)deselectPoint(true);});
// Click detection via pointer down/up + movement threshold — the `click`
// event is unreliable while OrbitControls is handling pointer gestures.
let _downX=0,_downY=0;
canvas.addEventListener("pointerdown",e=>{_downX=e.clientX;_downY=e.clientY;});
canvas.addEventListener("pointerup",e=>{
  if(Math.hypot(e.clientX-_downX,e.clientY-_downY)>=5)return; // a drag, not a click
  if(pinOn){ // pin mode: drop a marker where the ray meets the globe shell
    mouse.x=(e.clientX/innerWidth)*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;
    raycaster.setFromCamera(mouse,camera);
    const hit=globeGroup&&raycaster.intersectObject(globeGroup,true)[0];
    if(hit){const p=hit.point,m=Math.hypot(p.x,p.y,p.z)||1;let th=Math.atan2(p.y,p.x);if(th<0)th+=2*Math.PI;addPin(th,Math.acos(clamp(p.z/m,-1,1)));}
    return;
  }
  const idx=getHovered(e);
  if(streamStreamer){ // streaming: idx is a global row; inspect via the server
    if(rulerOn){if(_streamHoverPos)rulerAddPick(_streamHoverPos);return;}
    if(idx>=0)selectStreamRow(idx);
    else{const info=document.getElementById("info");if(info)info.classList.remove("visible");}
    return;
  }
  if(rulerOn){if(idx>=0)rulerAddPick(curPos(idx));return;} // ruler snaps to data points
  if(idx>=0)selectPoint(idx);else deselectPoint(true);
});

// Tabs + HUD toggle
document.querySelectorAll(".tab").forEach(t=>t.addEventListener("click",()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".tabpane").forEach(x=>x.classList.remove("active"));
  t.classList.add("active");document.getElementById("tab-"+t.dataset.tab).classList.add("active");}));
document.getElementById("hud-toggle").addEventListener("click",()=>document.body.classList.toggle("hud-hidden"));

// Legend select all / none
document.getElementById("sel-all").addEventListener("click",()=>setAll(true));
document.getElementById("sel-none").addEventListener("click",()=>setAll(false));

// Settings controls
spread.addEventListener("input",e=>{spreadF=parseFloat(e.target.value);spreadVal.textContent=spreadF.toFixed(1)+"×";pendingTransform=true;});
radial.addEventListener("input",e=>{radialG=parseFloat(e.target.value);radialVal.textContent=radialG.toFixed(1)+"×";pendingTransform=true;});
scaleS.addEventListener("input",e=>{const s=parseFloat(e.target.value);scaleVal.textContent=s.toFixed(1)+"×";applyScale(s);});
zoomS.addEventListener("input",e=>{const v=parseFloat(e.target.value);zoomVal.textContent=v.toFixed(2)+"×";controls.zoomSpeed=v;});
psize.addEventListener("input",e=>{const v=parseFloat(e.target.value);sizeVal.textContent=v.toFixed(1);applySize(v);});
globeCb.addEventListener("change",e=>{if(globeGroup)globeGroup.visible=e.target.checked;});
autorot.addEventListener("change",e=>{controls.autoRotate=e.target.checked;});
controls.addEventListener("start",()=>{controls.autoRotate=false;autorot.checked=false;});
uiS.addEventListener("input",e=>{uiScale=parseFloat(e.target.value);document.documentElement.style.setProperty("--ui",uiScale);uiVal.textContent=uiScale.toFixed(2)+"×";});
schemeSel.addEventListener("change",e=>applyPalette(e.target.value));

// Config save / load + reset
document.getElementById("save-cfg").addEventListener("click",()=>{
  const blob=new Blob([toToml(currentSettings())],{type:"text/plain"}),url=URL.createObjectURL(blob),a=document.createElement("a");
  a.href=url;a.download="sphereql-view.toml";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.getElementById("load-cfg").addEventListener("click",()=>cfgFile.click());
cfgFile.addEventListener("change",e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{
  try{applySettings(parseToml(r.result));}
  catch(err){console.warn("SphereQL: failed to load settings:",err);const b=document.getElementById("load-cfg"),t=b.textContent;b.textContent="✗ bad file";setTimeout(()=>{b.textContent=t;},1500);}
};r.readAsText(f);cfgFile.value="";});
document.getElementById("reset").addEventListener("click",resetDefaults);

// Open / drop a Scene JSON → rebuild the viewer around it.
document.getElementById("open-scene").addEventListener("click",()=>sceneFile.click());
sceneFile.addEventListener("change",e=>{loadSceneFromFile(e.target.files[0]);sceneFile.value="";});
const isFileDrag=e=>e.dataTransfer&&Array.from(e.dataTransfer.types||[]).indexOf("Files")>=0;
let _dragDepth=0;
window.addEventListener("dragenter",e=>{if(!isFileDrag(e))return;e.preventDefault();_dragDepth++;dropzone.classList.add("on");});
window.addEventListener("dragover",e=>{if(!isFileDrag(e))return;e.preventDefault();e.dataTransfer.dropEffect="copy";});
window.addEventListener("dragleave",e=>{if(!isFileDrag(e))return;_dragDepth=Math.max(0,_dragDepth-1);if(_dragDepth===0)dropzone.classList.remove("on");});
window.addEventListener("drop",e=>{e.preventDefault();_dragDepth=0;dropzone.classList.remove("on");const f=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0];if(f)loadSceneFromFile(f);});

// Tools: ruler / PNG snapshot / shareable link.
document.getElementById("tool-ruler").addEventListener("click",()=>setRuler(!rulerOn));
document.getElementById("tool-png").addEventListener("click",exportPNG);
document.getElementById("tool-share").addEventListener("click",shareLink);
document.getElementById("tool-pin").addEventListener("click",()=>setPinMode(!pinOn));
densityCb.addEventListener("change",e=>{if(pointsMat)pointsMat.uniforms.densityOn.value=e.target.checked?1:0;});

// Search
searchInput.addEventListener("input",()=>{
  if(!pointsGeo)return;
  const q=searchInput.value.toLowerCase(),sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
  if(!q){deselectPoint();return;}
  for(let i=0;i<N;i++){const match=(pts[i].label||"").toLowerCase().includes(q),vis=catVisible[pts[i].cat];
    sa[i]=(match&&vis)?baseSize*1.5:(vis?baseSize*0.45:0);const c=new THREE.Color(catColor[pts[i].cat]),f=match?1:0.25;
    ca[i*3]=c.r*f;ca[i*3+1]=c.g*f;ca[i*3+2]=c.b*f;}
  pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;pointsMat.uniforms.opacity.value=0.55;});

// Minimap click → aim camera at the (θ,φ) under the pointer
mini.addEventListener("click",e=>{
  const r=mini.getBoundingClientRect();
  const th=clamp((e.clientX-r.left)/r.width,0,1)*2*Math.PI,ph=clamp((e.clientY-r.top)/r.height,0,1)*Math.PI;
  const dir=new THREE.Vector3(Math.sin(ph)*Math.cos(th),Math.sin(ph)*Math.sin(th),Math.cos(ph));
  camera.position.copy(dir.multiplyScalar(camera.position.length()));controls.autoRotate=false;autorot.checked=false;controls.update();});

window.addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});

// ── Great-circle ruler ───────────────────────────────────────────────────
// Click two points; measure the angle between their directions (acos of the
// clamped dot of the unit vectors) and draw the connecting geodesic on the
// shell via slerp — the same arc construction as overlay.rs's geodesic_path.
function clearRuler(){rulerPicks=[];while(rulerGroup.children.length){const c=rulerGroup.children[0];disposeObject(c);rulerGroup.remove(c);}rulerReadout.classList.remove("on");}
function setRuler(on){
  rulerOn=on;document.getElementById("tool-ruler").classList.toggle("active",on);
  if(on){rulerReadout.querySelector(".rr-ang").textContent="—";rulerReadout.querySelector(".rr-sub").textContent="click two points · Esc to clear";rulerReadout.classList.add("on");}
  else clearRuler();
}
// Great-circle arc between two unit directions, sampled on the display shell
// (radius SR). Robust at the degenerate ends: coincident → a short segment;
// (near-)antipodal → a clean semicircle about a ⟂ axis (slerp's 1/sin(om)
// blows up there); else standard slerp. Shared by the ruler and query fans.
function shellArc(a,b){
  const om=Math.acos(clamp(a[0]*b[0]+a[1]*b[1]+a[2]*b[2],-1,1));
  const v=[],SEG=72;
  if(om<1e-4){
    v.push(new THREE.Vector3(a[0]*SR,a[1]*SR,a[2]*SR),new THREE.Vector3(b[0]*SR,b[1]*SR,b[2]*SR));
  }else if(om>Math.PI-1e-3){
    const h=Math.abs(a[0])<0.9?[1,0,0]:[0,1,0];
    const d=h[0]*a[0]+h[1]*a[1]+h[2]*a[2];
    let px=h[0]-d*a[0],py=h[1]-d*a[1],pz=h[2]-d*a[2];const pm=Math.hypot(px,py,pz)||1;px/=pm;py/=pm;pz/=pm;
    for(let i=0;i<=SEG;i++){const th=om*i/SEG,c=Math.cos(th),sn=Math.sin(th);
      v.push(new THREE.Vector3((a[0]*c+px*sn)*SR,(a[1]*c+py*sn)*SR,(a[2]*c+pz*sn)*SR));}
  }else{const s=Math.sin(om);for(let i=0;i<=SEG;i++){const t=i/SEG,w1=Math.sin((1-t)*om)/s,w2=Math.sin(t*om)/s;
    v.push(new THREE.Vector3((a[0]*w1+b[0]*w2)*SR,(a[1]*w1+b[1]*w2)*SR,(a[2]*w1+b[2]*w2)*SR));}}
  return v;
}
function rulerMeasure(){
  const a=rulerPicks[0],b=rulerPicks[1];
  const om=Math.acos(clamp(a[0]*b[0]+a[1]*b[1]+a[2]*b[2],-1,1));
  rulerGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(shellArc(a,b)),new THREE.LineBasicMaterial({color:0xffb454,transparent:true,opacity:0.95})));
  const deg=om*180/Math.PI;rulerLast={rad:om,deg:deg,chord:2*Math.sin(om/2)};
  rulerReadout.querySelector(".rr-ang").textContent=deg.toFixed(1)+"°  ·  "+om.toFixed(3)+" rad";
  rulerReadout.querySelector(".rr-sub").textContent="great-circle · chord "+(2*Math.sin(om/2)).toFixed(3);
  rulerReadout.classList.add("on");
}
function rulerAddPick(P){
  if(rulerPicks.length>=2)clearRuler();
  const m=Math.hypot(P[0],P[1],P[2])||1;
  rulerPicks.push([P[0]/m,P[1]/m,P[2]/m]);
  rulerGroup.add(marker(P,0xffb454,SR*0.02));
  if(rulerPicks.length===2)rulerMeasure();
  else{rulerReadout.querySelector(".rr-ang").textContent="•";rulerReadout.querySelector(".rr-sub").textContent="pick the second point";rulerReadout.classList.add("on");}
}

// ── PNG snapshot ─────────────────────────────────────────────────────────
function exportPNG(){
  try{renderer.render(scene,camera);const url=canvas.toDataURL("image/png");const a=document.createElement("a");a.href=url;a.download="sphereql-view.png";a.click();flashButton("tool-png","✓",1200);}
  catch(err){console.warn("SphereQL: PNG export failed:",err);flashButton("tool-png","✗");}
}

// ── Shareable view link (camera + settings only; never scene data) ────────
function shareLink(){
  const state={cam:[camera.position.x,camera.position.y,camera.position.z,controls.target.x,controls.target.y,controls.target.z],set:currentSettings(),tools:{ruler:rulerOn}};
  let hash;
  try{hash=btoa(encodeURIComponent(JSON.stringify(state)));}catch(err){flashButton("tool-share","✗");return;}
  try{history.replaceState(null,"","#v="+hash);}catch(err){location.hash="v="+hash;}
  if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(location.href).then(()=>flashButton("tool-share","✓ copied",1600),()=>flashButton("tool-share","✓ in URL",1600));
  else flashButton("tool-share","✓ in URL",1600);
}
// Restore a view from the URL hash (called once after the initial rebuild).
// Reads only numbers + known setting keys, all validated, so there is no
// injection surface even though the hash is attacker-controllable.
function applyViewHash(){
  if(typeof location==="undefined"||!location.hash)return;
  const m=location.hash.match(/[#&]v=([^&]+)/);if(!m)return;
  let state;
  try{state=JSON.parse(decodeURIComponent(atob(m[1])));}catch(err){console.warn("SphereQL: ignoring malformed view hash");return;}
  if(!state||typeof state!=="object")return;
  if(state.set&&typeof state.set==="object")applySettings(state.set);
  if(Array.isArray(state.cam)&&state.cam.length===6&&state.cam.every(v=>isFinite(+v))){
    camera.position.set(+state.cam[0],+state.cam[1],+state.cam[2]);
    controls.target.set(+state.cam[3],+state.cam[4],+state.cam[5]);
    controls.update();
  }
  if(state.tools&&state.tools.ruler)setRuler(true);
}

// ── Semantic query highlight (driven by the studio's pipeline.nearest) ────
// Emphasize a set of points by stable id (closest first), dim the rest, and
// fan geodesics from the nearest match to the others across the shell. Accepts
// an array of ids or of {id,...} objects (e.g. NearestOut). Returns the count
// of resolved matches. Pass an empty array to clear.
function clearQuery(){while(queryGroup.children.length){const c=queryGroup.children[0];disposeObject(c);queryGroup.remove(c);}}
function highlightByIds(ids){
  if(!pointsGeo)return 0;
  clearQuery();
  const order=[],seen=new Set();
  for(const raw of ids||[]){const key=String(raw&&raw.id!=null?raw.id:raw);const idx=idToIndex.get(key);if(idx!==undefined&&!seen.has(idx)){seen.add(idx);order.push(idx);}}
  if(order.length===0){deselectPoint();return 0;}
  const sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
  for(let i=0;i<N;i++){const c=new THREE.Color(catColor[pts[i].cat]);
    if(seen.has(i)){const top=i===order[0];sa[i]=baseSize*(top?1.8:1.4);ca[i*3]=c.r;ca[i*3+1]=c.g;ca[i*3+2]=c.b;}
    else{sa[i]=catVisible[pts[i].cat]?baseSize*0.4:0;ca[i*3]=c.r*0.22;ca[i*3+1]=c.g*0.22;ca[i*3+2]=c.b*0.22;}}
  pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;pointsMat.uniforms.opacity.value=0.6;
  const a=curPos(order[0]),am=Math.hypot(a[0],a[1],a[2])||1,ad=[a[0]/am,a[1]/am,a[2]/am];
  const mat=new THREE.LineBasicMaterial({color:0xffd95c,transparent:true,opacity:0.7});
  for(let j=1;j<order.length;j++){const b=curPos(order[j]),bm=Math.hypot(b[0],b[1],b[2])||1,bd=[b[0]/bm,b[1]/bm,b[2]/bm];
    queryGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(shellArc(ad,bd)),mat));}
  return order.length;
}

// ── Compare / morph (interpolate the cloud toward a second scene by id) ───
// setMorphTarget(sceneB) keys B's points by stable id; applyMorph(t) slerps
// each A-point's direction toward its id-matched B-direction (and lerps the
// radius) by t∈[0,1] — t=0 is A, t=1 is B. Points with no B match stay put.
function setMorphTarget(sceneB){
  morphTarget=null;
  if(!sceneB||!Array.isArray(sceneB.points))return 0;
  const map=new Map();
  for(const p of sceneB.points){if(p.id==null)continue;const x=+p.x,y=+p.y,z=+p.z;if(!isFinite(x)||!isFinite(y)||!isFinite(z))continue;const m=Math.hypot(x,y,z)||1;map.set(String(p.id),{d:[x/m,y/m,z/m],r:m});}
  morphTarget=map;
  // Push per-point morph targets into the GPU attributes the shader reads:
  // matched points carry their B direction/radius + a "has target" flag;
  // unmatched points are flagged off so the shader holds them at origPos.
  let matched=0;
  if(pointsGeo){
    const md=pointsGeo.getAttribute("aMorphDir"),mr=pointsGeo.getAttribute("aMorphR"),mh=pointsGeo.getAttribute("aMorphHas");
    for(let i=0;i<N;i++){const tgt=pts[i].id!=null?map.get(String(pts[i].id)):undefined;
      if(tgt){md.array[i*3]=tgt.d[0];md.array[i*3+1]=tgt.d[1];md.array[i*3+2]=tgt.d[2];mr.array[i]=tgt.r;mh.array[i]=1;matched++;}
      else{mh.array[i]=0;}}
    md.needsUpdate=true;mr.needsUpdate=true;mh.needsUpdate=true;
  }else{for(let i=0;i<N;i++)if(pts[i].id!=null&&map.has(String(pts[i].id)))matched++;}
  return matched;
}
// Morph slider: interpolate the cloud toward the id-matched target scene by
// t∈[0,1]. The per-point slerp runs in the vertex shader (gated on
// uHasMorph/uMorphT, using the aMorphDir/aMorphR/aMorphHas attributes that
// setMorphTarget filled); the antipodal great-circle sweep + radius lerp live
// in `sphTransform` (and its CPU mirror `curPos`). applyTransform pushes the
// uniforms and refreshes bridges/minimap.
function applyMorph(t){morphT=clamp(t,0,1);applyTransform();}
function clearMorph(){morphTarget=null;morphT=0;if(pointsMat){pointsMat.uniforms.uMorphT.value=0;pointsMat.uniforms.uHasMorph.value=0;}}

// ── Pins (drop annotated (θ,φ) markers on the globe shell) ────────────────
function setPinMode(on){pinOn=on;document.getElementById("tool-pin").classList.toggle("active",on);}
function clearPins(){pins=[];renderPins();}
function renderPins(){
  while(pinGroup.children.length){const c=pinGroup.children[0];disposeObject(c);pinGroup.remove(c);}
  pinsDiv.innerHTML="";pinEls=[];
  for(const pin of pins){
    const sp=Math.sin(pin.phi),dir=[sp*Math.cos(pin.theta),sp*Math.sin(pin.theta),Math.cos(pin.phi)],pos=[dir[0]*SR,dir[1]*SR,dir[2]*SR];
    pinGroup.add(marker(pos,0xffb454,SR*0.018));
    const el=document.createElement("div");el.className="plabel";
    const dot=document.createElement("span");dot.className="pdot";dot.textContent="📍";
    const txt=document.createElement("span");txt.textContent=pin.label; // textContent — never innerHTML
    el.appendChild(dot);el.appendChild(txt);el.title="click to remove";
    el.addEventListener("click",ev=>{ev.stopPropagation();const i=pins.indexOf(pin);if(i>=0){pins.splice(i,1);renderPins();}});
    pinsDiv.appendChild(el);pinEls.push({el,anchor:pos});
  }
}
function addPin(theta,phi,label){pins.push({theta,phi,label:label||("pin "+(pins.length+1))});renderPins();}
function updatePinLabels(){
  _cd.copy(camera.position).normalize();
  for(const {el,anchor} of pinEls){
    _av.set(anchor[0]*curScale,anchor[1]*curScale,anchor[2]*curScale);
    const al=Math.hypot(_av.x,_av.y,_av.z)||1;
    const facing=(_av.x*_cd.x+_av.y*_cd.y+_av.z*_cd.z)/al>-0.15;
    const sp=projectToScreen(anchor);
    if(sp.vis&&facing){el.style.display="flex";el.style.left=sp.x+"px";el.style.top=sp.y+"px";}else el.style.display="none";
  }
}

// ── Open / drop a foreign Scene ──────────────────────────────────────────
// Normalize an arbitrary parsed object into the Scene shape rebuild() expects.
// Accepts the full Scene (the `Scene::to_json` shape) or a bare points array,
// and tolerates minimal points: a point needs only finite x/y/z OR finite
// r/theta/phi — the missing pair is derived using the same convention as
// sphereql-core (x=r·sinφ·cosθ, y=r·sinφ·sinθ, z=r·cosφ; θ=atan2(y,x)∈[0,2π),
// φ=acos(z/r)). Throws (with a human message) when there is nothing to show.
function parseScene(obj){
  if(obj==null||typeof obj!=="object")throw new Error("not a JSON object");
  const raw=Array.isArray(obj)?{points:obj}:obj;
  if(!Array.isArray(raw.points))throw new Error("missing a `points` array");
  const out=[];
  for(const p of raw.points){
    if(!p||typeof p!=="object")continue;
    let x=+p.x,y=+p.y,z=+p.z,r=+p.r,theta=+p.theta,phi=+p.phi;
    const hasXYZ=isFinite(x)&&isFinite(y)&&isFinite(z);
    const hasSph=isFinite(r)&&isFinite(theta)&&isFinite(phi);
    if(!hasXYZ&&!hasSph)continue;
    if(!hasXYZ){const sp=Math.sin(phi);x=r*sp*Math.cos(theta);y=r*sp*Math.sin(theta);z=r*Math.cos(phi);}
    if(!hasSph){r=Math.hypot(x,y,z);theta=Math.atan2(y,x);if(theta<0)theta+=2*Math.PI;phi=Math.acos(clamp(r>1e-12?z/r:0,-1,1));}
    const q={x,y,z,r,theta,phi,cat:p.cat!=null?String(p.cat):"",label:p.label!=null?String(p.label):""};
    if(p.id!=null)q.id=String(p.id);
    if(isFinite(+p.certainty))q.certainty=+p.certainty;
    if(isFinite(+p.intensity))q.intensity=+p.intensity;
    out.push(q);
  }
  if(out.length===0)throw new Error("no usable points (each needs finite x/y/z or r/theta/phi)");
  // surface_radius: honour an explicit finite value, else the median ‖xyz‖.
  // Must match Rust `Scene::surface_radius_for` exactly — same filter
  // (finite AND > 0, so origin points don't drag the median down) and same
  // median index (norms[len/2]) — so a dropped scene lands on the same shell
  // the producer baked.
  let sr=+raw.surface_radius;
  if(!isFinite(sr)||sr<=0){const norms=out.map(p=>Math.hypot(p.x,p.y,p.z)).filter(n=>isFinite(n)&&n>0).sort((a,b)=>a-b);sr=norms.length?norms[norms.length>>1]:1.0;if(!isFinite(sr)||sr<=0)sr=1.0;}
  const st=raw.stats&&typeof raw.stats==="object"?raw.stats:{};
  // Coerce the count fields to finite numbers (they are interpolated into the
  // stats panel) so a hostile dropped scene can't smuggle markup through them.
  const sampled=isFinite(+st.sampled_from)?+st.sampled_from:undefined;
  const dropped=isFinite(+st.dropped_nonfinite)?+st.dropped_nonfinite:undefined;
  return{
    title:raw.title!=null?String(raw.title):"Imported scene",
    points:out,
    overlays:Array.isArray(raw.overlays)?raw.overlays.filter(o=>o&&typeof o==="object"&&typeof o.kind==="string"):[],
    stats:{projection_kind:st.projection_kind!=null?String(st.projection_kind):"imported",evr:isFinite(+st.evr)?+st.evr:0,evr_label:st.evr_label!=null?String(st.evr_label):"explained variance",sampled_from:sampled,dropped_nonfinite:dropped},
    surface_radius:sr,
    show_axes:!!raw.show_axes
  };
}
// Transient feedback on a button (✓/✗) without disturbing the rest of the UI.
function flashButton(id,text,ms){const b=document.getElementById(id);if(!b)return;const t=b.textContent;b.textContent=text;setTimeout(()=>{b.textContent=t;},ms||1500);}
function loadSceneFromText(text){
  let obj;
  try{obj=JSON.parse(text);}catch(err){console.warn("SphereQL: scene is not valid JSON:",err);flashButton("open-scene","✗ not JSON");return;}
  let sc;
  try{sc=parseScene(obj);}catch(err){console.warn("SphereQL: not a Scene:",err.message);flashButton("open-scene","✗ "+err.message,2200);return;}
  rebuild(sc);
  // Switch to the Overlays/Legend so the freshly-loaded scene is visible.
  flashButton("open-scene","✓ "+sc.points.length+" points");
}
function loadSceneFromFile(f){if(!f)return;const r=new FileReader();r.onload=()=>loadSceneFromText(r.result);r.onerror=()=>flashButton("open-scene","✗ read error");r.readAsText(f);}

// ── Scene swap ─────────────────────────────────────────────────────────────
function teardown(){
  disposeObject(pointsMesh);if(pointsMesh)scene.remove(pointsMesh);
  disposeObject(globeGroup);if(globeGroup)scene.remove(globeGroup);
  disposeObject(linesGroup);if(linesGroup)scene.remove(linesGroup);
  for(const k in overlayGroups){disposeObject(overlayGroups[k]);scene.remove(overlayGroups[k]);}
  overlayGroups={};
  if(pickMat&&pickMat.dispose)pickMat.dispose();
  pointsMesh=null;pointsGeo=null;pointsMat=null;pickMat=null;globeGroup=null;linesGroup=null;
  legendDiv.innerHTML="";oi.innerHTML="";labelTogglesDiv.innerHTML="";labelsDiv.innerHTML="";
  const info=document.getElementById("info");if(info)info.classList.remove("visible");
  tooltip.style.display="none";reticle.style.display="none";sellabel.style.display="none";
  setRuler(false); // fully disarm the ruler (flag + button + picks) on scene swap
  setPinMode(false);clearPins(); // pins annotated the outgoing scene's shell
  clearQuery(); // drop query geodesics referencing the outgoing scene
  clearMorph(); // morph target referenced the outgoing scene's ids
  selectedIdx=-1;hoveredIdx=-1;tgtTween=null;pendingTransform=false;
}

// Build the entire data-dependent scene (points, overlays, panels) from a
// Scene object `sc` — the same shape `Scene::to_json` emits. Resets all view
// settings to defaults so a swapped-in scene starts clean.
function rebuild(sc){
  teardown();
  pts=sc.points||[];N=pts.length;overlays=sc.overlays||[];SR=sc.surface_radius||1.0;showAxes=!!sc.show_axes;
  maxR=1;for(const p of pts){const m=Math.hypot(p.x,p.y,p.z);if(m>maxR)maxR=m;}

  // Reset view/transform state to defaults for the new scene.
  baseSize=DEF.size;curScale=DEF.scale;spreadF=DEF.spread;radialG=DEF.radial;uiScale=DEF.ui;
  soloCat=null;selectedIdx=-1;hoveredIdx=-1;pendingTransform=false;tgtTween=null;
  labelRefDist=DEF.scale*maxR*2.6;

  // Categories + colors.
  catSet=[...new Set(pts.map(p=>p.cat))].sort();
  catColor={};buildCatColor(DEF.palette);
  catVisible={};catSet.forEach(c=>catVisible[c]=true);
  catCounts={};pts.forEach(p=>catCounts[p.cat]=(catCounts[p.cat]||0)+1);

  document.getElementById("empty").style.display=N===0?"flex":"none";
  controls.minDistance=maxR*0.05;controls.maxDistance=maxR*100*8;

  // ── Reference globe (toggleable, scalable) ──────────────────────────────
  globeGroup=new THREE.Group();scene.add(globeGroup);
  globeGroup.add(new THREE.Mesh(new THREE.SphereGeometry(SR*0.999,48,48),
    new THREE.MeshBasicMaterial({color:0x0c1330,transparent:true,opacity:0.45,side:THREE.BackSide,depthWrite:false})));
  (function(){const mat=new THREE.LineBasicMaterial({color:0x5070c0,transparent:true,opacity:0.12}),SEG=120;
    for(let i=1;i<6;i++){const phi=Math.PI*i/6,rr=SR*Math.sin(phi),z=SR*Math.cos(phi),v=[];
      for(let s=0;s<=SEG;s++){const a=2*Math.PI*s/SEG;v.push(new THREE.Vector3(rr*Math.cos(a),rr*Math.sin(a),z));}
      globeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(v),mat));}
    for(let i=0;i<6;i++){const th=Math.PI*i/6,v=[];
      for(let s=0;s<=SEG;s++){const f=2*Math.PI*s/SEG;v.push(new THREE.Vector3(SR*Math.sin(f)*Math.cos(th),SR*Math.sin(f)*Math.sin(th),SR*Math.cos(f)));}
      globeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(v),mat));}})();
  if(showAxes){const ag=new THREE.BufferGeometry(),av=[],ac=[];
    [[1,0,0,1,.4,.4],[0,1,0,.4,1,.4],[0,0,1,.4,.55,1]].forEach(([x,y,z,r,g,b])=>{av.push(0,0,0,x*SR*1.25,y*SR*1.25,z*SR*1.25);ac.push(r,g,b,r,g,b);});
    ag.setAttribute("position",new THREE.Float32BufferAttribute(av,3));ag.setAttribute("color",new THREE.Float32BufferAttribute(ac,3));
    globeGroup.add(new THREE.LineSegments(ag,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.4})));}

  // ── Points (originals retained; crisp solid-disc shader) ────────────────
  origPos=new Float32Array(N*3);const positions=new Float32Array(N*3),colors=new Float32Array(N*3),sizes=new Float32Array(N);
  for(let i=0;i<N;i++){const p=pts[i],c=new THREE.Color(catColor[p.cat]);
    origPos[i*3]=p.x;origPos[i*3+1]=p.y;origPos[i*3+2]=p.z;positions[i*3]=p.x;positions[i*3+1]=p.y;positions[i*3+2]=p.z;
    colors[i*3]=c.r;colors[i*3+1]=c.g;colors[i*3+2]=c.b;sizes[i]=baseSize;}
  // Map a point's full-precision (x,y,z) → its index, so a bridge endpoint can
  // be tied to its exact source sphere (bridge.from === a point's xyz).
  posIndex=new Map();for(let i=0;i<N;i++)posIndex.set(pts[i].x+"|"+pts[i].y+"|"+pts[i].z,i);
  idToIndex=new Map();for(let i=0;i<N;i++){if(pts[i].id!=null)idToIndex.set(String(pts[i].id),i);}
  catDir={};
  {const sum={},cnt={};catSet.forEach(c=>{sum[c]=[0,0,0];cnt[c]=0;});
   for(let i=0;i<N;i++){const p=pts[i],m=Math.hypot(p.x,p.y,p.z)||1;sum[p.cat][0]+=p.x/m;sum[p.cat][1]+=p.y/m;sum[p.cat][2]+=p.z/m;cnt[p.cat]++;}
   catSet.forEach(c=>{const s=sum[c],m=Math.hypot(s[0],s[1],s[2]);catDir[c]=m>1e-9?[s[0]/m,s[1]/m,s[2]/m]:[0,0,1];});}
  catDirArr=Object.values(catDir);
  pointsGeo=new THREE.BufferGeometry();
  pointsGeo.setAttribute("position",new THREE.BufferAttribute(positions,3));
  pointsGeo.setAttribute("color",new THREE.BufferAttribute(colors,3));
  pointsGeo.setAttribute("size",new THREE.BufferAttribute(sizes,1));
  // Per-point angular density: bin the (original) directions into a θ×φ grid
  // (the minimap's scheme) and give each point its bin's normalized count, for
  // the density-shading heatmap toggle.
  {const GW=24,GH=12,bins=new Int32Array(GW*GH),binOf=new Int32Array(N),dens=new Float32Array(N);
   for(let i=0;i<N;i++){const p=pts[i],r=Math.hypot(p.x,p.y,p.z)||1;let th=Math.atan2(p.y,p.x);if(th<0)th+=2*Math.PI;const ph=Math.acos(clamp(p.z/r,-1,1));
     const gx=Math.min(GW-1,Math.floor(th/(2*Math.PI)*GW)),gy=Math.min(GH-1,Math.floor(ph/Math.PI*GH)),b=gy*GW+gx;binOf[i]=b;bins[b]++;}
   let maxBin=1;for(const v of bins)if(v>maxBin)maxBin=v;
   for(let i=0;i<N;i++)dens[i]=bins[binOf[i]]/maxBin;
   pointsGeo.setAttribute("density",new THREE.BufferAttribute(dens,1));}
  // Per-point transform inputs for the vertex shader: each point's category
  // centroid direction (the spread pivot) + its morph target (filled later by
  // setMorphTarget; benign identity defaults until then). With these the shader
  // transforms origPos by the uSpread/uRadial/uSR/uMorphT uniforms, so a slider
  // tick is a uniform write rather than an O(N) CPU buffer rewrite.
  {const cd=new Float32Array(N*3),mdv=new Float32Array(N*3),mrv=new Float32Array(N),mhv=new Float32Array(N);
   for(let i=0;i<N;i++){const c=catDir[pts[i].cat]||[0,0,1];cd[i*3]=c[0];cd[i*3+1]=c[1];cd[i*3+2]=c[2];mdv[i*3+2]=1;mrv[i]=1;}
   pointsGeo.setAttribute("aCatDir",new THREE.BufferAttribute(cd,3));
   pointsGeo.setAttribute("aMorphDir",new THREE.BufferAttribute(mdv,3));
   pointsGeo.setAttribute("aMorphR",new THREE.BufferAttribute(mrv,1));
   pointsGeo.setAttribute("aMorphHas",new THREE.BufferAttribute(mhv,1));}
  // Per-point pick id baked as an RGB color (read back from a 1px render in
  // getHovered to identify the point under the cursor).
  {const pc=new Float32Array(N*3);for(let i=0;i<N;i++){const c=pickEncode(i);pc[i*3]=c[0];pc[i*3+1]=c[1];pc[i*3+2]=c[2];}
   pointsGeo.setAttribute("aPickColor",new THREE.BufferAttribute(pc,3));}
  pointsMat=new THREE.ShaderMaterial({vertexColors:true,transparent:true,depthWrite:false,
    uniforms:{opacity:{value:1.0},densityOn:{value:DEF.density?1:0},uSpread:{value:DEF.spread},uRadial:{value:DEF.radial},uSR:{value:SR},uMorphT:{value:0},uHasMorph:{value:0}},
    vertexShader:VERTEX_TRANSFORM+`attribute float size;attribute float density;varying vec3 vc;varying float vd;void main(){vc=color;vd=density;vec3 tp=sphTransform(position);vec4 mv=modelViewMatrix*vec4(tp,1.0);gl_PointSize=size*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
    fragmentShader:`uniform float opacity;uniform float densityOn;varying vec3 vc;varying float vd;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;float a=smoothstep(0.5,0.44,d)*opacity;float core=smoothstep(0.32,0.0,d);vec3 col=mix(vc,vec3(1.0),core*0.4);if(densityOn>0.5){vec3 cold=vec3(0.25,0.5,1.0),hot=vec3(1.0,0.62,0.22);col=mix(cold,hot,vd);a*=mix(0.5,1.0,vd);}gl_FragColor=vec4(col,a);}`});
  // Pick material: same transform (shares pointsMat's uniform objects, so
  // slider/morph updates apply to both) + size, but writes each point's baked
  // id-color instead of its shaded color. Used only by the offscreen pick pass.
  pickMat=new THREE.ShaderMaterial({uniforms:pointsMat.uniforms,
    vertexShader:VERTEX_TRANSFORM+`attribute float size;attribute vec3 aPickColor;varying vec3 vpick;void main(){vpick=aPickColor;vec3 tp=sphTransform(position);vec4 mv=modelViewMatrix*vec4(tp,1.0);gl_PointSize=size*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
    fragmentShader:`varying vec3 vpick;void main(){gl_FragColor=vec4(vpick,1.0);}`});
  // The shader can push points beyond origPos's bounding sphere (radial/spread),
  // so skip frustum culling — otherwise the whole cloud can vanish at the edges.
  pointsMesh=new THREE.Points(pointsGeo,pointsMat);pointsMesh.frustumCulled=false;scene.add(pointsMesh);
  linesGroup=new THREE.Group();scene.add(linesGroup);

  // ── Overlays ────────────────────────────────────────────────────────────
  overlayGroups={};overlayKinds=new Set();bridgeLines=[];bridgesByPoint={};labelData=[];
  overlays.forEach(o=>{
   // A single malformed overlay (e.g. a dropped scene's bridge missing `from`)
   // must not abort the whole rebuild — skip it and keep the rest of the scene.
   try{
    if(!o||typeof o!=="object"||typeof o.kind!=="string")return;
    overlayKinds.add(o.kind);const g=groupFor(o.kind);const col=o.color?new THREE.Color(o.color).getHex():0x5cc8ff;
    if(o.kind==="centroid"){g.add(marker(o.pos,col,SR*0.022));labelData.push({kind:"centroid",anchor:o.pos,text:o.label,color:o.color||"#5cc8ff",cat:o.label});}
    else if(o.kind==="bridge"){const ch=classColor[o.classification]!==undefined?classColor[o.classification]:col;const ln=lineBetween(o.from,o.to,ch,0.18+0.55*(o.strength||0.5));g.add(ln);const fi=posIndex.get(o.from[0]+"|"+o.from[1]+"|"+o.from[2]),fidx=fi===undefined?-1:fi;bridgeLines.push({line:ln,from:o.from,to:o.to,fromIndex:fidx,color:ch});if(fidx>=0)(bridgesByPoint[fidx]||(bridgesByPoint[fidx]=[])).push({to:o.to,color:ch});}
    else if(o.kind==="geodesic_path"){g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints((o.vertices||[]).map(v3)),new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.9})));}
    else if(o.kind==="voronoi_cap"){g.add(capRing(o.center,o.half_angle,col));}
    else if(o.kind==="antipode"){g.add(marker(o.centroid,col,SR*0.016));g.add(marker(o.antipode,col,SR*0.016));g.add(lineBetween(o.centroid,o.antipode,col,0.18));labelData.push({kind:"antipode",anchor:o.antipode,text:"⤬ "+o.label,color:o.color||"#5cc8ff"});}
    else if(o.kind==="coverage_void"){(o.caps||[]).forEach(c=>g.add(capRing(c.center,c.half_angle,0x5cc8ff)));(o.voids||[]).forEach(vp=>g.add(marker(vp,0x222a40,SR*0.006)));}
    else if(o.kind==="domain_group"){g.add(marker(o.centroid,col,SR*0.027));(o.members||[]).forEach(m=>g.add(lineBetween(o.centroid,m,col,0.28)));labelData.push({kind:"domain_group",anchor:o.centroid,text:o.label,color:o.color||"#5cc8ff"});}
    else if(o.kind==="glob"){const m=new THREE.Mesh(new THREE.SphereGeometry(o.radius,22,22),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.1,depthWrite:false}));m.position.copy(v3(o.center));g.add(m);}
    else if(o.kind==="manifold_slice"){const pl=new THREE.Mesh(new THREE.PlaneGeometry(SR,SR),new THREE.MeshBasicMaterial({color:0x6f8fc8,transparent:true,opacity:0.12,side:THREE.DoubleSide}));pl.position.copy(v3(o.center));pl.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),v3(o.normal).normalize());g.add(pl);}
   }catch(err){console.warn("SphereQL: skipping malformed overlay",o&&o.kind,err);}
  });
  overlayKinds.forEach(k=>{if(overlayDefaultOff.has(k))overlayGroups[k].visible=false;});

  scalables=[pointsMesh,linesGroup,globeGroup,rulerGroup,queryGroup,pinGroup,...Object.values(overlayGroups)];

  // ── Legend + counts ─────────────────────────────────────────────────────
  legendRows={};
  catSet.forEach(cat=>{
    const row=document.createElement("div");row.className="lrow";
    row.innerHTML=`<span class="ldot" style="background:${catColor[cat]};color:${catColor[cat]}"></span><span class="lbl"></span><span class="lcnt">${catCounts[cat]}</span>`;
    row.querySelector(".lbl").textContent=cat;legendRows[cat]=row;
    row.addEventListener("click",()=>{catVisible[cat]=!catVisible[cat];row.classList.toggle("dim",!catVisible[cat]);updateVisibility();});
    legendDiv.appendChild(row);});

  // ── Overlay toggles ─────────────────────────────────────────────────────
  if(overlayKinds.size>0){[...overlayKinds].sort().forEach(kind=>{
    const on=overlayGroups[kind].visible,row=document.createElement("label");row.className="orow";
    row.innerHTML=`<input type="checkbox" ${on?"checked":""}><span></span>`;
    row.querySelector("span").textContent=OVERLAY_LABELS[kind]||kind;
    row.querySelector("input").addEventListener("change",e=>{overlayGroups[kind].visible=e.target.checked;});
    oi.appendChild(row);});}
  else{oi.innerHTML='<div class="muted">No overlays in this scene.</div>';}

  // ── Floating labels + per-kind toggles ──────────────────────────────────
  labelEls=[];labelKindOn={};
  labelData.forEach(ld=>{
    const el=document.createElement("div");el.className="vlabel";
    const dot=document.createElement("span");dot.className="vdot";dot.style.background=ld.color;dot.style.color=ld.color;
    const txt=document.createElement("span");txt.textContent=ld.text;
    el.appendChild(dot);el.appendChild(txt);
    el.title="click to focus"+(ld.kind==="centroid"?" · solo domain":"");
    el.addEventListener("click",ev=>{ev.stopPropagation();focusLabel(ld);});
    labelsDiv.appendChild(el);labelEls.push({el,ld});
  });
  labelKindsPresent=[...new Set(labelData.map(l=>l.kind))].sort();
  if(labelKindsPresent.length===0)labelTogglesDiv.innerHTML='<div class="muted">No labelled overlays.</div>';
  labelKindsPresent.forEach(kind=>{
    labelKindOn[kind]=true;
    const row=document.createElement("label");row.className="orow";
    row.innerHTML=`<input type="checkbox" checked><span></span>`;
    row.querySelector("span").textContent=LABEL_KIND_NAMES[kind]||kind;
    row.querySelector("input").addEventListener("change",e=>{labelKindOn[kind]=e.target.checked;});
    labelTogglesDiv.appendChild(row);
  });

  // ── Header + stats ──────────────────────────────────────────────────────
  const st=sc.stats||{};
  document.getElementById("hdr-sub").textContent=sc.title||"";
  document.getElementById("hdr-pill").textContent=st.projection_kind||"—";
  let rows="";
  if(N>0){const rV=pts.map(p=>p.r),thV=pts.map(p=>p.theta),phV=pts.map(p=>p.phi),evr=st.evr||0;
    rows=`
<div class="srow"><span>points</span><span class="v">${N.toLocaleString()}</span></div>
<div class="srow"><span>domains</span><span class="v">${catSet.length}</span></div>
<div class="srow"><span>projection</span><span class="v hl">${escHtml(st.projection_kind||"?")}</span></div>
<div class="srow"><span>${escHtml(st.evr_label||"explained variance")}</span><span class="v">${(evr*100).toFixed(1)}%</span></div>
<div class="bar"><i style="width:${clamp(evr*100,0,100).toFixed(1)}%"></i></div>
<div class="srow"><span>r</span><span class="v">${fmin(rV).toFixed(2)} – ${fmax(rV).toFixed(2)}</span></div>
<div class="srow"><span>θ</span><span class="v">${fmin(thV).toFixed(2)} – ${fmax(thV).toFixed(2)}</span></div>
<div class="srow"><span>φ</span><span class="v">${fmin(phV).toFixed(2)} – ${fmax(phV).toFixed(2)}</span></div>`;
  }else{rows=`<div class="srow"><span>points</span><span class="v">0</span></div>`;}
  if(st.sampled_from)rows+=`<div class="note">▴ sample of ${escHtml(st.sampled_from.toLocaleString())}</div>`;
  if(st.dropped_nonfinite)rows+=`<div class="note">▴ ${escHtml(st.dropped_nonfinite.toLocaleString())} non-finite dropped</div>`;
  statsDiv.innerHTML=rows;

  // ── Sync the static controls back to defaults for this scene ─────────────
  scaleS.value=DEF.scale;scaleVal.textContent=DEF.scale.toFixed(1)+"×";
  zoomS.value=DEF.zoom;zoomVal.textContent=DEF.zoom.toFixed(2)+"×";controls.zoomSpeed=DEF.zoom;
  radial.value=DEF.radial;radialVal.textContent=DEF.radial.toFixed(1)+"×";
  spread.value=DEF.spread;spreadVal.textContent=DEF.spread.toFixed(1)+"×";
  psize.value=DEF.size;sizeVal.textContent=DEF.size.toFixed(1);
  uiS.value=DEF.ui;uiVal.textContent=DEF.ui.toFixed(2)+"×";document.documentElement.style.setProperty("--ui",DEF.ui);
  schemeSel.value=DEF.palette;
  globeCb.checked=DEF.globe;globeGroup.visible=DEF.globe;
  autorot.checked=DEF.autorot;controls.autoRotate=DEF.autorot;
  densityCb.checked=DEF.density; // densityOn uniform already seeded from DEF
  searchInput.value="";

  // ── Init view (default scale + framing) ─────────────────────────────────
  applyScale(DEF.scale);drawMinimapBase();frameCamera();
}

// ── DataSource: where scene data comes from (inline blob vs streaming) ────
// Two ways to feed the viewer behind one async interface, so callers (the
// studio, a future streaming renderer) don't care which backs them:
//   manifest()      → {title,total_points,surface_radius,bounds,stats,overlays,palette,lod}
//   tiles(params)   → {count, positions:Float32Array(3n), cats:Uint16Array(n), rows:Uint32Array(n)}
//   pointMeta(rows) → [{row,label,cat,category,certainty,intensity,x,y,z,r,theta,phi}]
//   nearest(q,k)    → [{row,similarity}]   (q = {row} | {vector:[…]})
// The offline baked file inlines the whole scene as `D` and renders all of it
// (InlineSource). A server-backed build streams the visible working set as
// binary tiles (ServerSource); wiring that into a per-tile renderer is a later
// phase, but the contract + client live here and are exercised by js-tests.
// `tiles()` params {theta,phi,half_angle,budget,lod} describe a viewport cone +
// detail budget; InlineSource has the whole cloud so it ignores the cone and
// only honours `budget` (the same stratified decimation the server uses).

// Decode a binary SQT1 tile — the wire form emitted by sphereql-vis tile.rs:
//   header 16B: magic "SQT1" · version u16 · flags u16 · count u32 · reserved u32
//   record 20B: x f32 · y f32 · z f32 · cat u16 · _pad u16 · row u32   (all LE)
// Accepts an ArrayBuffer or a Uint8Array; throws on a malformed/short buffer.
function decodeTile(input){
  const bytes=input instanceof Uint8Array?input:new Uint8Array(input);
  if(bytes.length<16)throw new Error("tile shorter than 16-byte header");
  if(bytes[0]!==0x53||bytes[1]!==0x51||bytes[2]!==0x54||bytes[3]!==0x31)throw new Error("tile magic is not SQT1");
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const version=dv.getUint16(4,true);
  if(version>1)throw new Error("unsupported tile version "+version);
  const count=dv.getUint32(8,true),REC=20,actual=Math.floor((bytes.length-16)/REC);
  if(actual!==count)throw new Error("tile declares "+count+" records but holds "+actual);
  const positions=new Float32Array(count*3),cats=new Uint16Array(count),rows=new Uint32Array(count);
  for(let i=0;i<count;i++){const o=16+i*REC;
    positions[i*3]=dv.getFloat32(o,true);positions[i*3+1]=dv.getFloat32(o+4,true);positions[i*3+2]=dv.getFloat32(o+8,true);
    cats[i]=dv.getUint16(o+12,true);rows[i]=dv.getUint32(o+16,true);}
  return{count,positions,cats,rows};
}
// Sorted-unique category names — the stable cat→id ordering shared by rebuild()
// (`catSet`) and the server palette, so a tile's `cat` ids mean the same thing
// no matter which source produced it.
function catOrder(points){return[...new Set((points||[]).map(p=>p&&p.cat!=null?String(p.cat):""))].sort();}
// Proportional-per-category + even-stride thinning to ~budget indices; mirrors
// the server's stratified tile decimation so both sources thin a cloud the same
// deterministic way (every non-empty category keeps at least one point).
function stratify(points,catIds,budget){
  if(points.length<=budget)return points.map((_,i)=>i);
  const groups=new Map();
  for(let i=0;i<points.length;i++){const c=catIds[i];let g=groups.get(c);if(!g){g=[];groups.set(c,g);}g.push(i);}
  const total=points.length,out=[];
  for(const c of[...groups.keys()].sort((a,b)=>a-b)){const grp=groups.get(c);
    const share=Math.round(grp.length/total*budget),take=Math.max(1,Math.min(share,grp.length)),stride=Math.max(1,Math.floor(grp.length/take));
    for(let i=0,n=0;i<grp.length&&n<take;i+=stride,n++)out.push(grp[i]);}
  if(out.length>budget)out.length=budget;
  return out;
}
// Build the /tiles query string from a params object (finite fields only).
function tileQuery(p){p=p||{};const q=[];for(const k of["theta","phi","half_angle","budget","lod"])if(p[k]!=null&&isFinite(+p[k]))q.push(k+"="+(+p[k]));return q.join("&");}

// InlineSource — the offline blob. Renders all of `D`; serves the streaming
// interface from the in-memory scene. nearest() here is a *positional* cosine
// (the inline file has no raw embeddings) — a local stand-in for the server's
// ANN over the original vectors.
class InlineSource{
  constructor(scene){this.scene=scene&&typeof scene==="object"?scene:{points:[]};const pts=this.scene.points||[];this._cats=catOrder(pts);this._catId=new Map(this._cats.map((c,i)=>[c,i]));}
  _id(cat){const v=this._catId.get(cat!=null?String(cat):"");return v===undefined?0:v;}
  async manifest(){
    const pts=this.scene.points||[],counts={};for(const p of pts){const c=p.cat!=null?String(p.cat):"";counts[c]=(counts[c]||0)+1;}
    let sr=+this.scene.surface_radius;if(!isFinite(sr)||sr<=0)sr=1;
    const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for(const p of pts){const c=[+p.x,+p.y,+p.z];for(let k=0;k<3;k++){if(c[k]<min[k])min[k]=c[k];if(c[k]>max[k])max[k]=c[k];}}
    if(!pts.length){min[0]=min[1]=min[2]=-1;max[0]=max[1]=max[2]=1;}
    const pal=PALETTES.aurora;
    return{title:this.scene.title||"",total_points:pts.length,surface_radius:sr,bounds:{min,max},
      stats:this.scene.stats||{},overlays:this.scene.overlays||[],
      palette:this._cats.map((name,i)=>({name,color:pal[i%pal.length],count:counts[name]||0})),
      lod:{levels:4,base_budget:20000}};
  }
  async tiles(params){
    const pts=this.scene.points||[],catIds=pts.map(p=>this._id(p.cat));
    const budget=Math.max(1,(params&&+params.budget)||pts.length||1);
    const idx=stratify(pts,catIds,budget);
    const positions=new Float32Array(idx.length*3),cats=new Uint16Array(idx.length),rows=new Uint32Array(idx.length);
    for(let j=0;j<idx.length;j++){const i=idx[j],p=pts[i];positions[j*3]=+p.x;positions[j*3+1]=+p.y;positions[j*3+2]=+p.z;cats[j]=catIds[i];rows[j]=i;}
    return{count:idx.length,positions,cats,rows};
  }
  async pointMeta(rows){
    const pts=this.scene.points||[],out=[];
    for(const row of rows||[]){const p=pts[row];if(!p)continue;
      const x=+p.x,y=+p.y,z=+p.z;let r=+p.r,theta=+p.theta,phi=+p.phi;
      // Derive missing spherical coords from xyz (same convention as parseScene)
      // so meta is complete even for a scene that only carried Cartesian coords.
      if(!(isFinite(r)&&isFinite(theta)&&isFinite(phi))){r=Math.hypot(x,y,z);theta=Math.atan2(y,x);if(theta<0)theta+=2*Math.PI;phi=Math.acos(clamp(r>1e-12?z/r:0,-1,1));}
      out.push({row,label:p.label||"",cat:this._id(p.cat),category:p.cat!=null?String(p.cat):"",
        certainty:isFinite(+p.certainty)?+p.certainty:null,intensity:isFinite(+p.intensity)?+p.intensity:null,
        x,y,z,r,theta,phi});}
    return out;
  }
  async nearest(q,k){
    const pts=this.scene.points||[];k=Math.max(1,Math.min(k||10,256));
    let rx,ry,rz,self=-1;
    if(q&&q.row!=null&&pts[q.row]){const p=pts[q.row],m=Math.hypot(p.x,p.y,p.z)||1;rx=p.x/m;ry=p.y/m;rz=p.z/m;self=q.row;}
    else if(q&&Array.isArray(q.vector)&&q.vector.length>=3){const v=q.vector,m=Math.hypot(v[0],v[1],v[2])||1;rx=v[0]/m;ry=v[1]/m;rz=v[2]/m;}
    else return[];
    const hits=[];
    for(let i=0;i<pts.length;i++){if(i===self)continue;const p=pts[i],m=Math.hypot(p.x,p.y,p.z)||1;hits.push({row:i,similarity:(p.x*rx+p.y*ry+p.z*rz)/m});}
    hits.sort((a,b)=>b.similarity-a.similarity);
    return hits.slice(0,k);
  }
}

// Bounded in-memory (LRU) + optional IndexedDB cache for fetched tile blobs,
// keyed by the tile request. IndexedDB persists across reloads when present
// (browser); the memory tier alone suffices for tests and locked-down embeds.
function idbReq(req){return new Promise((res,rej)=>{req.onsuccess=()=>res(req.result);req.onerror=()=>rej(req.error);});}
class TileCache{
  constructor(opts){opts=opts||{};this.max=opts.max||256;this.mem=new Map();this.dbName=opts.dbName||"sphereql-tiles";this.store="tiles";
    this._idb=opts.indexedDB!==undefined?opts.indexedDB:(typeof indexedDB!=="undefined"?indexedDB:null);this._dbp=null;}
  _touch(key,val){this.mem.delete(key);this.mem.set(key,val);while(this.mem.size>this.max)this.mem.delete(this.mem.keys().next().value);}
  async get(key){
    if(this.mem.has(key)){const v=this.mem.get(key);this._touch(key,v);return v;}
    const db=await this._open();if(!db)return null;
    try{const v=await idbReq(db.transaction(this.store,"readonly").objectStore(this.store).get(key));if(v!=null){this._touch(key,v);return v;}}catch(e){}
    return null;
  }
  async put(key,buf){this._touch(key,buf);const db=await this._open();if(!db)return;
    try{db.transaction(this.store,"readwrite").objectStore(this.store).put(buf,key);}catch(e){}}
  _open(){
    if(!this._idb)return Promise.resolve(null);
    if(!this._dbp)this._dbp=new Promise(resolve=>{try{const req=this._idb.open(this.dbName,1);
      req.onupgradeneeded=()=>{try{req.result.createObjectStore(this.store);}catch(e){}};
      req.onsuccess=()=>resolve(req.result);req.onerror=()=>resolve(null);}catch(e){resolve(null);}});
    return this._dbp;
  }
}

// ServerSource — streams from the sphereql-vis-server HTTP API. Tiles arrive as
// binary SQT1 and are decoded (off-thread via a Worker-backed `decode`, else
// inline); blobs can be cached. The injectable `fetch` keeps it testable. Used
// by the streaming renderer and the studio's "connect to server" mode.
class ServerSource{
  constructor(baseUrl,opts){opts=opts||{};this.base=String(baseUrl||"").replace(/\/+$/,"");
    this._fetch=opts.fetch||(typeof fetch!=="undefined"?fetch.bind(typeof globalThis!=="undefined"?globalThis:null):null);
    this.cache=opts.cache||null;this.decode=opts.decode||decodeTile;}
  async manifest(){return this._json("/manifest");}
  async categoryStats(){return this._json("/category_stats");}
  async diagnostics(){return this._json("/diagnostics");}
  // Live "tune": ask the server to re-project the corpus with a different kind;
  // returns the fresh manifest (re-stream tiles after this to pick up the new
  // positions).
  async reproject(projection){return this._post("/reproject",{projection});}
  async tiles(params){
    const key="/tiles?"+tileQuery(params);
    let buf=this.cache?await this.cache.get(key):null;
    if(buf==null){const res=await this._fetch(this.base+key);if(!res.ok)throw new Error("tiles → "+res.status);buf=await res.arrayBuffer();if(this.cache)await this.cache.put(key,buf);}
    // Decode a throwaway copy when caching: a worker-backed `decode` transfers
    // (detaches) the buffer it is handed, which would corrupt the retained
    // cache entry and break every subsequent cache hit. The cache keeps the
    // pristine blob; the copy is what gets transferred.
    return this.decode(this.cache?buf.slice(0):buf);
  }
  async pointMeta(rows){return(await this._post("/points",{rows:rows||[]})).points||[];}
  async nearest(q,k){const body={k:k||10};if(q&&q.row!=null)body.row=q.row;if(q&&Array.isArray(q.vector))body.vector=q.vector;return(await this._post("/nearest",body)).neighbors||[];}
  async _json(path){const res=await this._fetch(this.base+path);if(!res.ok)throw new Error(path+" → "+res.status);return res.json();}
  async _post(path,body){const res=await this._fetch(this.base+path,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});if(!res.ok)throw new Error(path+" → "+res.status);return res.json();}
}

// Off-thread tile decode: a tiny Worker built from inlined source (so the file
// stays self-contained) that runs decodeTile and transfers the typed arrays
// back. Returns an async decode(buf); falls back to inline decode when Workers
// are unavailable (Node tests, locked-down embeds).
function makeWorkerDecoder(){
  if(typeof Worker==="undefined"||typeof URL==="undefined"||!URL.createObjectURL||typeof Blob==="undefined")return decodeTile;
  let worker;const pending=new Map();let seq=0;
  try{
    const src="var decodeTile="+decodeTile.toString()+";self.onmessage=function(e){try{var d=decodeTile(e.data.buf);self.postMessage({id:e.data.id,d:d},[d.positions.buffer,d.cats.buffer,d.rows.buffer]);}catch(err){self.postMessage({id:e.data.id,err:String(err&&err.message||err)});}};";
    worker=new Worker(URL.createObjectURL(new Blob([src],{type:"text/javascript"})));
    worker.onmessage=e=>{const cb=pending.get(e.data.id);if(!cb)return;pending.delete(e.data.id);if(e.data.err)cb.rej(new Error(e.data.err));else cb.res(e.data.d);};
  }catch(err){return decodeTile;}
  return buf=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});try{worker.postMessage({id,buf},[buf]);}catch(err){pending.delete(id);try{res(decodeTile(buf));}catch(e2){rej(e2);}}});
}

// ── Streaming tile renderer (out-of-core: only the visible working set) ────
// TileStreamer turns camera motion into tile requests against a DataSource and
// keeps a bounded working set of per-tile point meshes — added/removed through
// an injected `sink` (the THREE layer) — so the renderer never holds all N
// points. It keeps a persistent COARSE base tile (whole sphere, LOD 0) for
// context plus a DETAIL tile for where the camera looks (finer LOD as you zoom
// in); recently-seen detail tiles stay cached up to a mesh budget (LRU). Loads
// are guard-checked on resolve so a tile that arrived after its entry was
// evicted/cleared is dropped rather than shown. Pure orchestration — the only
// THREE contact is sink.addTile(key,{count,positions,cats,rows})/removeTile(key).
class TileStreamer{
  constructor(source,sink,opts){
    opts=opts||{};
    this.source=source;this.sink=sink;this.manifest=null;
    this.maxDetail=opts.maxDetail||48;
    this.lodLevels=opts.lodLevels||4;
    this.baseBudget=opts.baseBudget||20000;
    this.detailBudget=opts.detailBudget||40000;
    this.near=opts.near||1.05;this.far=opts.far||8;
    this.tiles=new Map();this._clock=0;this.filter={};
  }
  // The active filter as server tile params ({cats:"0,3", min_certainty:0.5}).
  _filterParams(){const f=this.filter||{},o={};if(Array.isArray(f.cats)&&f.cats.length)o.cats=f.cats.join(",");if(isFinite(f.minCertainty))o.min_certainty=+f.minCertainty;return o;}
  // Apply a {cats:[ids], minCertainty} filter: drop the working set and reload
  // the base so the whole streamed view reflects it (detail tiles reload on the
  // next update). Pass {} / null to clear.
  async setFilter(filter){this.filter=filter||{};this.clear();this.tiles=new Map();await this._ensureBase();}
  async start(){return this.startWith(await this.source.manifest());}
  // Configure from an already-fetched manifest (connectToServer fetches it once
  // to build the scene chrome, then hands it here) and load the base tile.
  async startWith(manifest){
    this.manifest=manifest;
    if(manifest){
      const lod=manifest.lod||{};
      if(lod.levels)this.lodLevels=lod.levels;
      if(lod.base_budget)this.baseBudget=lod.base_budget;
      const sr=manifest.surface_radius||1;this.near=sr*1.05;this.far=sr*8;
    }
    await this._ensureBase();
    return manifest;
  }
  async _ensureBase(){
    if(this.tiles.has("base"))return;
    this.tiles.set("base",{key:"base",base:true,state:"loading",used:this._clock++});
    const data=await this.source.tiles({half_angle:Math.PI,budget:this.baseBudget,lod:0,...this._filterParams()});
    const t=this.tiles.get("base");if(!t)return; // cleared while loading
    t.state="loaded";t.data=data;this.sink.addTile("base",data);
  }
  // Camera distance → LOD: near the shell = finest, far = coarsest.
  lodFor(dist){
    const lv=this.lodLevels;
    if(!isFinite(dist)||dist<=this.near)return lv-1;
    const t=clamp((this.far-dist)/(this.far-this.near),0,1);
    return clamp(Math.round(t*(lv-1)),0,lv-1);
  }
  // Camera → the detail tile request: a cone aimed where the camera looks, that
  // narrows as you zoom in, at a distance-derived LOD/budget.
  requestFor(cam){
    const lod=this.lodFor(cam.dist);
    const f=clamp((cam.dist-this.near)/(this.far-this.near),0,1);
    const ha=clamp(0.18+f*1.4,0.12,Math.PI);
    return {theta:isFinite(cam.theta)?cam.theta:0,phi:isFinite(cam.phi)?cam.phi:Math.PI/2,half_angle:ha,lod,budget:this.detailBudget,...this._filterParams()};
  }
  // Quantized key so small camera jitter maps to the same tile (debounce-by-key).
  keyFor(req){const q=v=>Math.round(v*12)/12;return "d:"+q(req.theta)+":"+q(req.phi)+":"+req.lod;}
  // Ensure the detail tile for this camera is loaded; touch it for LRU; evict
  // the least-recently-used detail tiles past the budget. Safe under rapid
  // moves: an identical viewport key dedups to a touch (no refetch).
  async update(cam){
    const req=this.requestFor(cam),key=this.keyFor(req);
    const existing=this.tiles.get(key);
    if(existing){existing.used=this._clock++;return key;}
    this.tiles.set(key,{key,base:false,state:"loading",used:this._clock++});
    let data;
    try{data=await this.source.tiles(req);}catch(e){this.tiles.delete(key);return null;}
    const t=this.tiles.get(key);
    if(!t)return null; // evicted / cleared while in flight → drop
    t.state="loaded";t.data=data;this.sink.addTile(key,data);
    this._evict();
    return key;
  }
  _evict(){
    const detail=[...this.tiles.values()].filter(t=>!t.base&&t.state==="loaded").sort((a,b)=>a.used-b.used);
    for(let i=0;i<detail.length-this.maxDetail;i++){this.tiles.delete(detail[i].key);this.sink.removeTile(detail[i].key);}
  }
  loadedKeys(){return[...this.tiles.values()].filter(t=>t.state==="loaded").map(t=>t.key);}
  clear(){for(const k of this.tiles.keys())this.sink.removeTile(k);this.tiles.clear();}
}

// Per-tile THREE.Points for the streamer's working set. Each tile geometry
// carries the streamed positions, palette colours (by cat id), point sizes, and
// the per-point pick id baked from the GLOBAL row (so id-buffer picking resolves
// a row across tiles). Server positions are final — the streaming material does
// no client-side transform.
function tileMeshSink(group,palette,material){
  const colors=(palette||[]).map(c=>new THREE.Color(c.color));
  const meshes=new Map();
  function addTile(key,data){
    if(meshes.has(key))removeTile(key);
    const n=data.count|0,geo=new THREE.BufferGeometry();
    const col=new Float32Array(n*3),size=new Float32Array(n),pick=new Float32Array(n*3);
    for(let i=0;i<n;i++){const c=colors[data.cats[i]]||new THREE.Color(0x90a4ae);col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;size[i]=baseSize;
      const pc=pickEncode(data.rows[i]);pick[i*3]=pc[0];pick[i*3+1]=pc[1];pick[i*3+2]=pc[2];}
    geo.setAttribute("position",new THREE.BufferAttribute(data.positions,3));
    geo.setAttribute("color",new THREE.BufferAttribute(col,3));
    geo.setAttribute("size",new THREE.BufferAttribute(size,1));
    geo.setAttribute("aPickColor",new THREE.BufferAttribute(pick,3));
    const mesh=new THREE.Points(geo,material);mesh.frustumCulled=true;
    mesh.userData={rows:data.rows}; // global rows, for CPU picking → inspector
    group.add(mesh);meshes.set(key,mesh);
  }
  // Dispose only the per-tile geometry — the material is shared across all
  // tiles (disposing it would free an in-use GPU program and force recompiles
  // on every eviction).
  function removeTile(key){const m=meshes.get(key);if(m){group.remove(m);if(m.geometry&&m.geometry.dispose)m.geometry.dispose();meshes.delete(key);}}
  function clear(){for(const k of[...meshes.keys()])removeTile(k);}
  return {addTile,removeTile,clear,count:()=>meshes.size,meshAt:k=>meshes.get(k)};
}

// Shared material for streamed tiles (solid-disc points, palette-coloured). No
// sphTransform — the server already projected the positions.
let _streamColorMat=null;
function streamColorMaterial(){
  if(_streamColorMat)return _streamColorMat;
  _streamColorMat=new THREE.ShaderMaterial({vertexColors:true,transparent:true,depthWrite:false,uniforms:{opacity:{value:1.0}},
    vertexShader:`attribute float size;varying vec3 vc;void main(){vc=color;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=size*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
    fragmentShader:`uniform float opacity;varying vec3 vc;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;float a=smoothstep(0.5,0.44,d)*opacity;float core=smoothstep(0.32,0.0,d);gl_FragColor=vec4(mix(vc,vec3(1.0),core*0.4),a);}`});
  return _streamColorMat;
}

// Streaming-mode legend from the manifest palette (name · count, in the
// category colour). Category-toggle filtering of streamed tiles is a later
// (Phase D) refinement; here it labels what's on screen.
// Accept only a safe CSS color literal (hex / rgb[a] / hsl[a] / plain word) from
// a server-supplied palette; otherwise fall back. Blocks CSS-attribute
// injection via a crafted manifest color (the palette comes from whatever server
// the #server hash names).
function safeColor(c){return typeof c==="string"&&/^#[0-9a-fA-F]{3,8}$|^rgba?\([\d.,\s%]+\)$|^hsla?\([\d.,\s%]+\)$|^[a-zA-Z]+$/.test(c.trim())?c.trim():"#90a4ae";}
function buildStreamLegend(palette){
  legendDiv.innerHTML="";legendRows={};catColor={};catVisible={};
  (palette||[]).forEach(c=>{
    const col=safeColor(c.color);catColor[c.name]=col;catVisible[c.name]=!_streamFilterOff.has(c.name);
    const row=document.createElement("div");row.className="lrow"+(_streamFilterOff.has(c.name)?" dim":"");
    row.innerHTML=`<span class="ldot" style="background:${col};color:${col}"></span><span class="lbl"></span><span class="lcnt">${(c.count||0).toLocaleString()}</span>`;
    row.querySelector(".lbl").textContent=c.name;legendRows[c.name]=row;
    // Click a category to filter it out of the stream (server-side tile filter).
    row.addEventListener("click",()=>{
      if(_streamFilterOff.has(c.name))_streamFilterOff.delete(c.name);else _streamFilterOff.add(c.name);
      row.classList.toggle("dim",_streamFilterOff.has(c.name));catVisible[c.name]=!_streamFilterOff.has(c.name);
      applyStreamFilter();
    });
    legendDiv.appendChild(row);
  });
}
// Recompute the streaming filter (enabled categories + min-certainty) and push
// it to the streamer, then refresh diagnostics.
function applyStreamFilter(){
  if(!streamStreamer)return;
  const cats=[];_streamPalette.forEach((c,i)=>{if(!_streamFilterOff.has(c.name))cats.push(i);});
  const allOn=cats.length===_streamPalette.length;
  const mcEl=document.getElementById("mincert"),mc=mcEl?parseFloat(mcEl.value)||0:0;
  streamStreamer.setFilter({cats:allOn?[]:cats,minCertainty:mc>0?mc:undefined}).then(()=>loadDiagnostics());
}

// ── Connect to a server (streaming, out-of-core) ──────────────────────────
let streamGroup=null,streamStreamer=null,_streamOnMove=null,_streamTimer=null;
let _streamHoverPos=null;       // xyz of the last hovered streamed point (for the reticle)
let _streamFilterOff=new Set(); // category names toggled OFF in streaming filter
let _streamPalette=[];          // the connected manifest's palette (cat → color/count)
// Point the viewer at a sphereql-vis-server: fetch the manifest, build the scene
// chrome (globe + overlays + stats + a palette legend) with NO inline points,
// then stream point tiles by viewport via a TileStreamer. Returns the streamer.
// Browser-validated (network + render loop).
async function connectToServer(baseUrl,opts){
  opts=opts||{};
  disconnectServer();
  const source=new ServerSource(baseUrl,{cache:new TileCache(),decode:makeWorkerDecoder()});
  const manifest=await source.manifest();
  // Chrome via rebuild with zero inline points: the globe (surface_radius),
  // overlays (manifest.overlays — same Overlay shape), and stats panel populate;
  // the empty pointsMesh costs nothing. The palette legend replaces the
  // (empty) per-point legend.
  _streamFilterOff=new Set();_streamPalette=manifest.palette||[];
  rebuild({title:manifest.title,stats:manifest.stats,overlays:manifest.overlays||[],surface_radius:manifest.surface_radius||1,show_axes:false,points:[]});
  buildStreamLegend(_streamPalette);
  streamGroup=new THREE.Group();scene.add(streamGroup);scalables.push(streamGroup);streamGroup.scale.setScalar(curScale);
  const sink=tileMeshSink(streamGroup,_streamPalette,streamColorMaterial());
  streamStreamer=new TileStreamer(source,sink,opts);
  await streamStreamer.startWith(manifest);
  // Camera → viewport tile updates, throttled so a drag doesn't spam requests.
  const camToReq=()=>{const p=camera.position,m=Math.hypot(p.x,p.y,p.z)||1;let th=Math.atan2(p.y,p.x);if(th<0)th+=2*Math.PI;return{theta:th,phi:Math.acos(clamp(p.z/m,-1,1)),dist:m/Math.max(curScale,1e-6)};};
  let pend=false;
  _streamOnMove=()=>{if(pend)return;pend=true;_streamTimer=setTimeout(()=>{pend=false;_streamTimer=null;if(streamStreamer)streamStreamer.update(camToReq());},120);};
  controls.addEventListener("change",_streamOnMove);
  // Debugger controls: show + wire the tune (re-project) + filter (min-certainty)
  // rows, and load the diagnostics dashboard.
  const showRow=id=>{const el=document.getElementById(id);if(el)el.style.display="block";};
  showRow("tune-row");showRow("filter-row");
  const tune=document.getElementById("tune-proj");
  if(tune)tune.onchange=async()=>{
    try{const m=await source.reproject(tune.value);
      document.getElementById("hdr-pill").textContent=(m.stats&&m.stats.projection_kind)||"";
      streamStreamer.clear();streamStreamer.tiles=new Map();await streamStreamer.startWith(m);streamStreamer.update(camToReq());
      loadDiagnostics();
    }catch(err){console.warn("SphereQL: reproject failed",err);}
  };
  const mc=document.getElementById("mincert");
  if(mc)mc.oninput=()=>{const v=document.getElementById("mincert-val");if(v)v.textContent=(+mc.value).toFixed(2);applyStreamFilter();};
  streamStreamer.update(camToReq());
  loadDiagnostics();
  return streamStreamer;
}
// Tear down an active server stream (its tile group + camera listener).
function disconnectServer(){
  if(_streamOnMove&&controls.removeEventListener)controls.removeEventListener("change",_streamOnMove);
  _streamOnMove=null;
  if(_streamTimer){clearTimeout(_streamTimer);_streamTimer=null;}
  if(streamStreamer){streamStreamer.clear();streamStreamer=null;}
  if(streamGroup){scene.remove(streamGroup);disposeObject(streamGroup);const i=scalables.indexOf(streamGroup);if(i>=0)scalables.splice(i,1);streamGroup=null;}
}

// ── Server debugger UI: inspect · diagnostics · tune · filter ──────────────
// Tiny diverging bar sparkline of a raw embedding vector into the inspector
// canvas (+x blue, −x warm); empty/absent vector hides the panel.
function renderVectorSparkline(vec){
  const cv=document.getElementById("info-vector"),wrap=document.getElementById("info-vector-wrap");
  if(!wrap)return;
  if(!vec||!vec.length){wrap.style.display="none";return;}
  wrap.style.display="block";
  const ctx=cv&&cv.getContext&&cv.getContext("2d");if(!ctx)return;
  const W=cv.width,H=cv.height,mid=H/2;ctx.clearRect(0,0,W,H);
  let mx=1e-9;for(const v of vec)mx=Math.max(mx,Math.abs(v));
  const n=vec.length,bw=W/n;
  for(let i=0;i<n;i++){const v=vec[i],h=Math.abs(v)/mx*(mid-1);
    ctx.fillStyle=v>=0?"#5cc8ff":"#ff8a65";ctx.fillRect(i*bw,v>=0?mid-h:mid,Math.max(1,bw-0.3),Math.max(1,h));}
  ctx.strokeStyle="rgba(120,160,255,0.25)";ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(W,mid);ctx.stroke();
}
// Inspect a streamed point by global row: fetch its metadata + ANN neighbors
// from the server and fill the info panel (label, category, coords, raw-vector
// sparkline, clickable neighbors). The inline selectPoint path is untouched.
async function selectStreamRow(row){
  if(!streamStreamer)return;
  const src=streamStreamer.source,info=document.getElementById("info");
  let meta,nbrs=[];
  try{const ms=await src.pointMeta([row]);meta=ms&&ms[0];if(!meta)return;nbrs=await src.nearest({row},6);}
  catch(err){console.warn("SphereQL: inspect fetch failed",err);return;}
  document.getElementById("info-label").textContent=meta.label||("point #"+row);
  const tag=document.getElementById("info-cat"),col=catColor[meta.category]||"#5cc8ff";
  tag.textContent=meta.category||"";tag.style.color=col;tag.style.background=col+"18";
  const f=x=>isFinite(+x)?(+x).toFixed(4):"—";
  document.getElementById("info-coords").innerHTML=`<span>θ</span><b>${f(meta.theta)}</b><span>φ</span><b>${f(meta.phi)}</b><span>r</span><b>${f(meta.r)}</b><span>cert</span><b>${f(meta.certainty)}</b>`;
  renderVectorSparkline(meta.vector);
  const nb=document.getElementById("info-neighbors"),rows=nbrs.map(h=>h.row),lab={};
  try{for(const m of await src.pointMeta(rows))lab[m.row]=m;}catch(err){/* labels are best-effort */}
  nb.innerHTML=nbrs.map(h=>{const m=lab[h.row],c=m?(catColor[m.category]||"#5cc8ff"):"#5cc8ff";
    return `<div class="nb" data-row="${h.row}" style="background:${c}22;border-left:2px solid ${c}"><span>${escHtml(m?(m.label||("#"+h.row)):("#"+h.row))}</span><span class="dist">${(+h.similarity).toFixed(3)}</span></div>`;}).join("");
  nb.querySelectorAll(".nb").forEach(el=>el.addEventListener("click",()=>selectStreamRow(parseInt(el.dataset.row))));
  document.getElementById("info-nb-head").textContent="Nearest · cosine";
  info.classList.add("visible");
}
// Render the /diagnostics payload (EVR + warnings + histograms + outliers) into
// the Diag tab; outliers are clickable → inspect.
function renderDiagnostics(d){
  const el=document.getElementById("diag-content");if(!el)return;
  if(!d){el.innerHTML='<div class="muted">no diagnostics</div>';return;}
  const evr=clamp((d.evr||0)*100,0,100);
  const hist=h=>{const bins=(h&&h.bins)||[],mx=Math.max(1,...bins);
    return `<div class="histo">${bins.map(b=>`<i style="height:${(b/mx*100).toFixed(1)}%"></i>`).join("")}</div><div class="histo-cap"><span>${h&&isFinite(h.min)?h.min.toFixed(2):""}</span><span>${h&&isFinite(h.max)?h.max.toFixed(2):""}</span></div>`;};
  let html=`<div class="srow"><span>projection</span><span class="v hl">${escHtml(d.projection_kind||"?")}</span></div>`;
  html+=`<div class="srow"><span>${escHtml(d.evr_label||"EVR")}</span><span class="v">${evr.toFixed(1)}%</span></div><div class="bar"><i style="width:${evr.toFixed(1)}%"></i></div>`;
  html+=`<div class="srow"><span>points</span><span class="v">${(d.total_points||0).toLocaleString()}</span></div>`;
  for(const w of d.warnings||[])html+=`<div class="warn ${escHtml(w.severity||"info")}">${escHtml(w.message||"")}</div>`;
  html+=`<h3 style="margin:13px 0 4px">Certainty</h3>${hist(d.certainty)}`;
  html+=`<h3 style="margin:13px 0 4px">Intensity</h3>${hist(d.intensity)}`;
  if((d.outliers||[]).length){html+='<h3 style="margin:13px 0 5px">Low-certainty outliers</h3>';
    html+=(d.outliers||[]).map(o=>`<div class="nb" data-row="${o.row}"><span>${escHtml(o.label||("#"+o.row))}</span><span class="dist">${(+o.certainty).toFixed(3)}</span></div>`).join("");}
  el.innerHTML=html;
  el.querySelectorAll(".nb[data-row]").forEach(x=>x.addEventListener("click",()=>selectStreamRow(parseInt(x.dataset.row))));
}
async function loadDiagnostics(){
  if(!streamStreamer)return;
  try{renderDiagnostics(await streamStreamer.source.diagnostics());}catch(err){console.warn("SphereQL: diagnostics failed",err);}
}

// ── Boot + render loop ─────────────────────────────────────────────────────
// The baked offline file inlines the whole scene as `D` and renders all of it
// through InlineSource. `dataSource` stays module-visible so the studio (and a
// future server-backed streaming renderer) can query/stream through one seam.
let dataSource=new InlineSource(D);
rebuild(dataSource.scene);
applyViewHash(); // restore a shared camera/settings view, if the URL carries one
// Opt-in streaming mode: launch with `#server=<url>` to stream from a
// sphereql-vis-server instead of rendering the inline blob. Offline by default
// — with no such hash the viewer never touches the network.
(function(){
  if(typeof location==="undefined")return;
  const m=(location.hash||"").match(/[#&]server=([^&]+)/);
  if(!m)return;
  let url;try{url=decodeURIComponent(m[1]);}catch(err){return;}
  connectToServer(url).catch(err=>{console.warn("SphereQL: server connect failed",err);flashButton("open-scene","✗ server "+(err&&err.message||err),2600);});
})();
function animate(){
  requestAnimationFrame(animate);
  if(pendingTransform){applyTransform();pendingTransform=false;}
  updateHover(); // coalesced hover pick (≤1 GPU readback/frame)
  if(tgtTween){tgtTween.t++;const k=Math.min(1,tgtTween.t/tgtTween.dur),e=k*k*(3-2*k);controls.target.lerpVectors(tgtTween.from,tgtTween.to,e);if(k>=1)tgtTween=null;}
  controls.update();
  {const hp=streamStreamer?_streamHoverPos:(hoveredIdx>=0?curPos(hoveredIdx):null);
   if(hp){const sp=projectToScreen(hp);if(sp.vis){reticle.style.display="block";reticle.style.left=sp.x+"px";reticle.style.top=sp.y+"px";}else reticle.style.display="none";}else reticle.style.display="none";}
  if(selectedIdx>=0){const sp=projectToScreen(curPos(selectedIdx));if(sp.vis){sellabel.style.display="block";sellabel.style.left=sp.x+"px";sellabel.style.top=(sp.y-16)+"px";}else sellabel.style.display="none";}
  updateLabels();updatePinLabels();
  drawMinimap();renderer.render(scene,camera);
}
animate();

// ── Compare embedding (opt-in via #embed) ────────────────────────────────
// When the viewer is hosted in a compare iframe (its URL hash contains
// `embed`), it accepts a scene + camera over postMessage and broadcasts its
// own camera moves to the parent. The broadcast is epsilon-gated (not a bare
// flag) so OrbitControls damping — which re-emits `change` for several frames
// after an applied update — cannot start a feedback storm. Inert otherwise, so
// the baked viewer never posts messages.
(function(){
  if(typeof location==="undefined"||!/(^|[#&])embed/.test(location.hash||""))return;
  let lastSent=null,applying=false;
  const camState=()=>[camera.position.x,camera.position.y,camera.position.z,controls.target.x,controls.target.y,controls.target.z];
  const drift=(a,b)=>{if(!a||!b)return Infinity;let d=0;for(let i=0;i<6;i++)d=Math.max(d,Math.abs(a[i]-b[i]));return d;};
  const eps=()=>1e-3*Math.max(1,maxR*curScale);
  controls.addEventListener("change",()=>{
    if(applying)return;
    const s=camState();
    if(drift(s,lastSent)<eps())return; // within tolerance of the last broadcast → skip (kills the damping echo)
    lastSent=s;
    try{parent.postMessage({type:"sphereql-cam",s},"*");}catch(err){/* cross-origin parent */}
  });
  window.addEventListener("message",e=>{
    // Only the embedding parent (the compare host) may drive this pane — reject
    // sibling/other-window messages. Scene data is still parseScene-sanitized,
    // so this guards against camera/lock spoofing, not XSS.
    if(e.source!==parent)return;
    const m=e.data;if(!m||typeof m!=="object")return;
    if(m.type==="sphereql-scene"&&m.scene){try{rebuild(parseScene(m.scene));}catch(err){console.warn("SphereQL: bad injected scene",err);}}
    else if(m.type==="sphereql-cam"&&Array.isArray(m.s)&&m.s.length===6&&m.s.every(v=>isFinite(v))){
      applying=true;
      camera.position.set(m.s[0],m.s[1],m.s[2]);controls.target.set(m.s[3],m.s[4],m.s[5]);controls.update();
      lastSent=m.s.slice(); // baseline at the applied pose so our own `change` is within eps
      applying=false;
    }
    else if(m.type==="sphereql-lock"){ // independent orbit / zoom locks from the compare host
      controls.enableRotate=!m.lockRotate;
      controls.enableZoom=!m.lockZoom; // touch-pinch zoom
      zoomLocked=!!m.lockZoom;         // wheel zoom
    }
  });
  // Tell the compare host our listener is live, so it can (re)inject a scene
  // even if it built one before this iframe finished loading.
  try{parent.postMessage({type:"sphereql-embed-ready"},"*");}catch(err){/* no/again cross-origin parent */}
})();
