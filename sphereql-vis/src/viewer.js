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
const DEF={scale:12,radial:1,spread:1,size:3.5,globe:true,autorot:false,ui:1,palette:"aurora",zoom:0.5};
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
const cfgFile=document.getElementById("cfg-file");
const sceneFile=document.getElementById("scene-file");
const dropzone=document.getElementById("dropzone");
const mini=document.getElementById("mini"),mctx=mini.getContext("2d"),MW=mini.width,MH=mini.height;
const miniBase=document.createElement("canvas");miniBase.width=MW;miniBase.height=MH;const mbctx=miniBase.getContext("2d");

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
// Persistent tool layer: the great-circle ruler draws here. Scaled with the
// scene (added to `scalables` in rebuild); its content is scene-scoped and
// cleared on teardown.
const rulerGroup=new THREE.Group();scene.add(rulerGroup);
const rulerReadout=document.getElementById("ruler-readout");
const _pv=new THREE.Vector3(),_fwd=new THREE.Vector3(),_tmp=new THREE.Vector3();
const _zc=new THREE.Vector3(),_zd=new THREE.Vector3(),_zn=new THREE.Vector3();
const _av=new THREE.Vector3(),_cd=new THREE.Vector3();

// ── Per-scene state (reassigned by rebuild) ──────────────────────────────
let pts=[],N=0,overlays=[],SR=1.0,maxR=1,showAxes=false;
let catSet=[],catColor={},catVisible={},catCounts={},catDir={},catDirArr=[],posIndex=new Map();
let origPos=new Float32Array(0);
let pointsGeo=null,pointsMat=null,pointsMesh=null,globeGroup=null,linesGroup=null;
let overlayGroups={},overlayKinds=new Set(),bridgeLines=[],bridgesByPoint={};
let labelData=[],labelEls=[],labelKindOn={},labelKindsPresent=[],soloCat=null,labelRefDist=1;
let legendRows={};
let scalables=[];
let baseSize=DEF.size,curScale=DEF.scale,spreadF=DEF.spread,radialG=DEF.radial,uiScale=DEF.ui;
let selectedIdx=-1,hoveredIdx=-1;
let tgtTween=null,pendingTransform=false;
let rulerOn=false,rulerPicks=[],rulerLast=null;

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
// Combined angular (domain spread) + radial transform; always from origPos.
function applyTransform(){
  const pa=pointsGeo.getAttribute("position").array;
  for(let i=0;i<N;i++){
    const ox=origPos[i*3],oy=origPos[i*3+1],oz=origPos[i*3+2],mag=Math.hypot(ox,oy,oz);
    if(mag<1e-9){pa[i*3]=ox;pa[i*3+1]=oy;pa[i*3+2]=oz;continue;}
    let dx=ox/mag,dy=oy/mag,dz=oz/mag;
    if(spreadF!==1){const c=catDir[pts[i].cat];let dot=clamp(c[0]*dx+c[1]*dy+c[2]*dz,-1,1),om=Math.acos(dot);
      if(om>=1e-4){const s=Math.sin(om),w1=Math.sin((1-spreadF)*om)/s,w2=Math.sin(spreadF*om)/s;
        const nx=c[0]*w1+dx*w2,ny=c[1]*w1+dy*w2,nz=c[2]*w1+dz*w2,nm=Math.hypot(nx,ny,nz)||1;dx=nx/nm;dy=ny/nm;dz=nz/nm;}}
    const nmag=Math.max(0.02,SR+(mag-SR)*radialG);
    pa[i*3]=dx*nmag;pa[i*3+1]=dy*nmag;pa[i*3+2]=dz*nmag;
  }
  pointsGeo.getAttribute("position").needsUpdate=true;pointsGeo.computeBoundingSphere();
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
function getHovered(e){
  if(!pointsMesh)return -1;
  mouse.x=(e.clientX/innerWidth)*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;
  raycaster.setFromCamera(mouse,camera);
  const hits=raycaster.intersectObject(pointsMesh);
  if(hits.length>0){const idx=hits[0].index;if(catVisible[pts[idx].cat])return idx;}
  return -1;
}
function curPos(i){const a=pointsGeo.getAttribute("position").array;return[a[i*3],a[i*3+1],a[i*3+2]];}

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
function currentSettings(){return{scale:curScale,zoom_speed:controls.zoomSpeed,radial:radialG,domain_spread:spreadF,point_size:baseSize,ui_scale:uiScale,color_scheme:schemeSel.value,reference_globe:globeCb.checked,auto_rotate:autorot.checked};}
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
  const a=pointsGeo.getAttribute("position").array;
  for(let i=0;i<N;i++){if(!catVisible[pts[i].cat])continue;const[th,ph]=tpOf(a[i*3],a[i*3+1],a[i*3+2]);
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
  const f=worldUnderCursor(e.clientX,e.clientY);
  const s=Math.exp(Math.sign(e.deltaY)*Math.min(Math.abs(e.deltaY),120)/120*controls.zoomSpeed*0.2);
  camera.position.sub(f).multiplyScalar(s).add(f);
  controls.target.sub(f).multiplyScalar(s).add(f);
  const d=camera.position.distanceTo(controls.target),cd=clamp(d,controls.minDistance,controls.maxDistance);
  if(d!==cd)camera.position.copy(controls.target).addScaledVector(_tmp.copy(camera.position).sub(controls.target).normalize(),cd);
  controls.update();
},{capture:true,passive:false});
canvas.addEventListener("mousemove",e=>{
  const idx=getHovered(e);hoveredIdx=idx;
  if(idx>=0){const p=pts[idx];
    tooltip.innerHTML=`<div class="tt-lbl">${escHtml(p.label||"Point "+idx)}</div><div class="tt-meta">${escHtml(p.cat)} · θ ${p.theta.toFixed(2)}  φ ${p.phi.toFixed(2)}  r ${p.r.toFixed(2)}</div>`;
    tooltip.style.display="block";tooltip.style.left=(e.clientX+16)+"px";tooltip.style.top=(e.clientY+14)+"px";canvas.style.cursor="crosshair";
  }else{tooltip.style.display="none";canvas.style.cursor="grab";}
});
canvas.addEventListener("mouseleave",()=>{hoveredIdx=-1;tooltip.style.display="none";});
window.addEventListener("keydown",e=>{if(e.key!=="Escape")return;if(rulerOn&&rulerPicks.length){clearRuler();return;}if(selectedIdx>=0)deselectPoint(true);});
// Click detection via pointer down/up + movement threshold — the `click`
// event is unreliable while OrbitControls is handling pointer gestures.
let _downX=0,_downY=0;
canvas.addEventListener("pointerdown",e=>{_downX=e.clientX;_downY=e.clientY;});
canvas.addEventListener("pointerup",e=>{
  if(Math.hypot(e.clientX-_downX,e.clientY-_downY)>=5)return; // a drag, not a click
  const idx=getHovered(e);
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
function rulerMeasure(){
  const a=rulerPicks[0],b=rulerPicks[1];
  const om=Math.acos(clamp(a[0]*b[0]+a[1]*b[1]+a[2]*b[2],-1,1));
  const v=[],SEG=72;
  if(om<1e-4){
    // Coincident directions: draw a short visible segment (a 1-vertex line
    // renders nothing) rather than degenerate slerp.
    v.push(new THREE.Vector3(a[0]*SR,a[1]*SR,a[2]*SR),new THREE.Vector3(b[0]*SR,b[1]*SR,b[2]*SR));
  }else if(om>Math.PI-1e-3){
    // (Near-)antipodal: sin(om)→0 makes slerp explode (the arc would collapse
    // through the globe interior). The geodesic is ambiguous, so sweep a clean
    // great semicircle from `a` about an axis perpendicular to it.
    const h=Math.abs(a[0])<0.9?[1,0,0]:[0,1,0];
    const d=h[0]*a[0]+h[1]*a[1]+h[2]*a[2];
    let px=h[0]-d*a[0],py=h[1]-d*a[1],pz=h[2]-d*a[2];const pm=Math.hypot(px,py,pz)||1;px/=pm;py/=pm;pz/=pm;
    for(let i=0;i<=SEG;i++){const th=om*i/SEG,c=Math.cos(th),sn=Math.sin(th);
      v.push(new THREE.Vector3((a[0]*c+px*sn)*SR,(a[1]*c+py*sn)*SR,(a[2]*c+pz*sn)*SR));}
  }else{const s=Math.sin(om);for(let i=0;i<=SEG;i++){const t=i/SEG,w1=Math.sin((1-t)*om)/s,w2=Math.sin(t*om)/s;
    v.push(new THREE.Vector3((a[0]*w1+b[0]*w2)*SR,(a[1]*w1+b[1]*w2)*SR,(a[2]*w1+b[2]*w2)*SR));}}
  rulerGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(v),new THREE.LineBasicMaterial({color:0xffb454,transparent:true,opacity:0.95})));
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
  // surface_radius: honour an explicit finite value, else the upper-median
  // ‖xyz‖ — matching Rust `Scene::surface_radius_for` (norms[len/2]) so a
  // dropped scene lands on the same shell the producer would compute.
  let sr=+raw.surface_radius;
  if(!isFinite(sr)||sr<=0){const norms=out.map(p=>Math.hypot(p.x,p.y,p.z)).sort((a,b)=>a-b);sr=norms.length?norms[norms.length>>1]:1.0;if(!isFinite(sr)||sr<=0)sr=1.0;}
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
  pointsMesh=null;pointsGeo=null;pointsMat=null;globeGroup=null;linesGroup=null;
  legendDiv.innerHTML="";oi.innerHTML="";labelTogglesDiv.innerHTML="";labelsDiv.innerHTML="";
  const info=document.getElementById("info");if(info)info.classList.remove("visible");
  tooltip.style.display="none";reticle.style.display="none";sellabel.style.display="none";
  setRuler(false); // fully disarm the ruler (flag + button + picks) on scene swap
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
  catDir={};
  {const sum={},cnt={};catSet.forEach(c=>{sum[c]=[0,0,0];cnt[c]=0;});
   for(let i=0;i<N;i++){const p=pts[i],m=Math.hypot(p.x,p.y,p.z)||1;sum[p.cat][0]+=p.x/m;sum[p.cat][1]+=p.y/m;sum[p.cat][2]+=p.z/m;cnt[p.cat]++;}
   catSet.forEach(c=>{const s=sum[c],m=Math.hypot(s[0],s[1],s[2]);catDir[c]=m>1e-9?[s[0]/m,s[1]/m,s[2]/m]:[0,0,1];});}
  catDirArr=Object.values(catDir);
  pointsGeo=new THREE.BufferGeometry();
  pointsGeo.setAttribute("position",new THREE.BufferAttribute(positions,3));
  pointsGeo.setAttribute("color",new THREE.BufferAttribute(colors,3));
  pointsGeo.setAttribute("size",new THREE.BufferAttribute(sizes,1));
  pointsMat=new THREE.ShaderMaterial({vertexColors:true,transparent:true,depthWrite:false,uniforms:{opacity:{value:1.0}},
    vertexShader:`attribute float size;varying vec3 vc;void main(){vc=color;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=size*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
    fragmentShader:`uniform float opacity;varying vec3 vc;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;float a=smoothstep(0.5,0.44,d)*opacity;float core=smoothstep(0.32,0.0,d);vec3 col=mix(vc,vec3(1.0),core*0.4);gl_FragColor=vec4(col,a);}`});
  pointsMesh=new THREE.Points(pointsGeo,pointsMat);scene.add(pointsMesh);
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

  scalables=[pointsMesh,linesGroup,globeGroup,rulerGroup,...Object.values(overlayGroups)];

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
  searchInput.value="";

  // ── Init view (default scale + framing) ─────────────────────────────────
  applyScale(DEF.scale);drawMinimapBase();frameCamera();
}

// ── Boot + render loop ─────────────────────────────────────────────────────
rebuild(D);
applyViewHash(); // restore a shared camera/settings view, if the URL carries one
function animate(){
  requestAnimationFrame(animate);
  if(pendingTransform){applyTransform();pendingTransform=false;}
  if(tgtTween){tgtTween.t++;const k=Math.min(1,tgtTween.t/tgtTween.dur),e=k*k*(3-2*k);controls.target.lerpVectors(tgtTween.from,tgtTween.to,e);if(k>=1)tgtTween=null;}
  controls.update();
  if(hoveredIdx>=0){const sp=projectToScreen(curPos(hoveredIdx));if(sp.vis){reticle.style.display="block";reticle.style.left=sp.x+"px";reticle.style.top=sp.y+"px";}else reticle.style.display="none";}else reticle.style.display="none";
  if(selectedIdx>=0){const sp=projectToScreen(curPos(selectedIdx));if(sp.vis){sellabel.style.display="block";sellabel.style.left=sp.x+"px";sellabel.style.top=(sp.y-16)+"px";}else sellabel.style.display="none";}
  updateLabels();
  drawMinimap();renderer.render(scene,camera);
}
animate();
