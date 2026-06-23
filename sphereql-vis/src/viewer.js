// SphereQL viewer — createViewer(rootEl, opts) factory.
// Each call returns an independent { rebuild, updateScene, drawChain,
// highlightByIds, setMorphTarget, applyMorph, clearMorph, dispose, camera }
// with its own GPU/DOM state, so two callers can run side by side.
// DOM nodes are resolved via data-* attributes with id-based fallbacks, so the
// current template.html (which uses ids) works without changes.
// Two-minds = two iframes; the #embed postMessage protocol wires cameras.
// The auto-boot block at the bottom creates window.viewer and re-exposes the
// classic globals (rebuild, parseScene, …) for studio.js compatibility.

// ── Module-level shared constants ────────────────────────────────────────────
const escHtml=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const fmin=a=>a.reduce((m,v)=>v<m?v:m,Infinity);
const fmax=a=>a.reduce((m,v)=>v>m?v:m,-Infinity);
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
// strength ∈ [0,1] for a point, robust to raw (un-parsed) emitter JSON:
// explicit `strength` > `certainty` > `intensity` > 1.0 (fully vivid). Shared by
// parseScene and the in-place buffer paths so rebuild/updateScene never NaN when
// a point lacks the field (the Rust ScenePoint emits certainty/intensity, not
// strength, so a directly-passed scene would otherwise blank the cloud).
function deriveStrength(p){
  if(isFinite(+p.strength))return clamp(+p.strength,0,1);
  if(isFinite(+p.certainty))return clamp(+p.certainty,0,1);
  if(isFinite(+p.intensity))return clamp(+p.intensity,0,1);
  return 1.0;
}
const reduceMotion=matchMedia("(prefers-reduced-motion:reduce)").matches;
const DEF={scale:12,radial:1,spread:1,size:3.5,globe:true,autorot:false,palette:"aurora",zoom:0.5,density:false};
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

// Vertex shader transform shared by the points and pick materials.
// Declares per-point attributes + uniforms and defines sphTransform(origPos)
// → displayed position. KEEP IN SYNC WITH curPos() inside createViewer.
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

// ── parseScene ───────────────────────────────────────────────────────────────
// Normalize arbitrary JSON → the shape rebuild/updateScene expect. Accepts the
// full Scene (Scene::to_json shape), a bare points array, or minimal points
// (xyz or r/theta/phi). Adds strength ∈ [0,1] derived from certainty, then
// intensity, defaulting to 1.0 for points that carry neither. Throws with a
// human message when there is nothing to show.
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
    const certainty=isFinite(+p.certainty)?+p.certainty:null;
    const intensity=isFinite(+p.intensity)?+p.intensity:null;
    const q={x,y,z,r,theta,phi,cat:p.cat!=null?String(p.cat):"",label:p.label!=null?String(p.label):"",strength:deriveStrength(p)};
    if(p.id!=null)q.id=String(p.id);
    if(certainty!=null)q.certainty=certainty;
    if(intensity!=null)q.intensity=intensity;
    out.push(q);
  }
  if(out.length===0)throw new Error("no usable points (each needs finite x/y/z or r/theta/phi)");
  let sr=+raw.surface_radius;
  if(!isFinite(sr)||sr<=0){const norms=out.map(p=>Math.hypot(p.x,p.y,p.z)).filter(n=>isFinite(n)&&n>0).sort((a,b)=>a-b);sr=norms.length?norms[norms.length>>1]:1.0;if(!isFinite(sr)||sr<=0)sr=1.0;}
  const st=raw.stats&&typeof raw.stats==="object"?raw.stats:{};
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

// ── createViewer(rootEl, opts) ───────────────────────────────────────────────
function createViewer(rootEl,opts){
  opts=opts||{};
  const onSelectCb=typeof opts.onSelect==="function"?opts.onSelect:null;

  // Resolve a DOM node: prefers data-* attribute, falls back to #id, then null.
  const q=(attr,id)=>rootEl.querySelector("[data-"+attr+"]")||(id?rootEl.querySelector("#"+id):null);

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const canvas=q("canvas","c");
  const tooltip=q("tooltip","tooltip");
  const reticle=q("reticle","reticle");
  const sellabel=q("sellabel","sellabel");
  const legendDiv=q("legend","legend-items");
  const oi=q("overlays","overlay-items");
  const statsDiv=q("stats","stats-content");
  const labelsDiv=q("labels","labels");
  const labelTogglesDiv=q("label-toggles","label-toggles");

  // ── THREE setup ──────────────────────────────────────────────────────────
  let W=Math.max(rootEl.clientWidth||rootEl.offsetWidth||800,1);
  let H=Math.max(rootEl.clientHeight||rootEl.offsetHeight||600,1);

  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(W,H);
  const scene=new THREE.Scene();
  scene.background=new THREE.Color(0x06060f);
  const camera=new THREE.PerspectiveCamera(54,W/H,0.01,200000);
  const controls=new THREE.OrbitControls(camera,canvas);
  controls.enableDamping=true;controls.dampingFactor=0.07;
  controls.autoRotate=false;controls.autoRotateSpeed=0.35;controls.zoomSpeed=DEF.zoom;
  scene.add(new THREE.AmbientLight(0x44557a,2.1));
  const dl=new THREE.DirectionalLight(0xcfe2ff,0.55);dl.position.set(3,5,4);scene.add(dl);
  const pickRT=typeof THREE.WebGLRenderTarget==="function"?new THREE.WebGLRenderTarget(1,1):null;
  // Persistent groups: query geodesics (cleared by teardown) and reasoning
  // chains (cleared by teardown; also managed via drawChain handles).
  const queryGroup=new THREE.Group();scene.add(queryGroup);
  const chainGroup=new THREE.Group();scene.add(chainGroup);
  const _pv=new THREE.Vector3(),_fwd=new THREE.Vector3(),_tmp=new THREE.Vector3();
  const _zc=new THREE.Vector3(),_zd=new THREE.Vector3(),_zn=new THREE.Vector3();
  const _av=new THREE.Vector3(),_cd=new THREE.Vector3();

  // ── Per-scene state ──────────────────────────────────────────────────────
  let pts=[],N=0,overlays=[],SR=1.0,maxR=1,showAxes=false;
  let catSet=[],catColor={},catVisible={},catCounts={},catDir={},catDirArr=[],posIndex=new Map();
  let idToIndex=new Map();
  let morphTarget=null,morphT=0;
  let origPos=new Float32Array(0);
  let pointsGeo=null,pointsMat=null,pickMat=null,pointsMesh=null,globeGroup=null,linesGroup=null;
  let overlayGroups={},overlayKinds=new Set(),bridgeLines=[],bridgesByPoint={};
  let labelData=[],labelEls=[],labelKindOn={},labelKindsPresent=[],soloCat=null;
  let legendRows={};
  let scalables=[];
  let baseSize=DEF.size,curScale=DEF.scale,spreadF=DEF.spread,radialG=DEF.radial;
  let selectedIdx=-1,hoveredIdx=-1;
  let tgtTween=null,pendingTransform=false;
  let zoomLocked=false; // set by #embed compare host
  // drawChain animation callbacks: each returns true when the animation is done.
  const chainAnimations=[];

  // ── Core helpers ─────────────────────────────────────────────────────────
  function buildCatColor(name){const pal=PALETTES[name]||PALETTES.aurora;catSet.forEach((c,i)=>catColor[c]=pal[i%pal.length]);}
  function frameCamera(){const d=DEF.scale*maxR*2.6;camera.position.set(d*0.12,d*0.3,d);controls.target.set(0,0,0);controls.update();}
  function tweenTarget(to){tgtTween={from:controls.target.clone(),to:to.clone(),t:0,dur:reduceMotion?1:22};}

  function transformPos(p){
    const x=p[0],y=p[1],z=p[2],mag=Math.hypot(x,y,z);
    if(mag<1e-9)return[x,y,z];
    let dx=x/mag,dy=y/mag,dz=z/mag;
    if(spreadF!==1){
      let bc=null,bd=-2;for(const c of catDirArr){const dt=c[0]*dx+c[1]*dy+c[2]*dz;if(dt>bd){bd=dt;bc=c;}}
      const om=Math.acos(clamp(bd,-1,1));
      if(bc&&om>=1e-4){const s=Math.sin(om),w1=Math.sin((1-spreadF)*om)/s,w2=Math.sin(spreadF*om)/s;
        const nx=bc[0]*w1+dx*w2,ny=bc[1]*w1+dy*w2,nz=bc[2]*w1+dz*w2,nm=Math.hypot(nx,ny,nz)||1;dx=nx/nm;dy=ny/nm;dz=nz/nm;}
    }
    const nmag=Math.max(0.02,SR+(mag-SR)*radialG);
    return[dx*nmag,dy*nmag,dz*nmag];
  }

  function applyTransform(){
    if(!pointsMat)return;
    const u=pointsMat.uniforms;
    u.uSpread.value=spreadF;u.uRadial.value=radialG;u.uSR.value=SR;
    u.uMorphT.value=morphTarget?morphT:0;u.uHasMorph.value=morphTarget?1:0;
    for(const b of bridgeLines){const a=b.fromIndex>=0?curPos(b.fromIndex):transformPos(b.from),c=transformPos(b.to),pos=b.line.geometry.getAttribute("position");pos.setXYZ(0,a[0],a[1],a[2]);pos.setXYZ(1,c[0],c[1],c[2]);pos.needsUpdate=true;}
    if(selectedIdx>=0)deselectPoint();
  }

  function applySize(sz){
    baseSize=sz;if(!pointsGeo)return;
    const sa=pointsGeo.getAttribute("size").array;
    for(let i=0;i<N;i++)sa[i]=catVisible[pts[i].cat]?sz:0;
    pointsGeo.getAttribute("size").needsUpdate=true;
    if(selectedIdx>=0)deselectPoint();
  }

  function v3(a){return new THREE.Vector3(a[0],a[1],a[2]);}
  function capRing(dir,ha,col){const d=v3(dir).normalize(),rr=SR*Math.sin(ha),off=SR*Math.cos(ha);
    const ring=new THREE.Mesh(new THREE.RingGeometry(rr*0.99,rr,64),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.45,side:THREE.DoubleSide}));
    ring.position.copy(d.clone().multiplyScalar(off));ring.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),d);return ring;}
  function lineBetween(a,b,col,op){return new THREE.Line(new THREE.BufferGeometry().setFromPoints([v3(a),v3(b)]),new THREE.LineBasicMaterial({color:col,transparent:true,opacity:op}));}
  function marker(pos,col,rad){const m=new THREE.Mesh(new THREE.SphereGeometry(rad,14,14),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.6,roughness:0.35}));m.position.copy(v3(pos));return m;}
  function groupFor(k){if(!overlayGroups[k]){const g=new THREE.Group();overlayGroups[k]=g;scene.add(g);}return overlayGroups[k];}
  function applyScale(s){curScale=s;scalables.forEach(o=>o.scale.setScalar(s));}

  function projectToScreen(p){
    _pv.set(p[0]*curScale,p[1]*curScale,p[2]*curScale);
    camera.getWorldDirection(_fwd);
    const inFront=_tmp.copy(_pv).sub(camera.position).dot(_fwd)>0;
    _pv.project(camera);
    return{x:(_pv.x*0.5+0.5)*W,y:(-_pv.y*0.5+0.5)*H,vis:inFront&&_pv.z<=1};
  }

  // Zoom-to-cursor: keep the world point under the pointer fixed while dollying.
  function worldUnderCursor(mx,my){
    const rect=canvas.getBoundingClientRect();
    _zc.set((mx-rect.left)/rect.width*2-1,-(my-rect.top)/rect.height*2+1,0.5).unproject(camera);
    _zd.copy(_zc).sub(camera.position).normalize();
    camera.getWorldDirection(_zn);
    const denom=_zd.dot(_zn);
    if(Math.abs(denom)<1e-6)return controls.target.clone();
    const tt=(_zn.dot(controls.target)-_zn.dot(camera.position))/denom;
    return camera.position.clone().add(_zd.multiplyScalar(tt));
  }

  // 24-bit id encode/decode for GPU pick buffer.
  function pickEncode(i){const id=i+1;return[(id&255)/255,((id>>8)&255)/255,((id>>16)&255)/255];}
  function pickDecode(r,g,b){return((r&255)|((g&255)<<8)|((b&255)<<16))-1;}

  function pickGPU(e){
    if(!pickRT||!pickMat||!pointsMesh||!renderer.readRenderTargetPixels)return -2;
    const rect=canvas.getBoundingClientRect();
    const cx=(e.clientX-rect.left)|0,cy=(e.clientY-rect.top)|0;
    const prev=renderer.getRenderTarget?renderer.getRenderTarget():null,vis=[];
    const pcol=renderer.getClearColor?renderer.getClearColor(new THREE.Color()):null,palpha=renderer.getClearAlpha?renderer.getClearAlpha():1;
    scene.children.forEach(c=>{vis.push(c.visible);if(c!==pointsMesh)c.visible=false;});
    try{
      pointsMesh.material=pickMat;
      camera.setViewOffset(W,H,cx,cy,1,1);
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

  function pickCPU(e){
    let best=-1,bestD=14*14;
    for(let i=0;i<N;i++){if(!catVisible[pts[i].cat])continue;const sp=projectToScreen(curPos(i));if(!sp.vis)continue;
      const dx=sp.x-e.clientX,dy=sp.y-e.clientY,d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=i;}}
    return best;
  }

  function getHovered(e){
    if(!pointsMesh)return -1;
    let idx=-2;
    try{idx=pickGPU(e);}catch(err){idx=-2;}
    if(idx===-2)return pickCPU(e);
    if(idx>=0&&idx<N&&catVisible[pts[idx].cat])return idx;
    return -1;
  }

  // CPU mirror of the GLSL sphTransform — KEEP IN SYNC with VERTEX_TRANSFORM.
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

  // opts.skipTween: leave the camera where it is (used when re-applying the
  // current selection after an in-place updateScene tick). The onSelect
  // callback fires only when the selection actually changes, so re-applying
  // does not re-notify the host or recenter the view.
  function selectPoint(idx,opts){
    opts=opts||{};
    const changed=idx!==selectedIdx;
    selectedIdx=idx;hoveredIdx=-1;const P=curPos(idx);
    if(!opts.skipTween)tweenTarget(new THREE.Vector3(P[0]*curScale,P[1]*curScale,P[2]*curScale));
    if(changed&&onSelectCb&&pts[idx]&&pts[idx].id!=null)onSelectCb(String(pts[idx].id));
    const dists=pts.map((q,i)=>{if(i===idx||!catVisible[q.cat])return{i,d:Infinity};const c=curPos(i),dx=P[0]-c[0],dy=P[1]-c[1],dz=P[2]-c[2];return{i,d:Math.sqrt(dx*dx+dy*dy+dz*dz)};}).filter(d=>d.d<Infinity).sort((a,b)=>a.d-b.d).slice(0,5);
    const near=new Set([idx,...dists.map(d=>d.i)]);
    const sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
    for(let i=0;i<N;i++){const base=new THREE.Color(catColor[pts[i].cat]);
      if(near.has(i)){sa[i]=i===idx?baseSize*1.7:baseSize*1.4;ca[i*3]=base.r;ca[i*3+1]=base.g;ca[i*3+2]=base.b;}
      else{sa[i]=baseSize*0.5;ca[i*3]=base.r*0.28;ca[i*3+1]=base.g*0.28;ca[i*3+2]=base.b*0.28;}}
    pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;
    pointsMat.uniforms.opacity.value=0.4;
    while(linesGroup.children.length)linesGroup.remove(linesGroup.children[0]);
    const lm=new THREE.LineBasicMaterial({color:0x5cc8ff,transparent:true,opacity:0.5});
    for(const d of dists){const c=curPos(d.i);linesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(P[0],P[1],P[2]),new THREE.Vector3(c[0],c[1],c[2])]),lm));}
    const myBridges=bridgesByPoint[idx];
    if(myBridges)for(const br of myBridges){const c=transformPos(br.to);
      linesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(P[0],P[1],P[2]),new THREE.Vector3(c[0],c[1],c[2])]),new THREE.LineBasicMaterial({color:br.color,transparent:true,opacity:0.9})));}
    const p=pts[idx];
    if(sellabel){sellabel.innerHTML=`<span class="sl-dot" style="background:${catColor[p.cat]};color:${catColor[p.cat]}"></span>${escHtml(p.label||"Point "+idx)}`;sellabel.style.display="block";}
    const infoLabel=q("info-label","info-label");if(infoLabel)infoLabel.textContent=p.label||"Point "+idx;
    const infoTag=q("info-cat","info-cat");if(infoTag){infoTag.textContent=p.cat;infoTag.style.color=catColor[p.cat];infoTag.style.background=catColor[p.cat]+"18";}
    const infoCoords=q("info-coords","info-coords");
    if(infoCoords)infoCoords.innerHTML=`<span>θ</span><b>${p.theta.toFixed(4)}</b><span>φ</span><b>${p.phi.toFixed(4)}</b><span>r</span><b>${p.r.toFixed(4)}</b><span>str</span><b>${p.strength.toFixed(2)}</b>${myBridges?`<span>bridges</span><b>${myBridges.length}</b>`:""}`;
    const nb=q("info-neighbors","info-neighbors");
    if(nb){nb.innerHTML=dists.map(d=>{const dc=catColor[pts[d.i].cat];return`<div class="nb" data-idx="${d.i}" style="background:${dc}22;border-left:2px solid ${dc}"><span>${escHtml(pts[d.i].label||"Point "+d.i)}</span><span class="dist">${d.d.toFixed(3)}</span></div>`;}).join("");
      nb.querySelectorAll(".nb").forEach(el=>el.addEventListener("click",()=>selectPoint(parseInt(el.dataset.idx))));}
    const infoEl=q("info","info");if(infoEl)infoEl.classList.add("visible");
  }

  function deselectPoint(revertCam){
    selectedIdx=-1;if(sellabel)sellabel.style.display="none";
    if(!pointsGeo)return;
    const sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
    for(let i=0;i<N;i++){sa[i]=catVisible[pts[i].cat]?baseSize:0;const c=new THREE.Color(catColor[pts[i].cat]);ca[i*3]=c.r;ca[i*3+1]=c.g;ca[i*3+2]=c.b;}
    pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;
    pointsMat.uniforms.opacity.value=1.0;
    while(linesGroup.children.length)linesGroup.remove(linesGroup.children[0]);
    const infoEl=q("info","info");if(infoEl)infoEl.classList.remove("visible");
    if(revertCam)tweenTarget(new THREE.Vector3(0,0,0));
  }

  function setAll(v){catSet.forEach(c=>{catVisible[c]=v;if(legendRows[c])legendRows[c].classList.toggle("dim",!v);});updateVisibility();}
  function updateVisibility(){
    if(!pointsGeo)return;
    const sa=pointsGeo.getAttribute("size").array;
    for(let i=0;i<N;i++)sa[i]=catVisible[pts[i].cat]?baseSize:0;
    pointsGeo.getAttribute("size").needsUpdate=true;
    if(selectedIdx>=0)deselectPoint();
  }

  // ── Floating overlay labels ──────────────────────────────────────────────
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
      const facing=(_av.x*_cd.x+_av.y*_cd.y+_av.z*_cd.z)/al>-0.15;
      const sp=projectToScreen(ld.anchor);
      if(sp.vis&&facing){
        const dist=camera.position.distanceTo(_av),s=clamp((curScale*SR*2.6)/Math.max(dist,1e-3),0.5,2.2);
        el.style.display="flex";el.style.left=sp.x+"px";el.style.top=sp.y+"px";el.style.fontSize=(10.5*s).toFixed(1)+"px";
        el.classList.toggle("solo",ld.kind==="centroid"&&soloCat===ld.cat);
      }else el.style.display="none";
    }
  }

  function disposeObject(o){if(!o)return;o.traverse(c=>{if(c.geometry)c.geometry.dispose();if(c.material){const m=c.material;(Array.isArray(m)?m:[m]).forEach(x=>{if(x&&x.dispose)x.dispose();});}});}

  // ── Great-circle arc (shared by highlightByIds + drawChain) ─────────────
  // Robust slerp from direction a → b sampled at SR; handles coincident and
  // antipodal cases.
  function shellArc(a,b){
    const om=Math.acos(clamp(a[0]*b[0]+a[1]*b[1]+a[2]*b[2],-1,1));
    const v=[],SEG=72;
    if(om<1e-4){
      v.push(new THREE.Vector3(a[0]*SR,a[1]*SR,a[2]*SR),new THREE.Vector3(b[0]*SR,b[1]*SR,b[2]*SR));
    }else if(om>Math.PI-1e-3){
      const h=Math.abs(a[0])<0.9?[1,0,0]:[0,1,0];
      const dd=h[0]*a[0]+h[1]*a[1]+h[2]*a[2];
      let px=h[0]-dd*a[0],py=h[1]-dd*a[1],pz=h[2]-dd*a[2];const pm=Math.hypot(px,py,pz)||1;px/=pm;py/=pm;pz/=pm;
      for(let i=0;i<=SEG;i++){const th=om*i/SEG,c=Math.cos(th),sn=Math.sin(th);
        v.push(new THREE.Vector3((a[0]*c+px*sn)*SR,(a[1]*c+py*sn)*SR,(a[2]*c+pz*sn)*SR));}
    }else{const s=Math.sin(om);for(let i=0;i<=SEG;i++){const t=i/SEG,w1=Math.sin((1-t)*om)/s,w2=Math.sin(t*om)/s;
      v.push(new THREE.Vector3((a[0]*w1+b[0]*w2)*SR,(a[1]*w1+b[1]*w2)*SR,(a[2]*w1+b[2]*w2)*SR));}}
    return v;
  }

  // ── Semantic-query highlight ─────────────────────────────────────────────
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
    pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;
    pointsMat.uniforms.opacity.value=0.6;
    const a=curPos(order[0]),am=Math.hypot(a[0],a[1],a[2])||1,ad=[a[0]/am,a[1]/am,a[2]/am];
    const mat=new THREE.LineBasicMaterial({color:0xffd95c,transparent:true,opacity:0.7});
    for(let j=1;j<order.length;j++){const b=curPos(order[j]),bm=Math.hypot(b[0],b[1],b[2])||1,bd=[b[0]/bm,b[1]/bm,b[2]/bm];
      queryGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(shellArc(ad,bd)),mat));}
    return order.length;
  }

  // ── Compare / morph ──────────────────────────────────────────────────────
  function setMorphTarget(sceneB){
    morphTarget=null;
    if(!sceneB||!Array.isArray(sceneB.points))return 0;
    const map=new Map();
    for(const p of sceneB.points){if(p.id==null)continue;const x=+p.x,y=+p.y,z=+p.z;if(!isFinite(x)||!isFinite(y)||!isFinite(z))continue;const m=Math.hypot(x,y,z)||1;map.set(String(p.id),{d:[x/m,y/m,z/m],r:m});}
    morphTarget=map;
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
  function applyMorph(t){morphT=clamp(t,0,1);applyTransform();}
  function clearMorph(){morphTarget=null;morphT=0;if(pointsMat){pointsMat.uniforms.uMorphT.value=0;pointsMat.uniforms.uHasMorph.value=0;}}

  // ── Scene swap ───────────────────────────────────────────────────────────
  function teardown(){
    disposeObject(pointsMesh);if(pointsMesh)scene.remove(pointsMesh);
    disposeObject(globeGroup);if(globeGroup)scene.remove(globeGroup);
    disposeObject(linesGroup);if(linesGroup)scene.remove(linesGroup);
    for(const k in overlayGroups){disposeObject(overlayGroups[k]);scene.remove(overlayGroups[k]);}
    overlayGroups={};
    if(pickMat&&pickMat.dispose)pickMat.dispose();
    pointsMesh=null;pointsGeo=null;pointsMat=null;pickMat=null;globeGroup=null;linesGroup=null;
    while(chainGroup.children.length){const c=chainGroup.children[0];disposeObject(c);chainGroup.remove(c);}
    chainAnimations.length=0;
    clearQuery();clearMorph();
    if(legendDiv)legendDiv.innerHTML="";if(oi)oi.innerHTML="";
    if(labelTogglesDiv)labelTogglesDiv.innerHTML="";if(labelsDiv)labelsDiv.innerHTML="";
    const infoEl=q("info","info");if(infoEl)infoEl.classList.remove("visible");
    if(tooltip)tooltip.style.display="none";if(reticle)reticle.style.display="none";if(sellabel)sellabel.style.display="none";
    selectedIdx=-1;hoveredIdx=-1;tgtTween=null;pendingTransform=false;
  }

  // ── rebuild(sc) ──────────────────────────────────────────────────────────
  // Full scene swap from a parsed Scene object. Resets view settings to
  // defaults and re-frames the camera unless opts.preserveCamera is set.
  function rebuild(sc,_opts){
    const preserveCamera=_opts&&_opts.preserveCamera;
    teardown();
    pts=sc.points||[];N=pts.length;overlays=sc.overlays||[];SR=sc.surface_radius||1.0;showAxes=!!sc.show_axes;
    maxR=1;for(const p of pts){const m=Math.hypot(p.x,p.y,p.z);if(m>maxR)maxR=m;}

    baseSize=DEF.size;curScale=DEF.scale;spreadF=DEF.spread;radialG=DEF.radial;
    soloCat=null;selectedIdx=-1;hoveredIdx=-1;pendingTransform=false;tgtTween=null;

    catSet=[...new Set(pts.map(p=>p.cat))].sort();
    catColor={};buildCatColor(DEF.palette);
    catVisible={};catSet.forEach(c=>catVisible[c]=true);
    catCounts={};pts.forEach(p=>catCounts[p.cat]=(catCounts[p.cat]||0)+1);

    const emptyEl=q("empty","empty");if(emptyEl)emptyEl.style.display=N===0?"flex":"none";
    controls.minDistance=maxR*0.05;controls.maxDistance=maxR*100*8;

    // ── Reference globe ──────────────────────────────────────────────────
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

    // ── Points ───────────────────────────────────────────────────────────
    origPos=new Float32Array(N*3);
    const positions=new Float32Array(N*3),colors=new Float32Array(N*3),sizes=new Float32Array(N),strArr=new Float32Array(N);
    for(let i=0;i<N;i++){const p=pts[i],c=new THREE.Color(catColor[p.cat]);
      origPos[i*3]=p.x;origPos[i*3+1]=p.y;origPos[i*3+2]=p.z;
      positions[i*3]=p.x;positions[i*3+1]=p.y;positions[i*3+2]=p.z;
      colors[i*3]=c.r;colors[i*3+1]=c.g;colors[i*3+2]=c.b;
      sizes[i]=baseSize;strArr[i]=deriveStrength(p);}
    posIndex=new Map();for(let i=0;i<N;i++)posIndex.set(pts[i].x+"|"+pts[i].y+"|"+pts[i].z,i);
    idToIndex=new Map();for(let i=0;i<N;i++){if(pts[i].id!=null)idToIndex.set(String(pts[i].id),i);}

    catDir={};
    {const sum={};catSet.forEach(c=>{sum[c]=[0,0,0];});
     for(let i=0;i<N;i++){const p=pts[i],m=Math.hypot(p.x,p.y,p.z)||1;sum[p.cat][0]+=p.x/m;sum[p.cat][1]+=p.y/m;sum[p.cat][2]+=p.z/m;}
     catSet.forEach(c=>{const s=sum[c],m=Math.hypot(s[0],s[1],s[2]);catDir[c]=m>1e-9?[s[0]/m,s[1]/m,s[2]/m]:[0,0,1];});}
    catDirArr=Object.values(catDir);

    pointsGeo=new THREE.BufferGeometry();
    pointsGeo.setAttribute("position",new THREE.BufferAttribute(positions,3));
    pointsGeo.setAttribute("color",new THREE.BufferAttribute(colors,3));
    pointsGeo.setAttribute("size",new THREE.BufferAttribute(sizes,1));
    pointsGeo.setAttribute("aStrength",new THREE.BufferAttribute(strArr,1));

    // Per-point angular density for the density-shading heatmap toggle.
    {const GW=24,GH=12,bins=new Int32Array(GW*GH),binOf=new Int32Array(N),dens=new Float32Array(N);
     for(let i=0;i<N;i++){const p=pts[i],r=Math.hypot(p.x,p.y,p.z)||1;let th=Math.atan2(p.y,p.x);if(th<0)th+=2*Math.PI;const ph=Math.acos(clamp(p.z/r,-1,1));
       const gx=Math.min(GW-1,Math.floor(th/(2*Math.PI)*GW)),gy=Math.min(GH-1,Math.floor(ph/Math.PI*GH)),b=gy*GW+gx;binOf[i]=b;bins[b]++;}
     let maxBin=1;for(const v of bins)if(v>maxBin)maxBin=v;
     for(let i=0;i<N;i++)dens[i]=bins[binOf[i]]/maxBin;
     pointsGeo.setAttribute("density",new THREE.BufferAttribute(dens,1));}

    // Category-spread pivot + morph target attributes for the vertex shader.
    {const cd=new Float32Array(N*3),mdv=new Float32Array(N*3),mrv=new Float32Array(N),mhv=new Float32Array(N);
     for(let i=0;i<N;i++){const c=catDir[pts[i].cat]||[0,0,1];cd[i*3]=c[0];cd[i*3+1]=c[1];cd[i*3+2]=c[2];mdv[i*3+2]=1;mrv[i]=1;}
     pointsGeo.setAttribute("aCatDir",new THREE.BufferAttribute(cd,3));
     pointsGeo.setAttribute("aMorphDir",new THREE.BufferAttribute(mdv,3));
     pointsGeo.setAttribute("aMorphR",new THREE.BufferAttribute(mrv,1));
     pointsGeo.setAttribute("aMorphHas",new THREE.BufferAttribute(mhv,1));}

    // Per-point GPU pick id baked as RGB.
    {const pc=new Float32Array(N*3);for(let i=0;i<N;i++){const c=pickEncode(i);pc[i*3]=c[0];pc[i*3+1]=c[1];pc[i*3+2]=c[2];}
     pointsGeo.setAttribute("aPickColor",new THREE.BufferAttribute(pc,3));}

    // Points material — strength ∈ [0,1] drives both size (0.3+0.7*s) and
    // fragment opacity (max(0.15,s)), so vivid memories bloom and fading
    // ones shrink/dim before a tick removes them.
    pointsMat=new THREE.ShaderMaterial({vertexColors:true,transparent:true,depthWrite:false,
      uniforms:{opacity:{value:1.0},densityOn:{value:DEF.density?1:0},uSpread:{value:DEF.spread},uRadial:{value:DEF.radial},uSR:{value:SR},uMorphT:{value:0},uHasMorph:{value:0}},
      vertexShader:VERTEX_TRANSFORM+`attribute float size;attribute float density;attribute float aStrength;varying vec3 vc;varying float vd;varying float vStr;void main(){vc=color;vd=density;vStr=aStrength;vec3 tp=sphTransform(position);vec4 mv=modelViewMatrix*vec4(tp,1.0);gl_PointSize=size*(0.3+0.7*aStrength)*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
      fragmentShader:`uniform float opacity;uniform float densityOn;varying vec3 vc;varying float vd;varying float vStr;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;float a=smoothstep(0.5,0.44,d)*opacity*max(0.15,vStr);float core=smoothstep(0.32,0.0,d);vec3 col=mix(vc,vec3(1.0),core*0.4);if(densityOn>0.5){vec3 cold=vec3(0.25,0.5,1.0),hot=vec3(1.0,0.62,0.22);col=mix(cold,hot,vd);a*=mix(0.5,1.0,vd);}gl_FragColor=vec4(col,a);}`});
    // Pick material shares uniforms (so transforms stay consistent) but writes
    // per-point id-color and applies the same strength-based size as the main
    // material, keeping GPU pick targets aligned with what's drawn.
    pickMat=new THREE.ShaderMaterial({uniforms:pointsMat.uniforms,
      vertexShader:VERTEX_TRANSFORM+`attribute float size;attribute float aStrength;attribute vec3 aPickColor;varying vec3 vpick;void main(){vpick=aPickColor;vec3 tp=sphTransform(position);vec4 mv=modelViewMatrix*vec4(tp,1.0);gl_PointSize=size*(0.3+0.7*aStrength)*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
      fragmentShader:`varying vec3 vpick;void main(){gl_FragColor=vec4(vpick,1.0);}`});
    pointsMesh=new THREE.Points(pointsGeo,pointsMat);pointsMesh.frustumCulled=false;scene.add(pointsMesh);
    linesGroup=new THREE.Group();scene.add(linesGroup);

    // ── Overlays ─────────────────────────────────────────────────────────
    overlayGroups={};overlayKinds=new Set();bridgeLines=[];bridgesByPoint={};labelData=[];
    overlays.forEach(o=>{
      try{
        if(!o||typeof o!=="object"||typeof o.kind!=="string")return;
        overlayKinds.add(o.kind);const g=groupFor(o.kind);const col=o.color?new THREE.Color(o.color).getHex():0x5cc8ff;
        if(o.kind==="centroid"){g.add(marker(o.pos,col,SR*0.022));labelData.push({kind:"centroid",anchor:o.pos,text:o.label,color:o.color||"#5cc8ff",cat:o.label});}
        else if(o.kind==="bridge"){const ch=classColor[o.classification]!==undefined?classColor[o.classification]:col;const ln=lineBetween(o.from,o.to,ch,0.18+0.55*(o.strength||0.5));g.add(ln);const fi=posIndex.get(o.from[0]+"|"+o.from[1]+"|"+o.from[2]),fidx=fi===undefined?-1:fi;bridgeLines.push({line:ln,from:o.from,to:o.to,fromIndex:fidx,color:ch});if(fidx>=0)(bridgesByPoint[fidx]||(bridgesByPoint[fidx]=[])).push({to:o.to,color:ch});}
        else if(o.kind==="geodesic_path"){drawChainIntoGroup(groupFor(o.kind),o);}
        else if(o.kind==="voronoi_cap"){g.add(capRing(o.center,o.half_angle,col));}
        else if(o.kind==="antipode"){g.add(marker(o.centroid,col,SR*0.016));g.add(marker(o.antipode,col,SR*0.016));g.add(lineBetween(o.centroid,o.antipode,col,0.18));labelData.push({kind:"antipode",anchor:o.antipode,text:"⤬ "+o.label,color:o.color||"#5cc8ff"});}
        else if(o.kind==="coverage_void"){(o.caps||[]).forEach(c=>g.add(capRing(c.center,c.half_angle,0x5cc8ff)));(o.voids||[]).forEach(vp=>g.add(marker(vp,0x222a40,SR*0.006)));}
        else if(o.kind==="domain_group"){g.add(marker(o.centroid,col,SR*0.027));(o.members||[]).forEach(m=>g.add(lineBetween(o.centroid,m,col,0.28)));labelData.push({kind:"domain_group",anchor:o.centroid,text:o.label,color:o.color||"#5cc8ff"});}
        else if(o.kind==="glob"){const m=new THREE.Mesh(new THREE.SphereGeometry(o.radius,22,22),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.1,depthWrite:false}));m.position.copy(v3(o.center));g.add(m);}
        else if(o.kind==="manifold_slice"){const pl=new THREE.Mesh(new THREE.PlaneGeometry(SR,SR),new THREE.MeshBasicMaterial({color:0x6f8fc8,transparent:true,opacity:0.12,side:THREE.DoubleSide}));pl.position.copy(v3(o.center));pl.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),v3(o.normal).normalize());g.add(pl);}
      }catch(err){console.warn("SphereQL: skipping malformed overlay",o&&o.kind,err);}
    });
    overlayKinds.forEach(k=>{if(overlayDefaultOff.has(k))overlayGroups[k].visible=false;});

    scalables=[pointsMesh,linesGroup,globeGroup,queryGroup,chainGroup,...Object.values(overlayGroups)];

    // ── Legend ───────────────────────────────────────────────────────────
    legendRows={};
    if(legendDiv){
      catSet.forEach(cat=>{
        const row=document.createElement("div");row.className="lrow";
        row.innerHTML=`<span class="ldot" style="background:${catColor[cat]};color:${catColor[cat]}"></span><span class="lbl"></span><span class="lcnt">${catCounts[cat]}</span>`;
        row.querySelector(".lbl").textContent=cat;legendRows[cat]=row;
        row.addEventListener("click",()=>{catVisible[cat]=!catVisible[cat];row.classList.toggle("dim",!catVisible[cat]);updateVisibility();});
        legendDiv.appendChild(row);});}

    // ── Overlay toggles ──────────────────────────────────────────────────
    if(oi){if(overlayKinds.size>0){[...overlayKinds].sort().forEach(kind=>{
      const on=overlayGroups[kind].visible,row=document.createElement("label");row.className="orow";
      row.innerHTML=`<input type="checkbox" ${on?"checked":""}><span></span>`;
      row.querySelector("span").textContent=OVERLAY_LABELS[kind]||kind;
      row.querySelector("input").addEventListener("change",e=>{overlayGroups[kind].visible=e.target.checked;});
      oi.appendChild(row);});}
    else{oi.innerHTML='<div class="muted">No overlays in this scene.</div>';}}

    // ── Floating labels + per-kind toggles ───────────────────────────────
    labelEls=[];labelKindOn={};
    if(labelsDiv){labelData.forEach(ld=>{
      const el=document.createElement("div");el.className="vlabel";
      const dot=document.createElement("span");dot.className="vdot";dot.style.background=ld.color;dot.style.color=ld.color;
      const txt=document.createElement("span");txt.textContent=ld.text;
      el.appendChild(dot);el.appendChild(txt);
      el.title="click to focus"+(ld.kind==="centroid"?" · solo domain":"");
      el.addEventListener("click",ev=>{ev.stopPropagation();focusLabel(ld);});
      labelsDiv.appendChild(el);labelEls.push({el,ld});});}
    labelKindsPresent=[...new Set(labelData.map(l=>l.kind))].sort();
    if(labelTogglesDiv){
      if(labelKindsPresent.length===0)labelTogglesDiv.innerHTML='<div class="muted">No labelled overlays.</div>';
      labelKindsPresent.forEach(kind=>{
        labelKindOn[kind]=true;
        const row=document.createElement("label");row.className="orow";
        row.innerHTML=`<input type="checkbox" checked><span></span>`;
        row.querySelector("span").textContent=LABEL_KIND_NAMES[kind]||kind;
        row.querySelector("input").addEventListener("change",e=>{labelKindOn[kind]=e.target.checked;});
        labelTogglesDiv.appendChild(row);});}

    // ── Header + stats ───────────────────────────────────────────────────
    const st=sc.stats||{};
    const hdrSub=q("hdr-sub","hdr-sub");if(hdrSub)hdrSub.textContent=sc.title||"";
    const hdrPill=q("hdr-pill","hdr-pill");if(hdrPill)hdrPill.textContent=st.projection_kind||"—";
    if(statsDiv){
      let rows="";
      if(N>0){const rV=pts.map(p=>p.r),thV=pts.map(p=>p.theta),phV=pts.map(p=>p.phi),evr=st.evr||0;
        rows=`<div class="srow"><span>points</span><span class="v">${N.toLocaleString()}</span></div>
<div class="srow"><span>domains</span><span class="v">${catSet.length}</span></div>
<div class="srow"><span>projection</span><span class="v hl">${escHtml(st.projection_kind||"?")}</span></div>
<div class="srow"><span>${escHtml(st.evr_label||"explained variance")}</span><span class="v">${(evr*100).toFixed(1)}%</span></div>
<div class="bar"><i style="width:${clamp(evr*100,0,100).toFixed(1)}%"></i></div>
<div class="srow"><span>r</span><span class="v">${fmin(rV).toFixed(2)} – ${fmax(rV).toFixed(2)}</span></div>
<div class="srow"><span>θ</span><span class="v">${fmin(thV).toFixed(2)} – ${fmax(thV).toFixed(2)}</span></div>
<div class="srow"><span>φ</span><span class="v">${fmin(phV).toFixed(2)} – ${fmax(phV).toFixed(2)}</span></div>`;}
      else{rows=`<div class="srow"><span>points</span><span class="v">0</span></div>`;}
      if(st.sampled_from)rows+=`<div class="note">▴ sample of ${escHtml(st.sampled_from.toLocaleString())}</div>`;
      if(st.dropped_nonfinite)rows+=`<div class="note">▴ ${escHtml(st.dropped_nonfinite.toLocaleString())} non-finite dropped</div>`;
      statsDiv.innerHTML=rows;}

    if(!preserveCamera){applyScale(DEF.scale);frameCamera();}
    else{applyScale(curScale);} // scalables are new; re-apply whatever scale was in effect
  }

  // ── updateScene(sc) ──────────────────────────────────────────────────────
  // View-preserving scene update — the per-tick workhorse. If N and catSet are
  // unchanged, rewrites GPU buffers in-place (position, color, strength, catDir
  // pivots) and the id→index map — camera, spread, radial, scale, and any active
  // selection stay put. If structure changes (different N or categories), does a
  // full rebuild while preserving the entire view state (camera + transforms).
  //
  // Boundary: the fast path updates POINTS only. Overlays (centroids, bridges,
  // geodesic paths) are refreshed solely on rebuild(); the live reasoning chain
  // should be driven through drawChain(), whose {clear} handle lets the host
  // swap it without restarting every point. Rebuilding overlays per tick would
  // restart their draw-on animation each frame, so it is intentionally avoided.
  function updateScene(sc){
    clearMorph(); // stale morph attrs would desync from new positions
    if(!pointsGeo){rebuild(sc);return;}
    const cats2=[...new Set(sc.points.map(p=>p.cat!=null?String(p.cat):""))].sort();
    const sameStructure=sc.points.length===N&&cats2.join("\0")===catSet.join("\0");
    if(!sameStructure){
      // Snapshot the full view state before teardown clears it.
      const savedCam={pos:camera.position.clone(),tgt:controls.target.clone()};
      const savedView={scale:curScale,size:baseSize,spread:spreadF,radial:radialG,zoom:controls.zoomSpeed};
      rebuild(sc,{preserveCamera:true}); // resets DEF but does not call frameCamera
      // Restore view state — scalables are freshly created so applyScale is safe.
      applyScale(savedView.scale);
      baseSize=savedView.size;spreadF=savedView.spread;radialG=savedView.radial;
      controls.zoomSpeed=savedView.zoom;
      applySize(savedView.size);
      applyTransform();
      camera.position.copy(savedCam.pos);controls.target.copy(savedCam.tgt);controls.update();
      return;
    }
    // Fast path: same N, same cats — update buffers only, no teardown.
    // NOTE: overlays are NOT touched here — the contract routes the reasoning
    // chain through drawChain() (stable {clear} handle) and refreshes static
    // overlays only on rebuild(). See the header comment.
    pts=sc.points;
    const posAttr=pointsGeo.getAttribute("position");
    const colAttr=pointsGeo.getAttribute("color");
    const strAttr=pointsGeo.getAttribute("aStrength");
    const cdAttr=pointsGeo.getAttribute("aCatDir");
    // Recompute catDir centroids since point positions may have shifted.
    const sum={};catSet.forEach(c=>{sum[c]=[0,0,0];});
    for(let i=0;i<N;i++){const p=pts[i],m=Math.hypot(p.x,p.y,p.z)||1;sum[p.cat][0]+=p.x/m;sum[p.cat][1]+=p.y/m;sum[p.cat][2]+=p.z/m;}
    catSet.forEach(c=>{const s=sum[c],m=Math.hypot(s[0],s[1],s[2]);catDir[c]=m>1e-9?[s[0]/m,s[1]/m,s[2]/m]:[0,0,1];});
    catDirArr=Object.values(catDir);
    // Rebuild the id→index map: slots recycle on forget, so the concept at
    // buffer slot i may have changed. highlightByIds keys on this map.
    idToIndex=new Map();
    for(let i=0;i<N;i++){
      const p=pts[i];
      if(p.id!=null)idToIndex.set(String(p.id),i);
      origPos[i*3]=p.x;origPos[i*3+1]=p.y;origPos[i*3+2]=p.z;
      posAttr.setXYZ(i,p.x,p.y,p.z);
      const c=new THREE.Color(catColor[p.cat]);colAttr.setXYZ(i,c.r,c.g,c.b);
      strAttr.setX(i,deriveStrength(p));
      const cd=catDir[p.cat]||[0,0,1];cdAttr.setXYZ(i,cd[0],cd[1],cd[2]);
    }
    posAttr.needsUpdate=true;colAttr.needsUpdate=true;strAttr.needsUpdate=true;cdAttr.needsUpdate=true;
    // Re-apply the active selection in place: recompute neighbor lines/highlight
    // against the new positions, but keep the camera and don't re-notify the host.
    if(selectedIdx>=0)selectPoint(selectedIdx,{skipTween:true});
  }

  // ── drawChain(chain) ─────────────────────────────────────────────────────
  // Render a reasoning chain as an animated brighter line with ordered hop
  // markers and billboarded relation labels. Accepts a geodesic_path overlay
  // straight from the emitter.
  // chain: {
  //   vertices: [[x,y,z], …]   densely-sampled (slerp) arc positions
  //   color:    "#rrggbb" | 0xrrggbb   default bright gold
  //   nodes:    [{id,pos,rel}, …]   waypoints; pos → sphere marker, and
  //                                 nodes[k].rel is the relation INTO node k
  //                                 (null at the start), shown as a billboarded
  //                                 -[rel]-> tag at the midpoint of the segment.
  //   // legacy fallback (if no nodes): hops:[[x,y,z],…], edges:[{label}|str,…]
  // }
  // Returns { clear } — call clear() to remove the chain from the scene.
  function drawChain(chain){
    chain=chain||{};
    // Normalize to a numeric hex: the emitter sends color as a CSS string
    // ("#ffcc44"), but makeTextSprite needs an integer for its bitwise mask.
    const col=chain.color!=null?new THREE.Color(chain.color).getHex():0xffd95c;
    const verts=(chain.vertices||[]).map(v=>new THREE.Vector3(v[0],v[1],v[2]));
    // Accept nodes:[{id,pos,rel},...] (emitter format) or legacy hops/edges.
    // nodes[k].rel = the relation INTO node k from the preceding hop;
    // so edge[i] labels the arc hops[i]→hops[i+1] and equals nodes[i+1].rel.
    const rawNodes=chain.nodes||[];
    const hops=rawNodes.length?rawNodes.map(n=>new THREE.Vector3(n.pos[0],n.pos[1],n.pos[2])):(chain.hops||[]).map(v=>new THREE.Vector3(v[0],v[1],v[2]));
    const edges=rawNodes.length?rawNodes.slice(1).map(n=>n.rel?{label:n.rel}:null):(chain.edges||[]);
    if(verts.length<2)return{clear:()=>{}};

    const grp=new THREE.Group();
    grp.scale.setScalar(curScale);
    chainGroup.add(grp);

    // Animated line draw-on via drawRange.
    const geo=new THREE.BufferGeometry().setFromPoints(verts);
    const lineMat=new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.95});
    grp.add(new THREE.Line(geo,lineMat));
    // Softer glow pass layered beneath (reuses same geometry; disposed with grp).
    grp.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.22})));

    // Hop markers — vivid spheres at each semantic waypoint.
    for(const hp of hops){
      grp.add(marker([hp.x,hp.y,hp.z],col,SR*0.022));
    }

    // Billboarded relation labels between consecutive hops.
    for(let i=0;i<edges.length&&i<hops.length-1;i++){
      const raw=edges[i];
      const text=typeof raw==="string"?raw:(raw&&raw.label?"-["+raw.label+"]->":(null));
      if(!text)continue;
      const mid=new THREE.Vector3().addVectors(hops[i],hops[i+1]).multiplyScalar(0.5);
      const sprite=makeTextSprite(text,col);
      sprite.position.copy(mid);
      grp.add(sprite);
    }

    // Animate the arc drawing on over ~45 frames; reduced-motion skips it.
    geo.setDrawRange(0,reduceMotion?verts.length:0);
    let drawn=0;
    const total=verts.length;
    const speed=Math.max(2,Math.ceil(total/45));
    if(!reduceMotion){
      chainAnimations.push(()=>{
        drawn=Math.min(drawn+speed,total);
        geo.setDrawRange(0,drawn);
        return drawn>=total;
      });
    }

    return{clear(){chainGroup.remove(grp);disposeObject(grp);}};
  }

  // Internal variant used by rebuild() for geodesic_path overlays. Renders
  // into an existing group (already managed by overlayGroups) rather than
  // chainGroup, so overlay visibility toggles work.
  function drawChainIntoGroup(grp,o){
    const col=o.color?new THREE.Color(o.color).getHex():0x5cc8ff;
    const verts=(o.vertices||[]).map(v3);
    const rawNodes=o.nodes||[];
    const hops=rawNodes.length?rawNodes.map(n=>v3(n.pos)):(o.hops||[]).map(v3);
    const edges=rawNodes.length?rawNodes.slice(1).map(n=>n.rel?{label:n.rel}:null):(o.edges||[]);
    if(verts.length<2){grp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(verts.length?verts:[new THREE.Vector3()]),new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.9})));return;}
    const geo=new THREE.BufferGeometry().setFromPoints(verts);
    grp.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.95})));
    grp.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.22})));
    for(const hp of hops)grp.add(marker([hp.x,hp.y,hp.z],col,SR*0.022));
    for(let i=0;i<edges.length&&i<hops.length-1;i++){
      const raw=edges[i];const text=typeof raw==="string"?raw:(raw&&raw.label?"-["+raw.label+"]->":(null));
      if(!text)continue;
      const mid=new THREE.Vector3().addVectors(hops[i],hops[i+1]).multiplyScalar(0.5);
      const sprite=makeTextSprite(text,col);sprite.position.copy(mid);grp.add(sprite);
    }
    // Overlay-group chains animate on scene load.
    geo.setDrawRange(0,reduceMotion?verts.length:0);
    let drawn=0;const total=verts.length,speed=Math.max(2,Math.ceil(total/45));
    if(!reduceMotion)chainAnimations.push(()=>{drawn=Math.min(drawn+speed,total);geo.setDrawRange(0,drawn);return drawn>=total;});
  }

  // Canvas-texture sprite for billboarded relation labels.
  function makeTextSprite(text,colorHex){
    const cv=document.createElement("canvas");cv.width=320;cv.height=64;
    const ctx=cv.getContext("2d");
    ctx.clearRect(0,0,320,64);
    ctx.fillStyle="rgba(6,6,15,0.82)";
    if(ctx.roundRect){ctx.roundRect(4,10,312,44,8);ctx.fill();}else{ctx.fillRect(4,10,312,44);}
    const c="#"+((colorHex||0x5cc8ff)&0xffffff).toString(16).padStart(6,"0");
    ctx.strokeStyle=c;ctx.lineWidth=1.5;
    if(ctx.roundRect)ctx.stroke();
    ctx.fillStyle=c;ctx.font="bold 22px monospace";ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText(text.length>28?text.slice(0,27)+"…":text,160,32);
    const tex=new THREE.CanvasTexture(cv);
    const mat=new THREE.SpriteMaterial({map:tex,transparent:true,opacity:0.92});
    const sprite=new THREE.Sprite(mat);
    sprite.scale.set(SR*0.45,SR*0.11,1);
    return sprite;
  }

  // ── Hover ────────────────────────────────────────────────────────────────
  let _hoverEv=null;
  function updateHover(){
    if(!_hoverEv)return;const e=_hoverEv;_hoverEv=null;
    const idx=getHovered(e);hoveredIdx=idx;
    if(idx>=0){const p=pts[idx];
      if(tooltip){tooltip.innerHTML=`<div class="tt-lbl">${escHtml(p.label||"Point "+idx)}</div><div class="tt-meta">${escHtml(p.cat)} · θ ${p.theta.toFixed(2)}  φ ${p.phi.toFixed(2)}  r ${p.r.toFixed(2)}  str ${p.strength.toFixed(2)}</div>`;
        tooltip.style.display="block";tooltip.style.left=(e.clientX+16)+"px";tooltip.style.top=(e.clientY+14)+"px";}
      canvas.style.cursor="crosshair";
    }else{if(tooltip)tooltip.style.display="none";canvas.style.cursor="grab";}
  }

  // ── Restore camera from URL hash (offline path only) ────────────────────
  function applyViewHash(){
    if(typeof location==="undefined"||!location.hash)return;
    const m=location.hash.match(/[#&]v=([^&]+)/);if(!m)return;
    let state;
    try{state=JSON.parse(decodeURIComponent(atob(m[1])));}catch(err){return;}
    if(!state||typeof state!=="object")return;
    if(Array.isArray(state.cam)&&state.cam.length===6&&state.cam.every(v=>isFinite(+v))){
      camera.position.set(+state.cam[0],+state.cam[1],+state.cam[2]);
      controls.target.set(+state.cam[3],+state.cam[4],+state.cam[5]);
      controls.update();
    }
  }

  // ── Event listeners ──────────────────────────────────────────────────────
  const ac=new AbortController();
  const sig={signal:ac.signal};

  // Zoom to cursor.
  canvas.addEventListener("wheel",e=>{
    e.preventDefault();e.stopImmediatePropagation();
    if(zoomLocked)return;
    const f=worldUnderCursor(e.clientX,e.clientY);
    const s=Math.exp(Math.sign(e.deltaY)*Math.min(Math.abs(e.deltaY),120)/120*controls.zoomSpeed*0.2);
    camera.position.sub(f).multiplyScalar(s).add(f);
    controls.target.sub(f).multiplyScalar(s).add(f);
    const d=camera.position.distanceTo(controls.target),cd=clamp(d,controls.minDistance,controls.maxDistance);
    if(d!==cd)camera.position.copy(controls.target).addScaledVector(_tmp.copy(camera.position).sub(controls.target).normalize(),cd);
    controls.update();
  },{capture:true,passive:false,...sig});

  canvas.addEventListener("mousemove",e=>{_hoverEv=e;},sig);
  canvas.addEventListener("mouseleave",()=>{_hoverEv=null;hoveredIdx=-1;if(tooltip)tooltip.style.display="none";},sig);
  window.addEventListener("keydown",e=>{if(e.key==="Escape"&&selectedIdx>=0)deselectPoint(true);},sig);

  let _downX=0,_downY=0;
  canvas.addEventListener("pointerdown",e=>{_downX=e.clientX;_downY=e.clientY;},sig);
  canvas.addEventListener("pointerup",e=>{
    if(Math.hypot(e.clientX-_downX,e.clientY-_downY)>=5)return;
    const idx=getHovered(e);
    if(idx>=0)selectPoint(idx);else deselectPoint(true);
  },sig);

  window.addEventListener("resize",()=>{
    W=Math.max(rootEl.clientWidth||rootEl.offsetWidth,1);
    H=Math.max(rootEl.clientHeight||rootEl.offsetHeight,1);
    camera.aspect=W/H;camera.updateProjectionMatrix();renderer.setSize(W,H);
  },sig);

  const ro=typeof ResizeObserver!=="undefined"?new ResizeObserver(()=>{
    const w=Math.max(rootEl.clientWidth||rootEl.offsetWidth,1);
    const h=Math.max(rootEl.clientHeight||rootEl.offsetHeight,1);
    if(w===W&&h===H)return;W=w;H=h;
    camera.aspect=W/H;camera.updateProjectionMatrix();renderer.setSize(W,H);
  }):null;
  if(ro)ro.observe(rootEl);

  // Legend select-all / select-none buttons (if present in rootEl).
  (function(){
    const sa=rootEl.querySelector("#sel-all,[data-sel-all]");
    const sn=rootEl.querySelector("#sel-none,[data-sel-none]");
    if(sa)sa.addEventListener("click",()=>setAll(true),sig);
    if(sn)sn.addEventListener("click",()=>setAll(false),sig);
  })();

  // ── Animation loop ───────────────────────────────────────────────────────
  let aniRunning=true,aniHandle=0;
  function animate(){
    if(!aniRunning)return;
    aniHandle=requestAnimationFrame(animate);
    if(pendingTransform){applyTransform();pendingTransform=false;}
    updateHover();
    if(tgtTween){tgtTween.t++;const k=Math.min(1,tgtTween.t/tgtTween.dur),e=k*k*(3-2*k);controls.target.lerpVectors(tgtTween.from,tgtTween.to,e);if(k>=1)tgtTween=null;}
    controls.update();
    // Advance chain draw-on animations; remove finished ones.
    for(let i=chainAnimations.length-1;i>=0;i--){if(chainAnimations[i]())chainAnimations.splice(i,1);}
    // Hover reticle.
    const hp=hoveredIdx>=0?curPos(hoveredIdx):null;
    if(hp){const sp=projectToScreen(hp);if(sp.vis){if(reticle){reticle.style.display="block";reticle.style.left=sp.x+"px";reticle.style.top=sp.y+"px";}}else if(reticle)reticle.style.display="none";}
    else if(reticle)reticle.style.display="none";
    // Selected-point floating label.
    if(selectedIdx>=0&&sellabel){const sp=projectToScreen(curPos(selectedIdx));if(sp.vis){sellabel.style.display="block";sellabel.style.left=sp.x+"px";sellabel.style.top=(sp.y-16)+"px";}else sellabel.style.display="none";}
    updateLabels();
    renderer.render(scene,camera);
  }
  animate();

  // ── dispose ──────────────────────────────────────────────────────────────
  function dispose(){
    aniRunning=false;cancelAnimationFrame(aniHandle);
    if(ro)ro.disconnect();
    ac.abort(); // removes all our addEventListener(...,{signal}) handlers
    if(controls&&controls.dispose)controls.dispose(); // OrbitControls' own canvas listeners
    teardown();
    if(pickRT)pickRT.dispose();
    renderer.dispose();
  }

  // ── #embed compare-mode ──────────────────────────────────────────────────
  // When hosted in a compare iframe (hash contains `embed`), accept a scene +
  // camera over postMessage and broadcast camera moves to the parent.
  // Epsilon-gated so OrbitControls damping can't start a feedback storm.
  (function(){
    if(typeof location==="undefined"||!/(^|[#&])embed/.test(location.hash||""))return;
    let lastSent=null,applying=false;
    const camState=()=>[camera.position.x,camera.position.y,camera.position.z,controls.target.x,controls.target.y,controls.target.z];
    const drift=(a,b)=>{if(!a||!b)return Infinity;let d=0;for(let i=0;i<6;i++)d=Math.max(d,Math.abs(a[i]-b[i]));return d;};
    const eps=()=>1e-3*Math.max(1,maxR*curScale);
    controls.addEventListener("change",()=>{
      if(applying)return;
      const s=camState();
      if(drift(s,lastSent)<eps())return;
      lastSent=s;
      try{parent.postMessage({type:"sphereql-cam",s},"*");}catch(err){}
    });
    window.addEventListener("message",e=>{
      if(e.source!==parent)return;
      const m=e.data;if(!m||typeof m!=="object")return;
      if(m.type==="sphereql-scene"&&m.scene){try{rebuild(parseScene(m.scene));}catch(err){console.warn("SphereQL: bad injected scene",err);}}
      else if(m.type==="sphereql-cam"&&Array.isArray(m.s)&&m.s.length===6&&m.s.every(v=>isFinite(v))){
        applying=true;
        camera.position.set(m.s[0],m.s[1],m.s[2]);controls.target.set(m.s[3],m.s[4],m.s[5]);controls.update();
        lastSent=m.s.slice();
        applying=false;
      }
      else if(m.type==="sphereql-lock"){
        controls.enableRotate=!m.lockRotate;
        controls.enableZoom=!m.lockZoom;
        zoomLocked=!!m.lockZoom;
      }
    });
    try{parent.postMessage({type:"sphereql-embed-ready"},"*");}catch(err){}
  })();

  return{rebuild,updateScene,drawChain,highlightByIds,setMorphTarget,applyMorph,clearMorph,dispose,camera,applyViewHash};
}

// ── Auto-boot ────────────────────────────────────────────────────────────────
// When viewer.js is inlined into the baked HTML page (template.html), D is the
// scene payload and document.body is the rootEl. Two-minds = two iframes, each
// with its own boot block + globals; the #embed protocol wires their cameras.
// Expose the classic globals so studio.js works unchanged.
if(typeof document!=="undefined"&&typeof D!=="undefined"){
  const viewer=createViewer(document.body);
  window.viewer=viewer;
  window.rebuild=sc=>viewer.rebuild(sc);
  window.parseScene=parseScene;
  window.highlightByIds=ids=>viewer.highlightByIds(ids);
  window.setMorphTarget=sc=>viewer.setMorphTarget(sc);
  window.applyMorph=t=>viewer.applyMorph(t);
  window.clearMorph=()=>viewer.clearMorph();
  viewer.rebuild(D);
  viewer.applyViewHash();
}
