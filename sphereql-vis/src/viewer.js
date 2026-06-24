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
// Pick id ↔ rgb, shared by the inline path (createViewer's pickEncode/pickDecode,
// where id = index+1) and the streaming tile sink (where id = global row + 1).
// Kept module-level so the dependency-injected tileMeshSink needs no factory state.
function pickEncodeId(id){const v=(id+1)|0;return[(v&255)/255,((v>>8)&255)/255,((v>>16)&255)/255];}
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
const DEF={scale:12,radial:1,spread:1,size:3.5,globe:true,autorot:false,palette:"aurora",zoom:0.5,density:false,ui:1};
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
    if(om<1e-4){}
    else if(om>3.141592653589793-1e-4){
      // Antipodal to the category pivot: the slerp's 1/sin(om) blows up, so
      // rotate d toward a deterministic helper axis by (1-uSpread)*om instead.
      // Same construction as the morph branch above, so this GPU path and the
      // curPos/transformPos CPU mirrors place the point identically (pick==draw).
      vec3 h=abs(d.x)<0.9?vec3(1.0,0.0,0.0):vec3(0.0,1.0,0.0);
      float hd=dot(h,d);vec3 pp=normalize(h-hd*d);
      float th=(1.0-uSpread)*om;d=d*cos(th)+pp*sin(th);
    }
    else{float s=sin(om);float w1=sin((1.0-uSpread)*om)/s;float w2=sin(uSpread*om)/s;d=normalize(aCatDir*w1+d*w2);}
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

// ── DataSource seam + SQT1 tile primitives ──────────────────────────────────
// The out-of-core streaming layer. Everything here is pure / duck-typed and
// DOM-free, so it lives at module scope (like escHtml/clamp/PALETTES above) and
// is unit-testable without a viewer instance. createViewer (Prompt 22) wires a
// concrete DataSource — InlineSource for the offline blob, ServerSource for the
// streaming server — into its render path; this PR adds only the primitives.

// `tiles(params)` params {theta,phi,half_angle,budget,lod} describe a viewport
// cone + detail budget; InlineSource has the whole cloud so it ignores the cone
// and only honours `budget` (the same stratified decimation the server uses).

// Decode a binary SQT1 tile — the wire form emitted by sphereql-vis tile.rs:
//   header 16B: magic "SQT1" · version u16 · flags u16 · count u32 · reserved u32
//   record 20B: x f32 · y f32 · z f32 · cat u16 · _pad u16 · row u32   (all LE)
// Accepts an ArrayBuffer or a Uint8Array; throws on a malformed/short buffer
// (the throws mirror Rust's TileError: TooShort / BadMagic / UnsupportedVersion
// / LengthMismatch — see suite 08).
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
function tileQuery(p){p=p||{};const q=[];for(const k of["theta","phi","half_angle","budget","lod"])if(p[k]!=null&&isFinite(+p[k]))q.push(k+"="+(+p[k]));if(p.cats!=null)q.push("cats="+encodeURIComponent(String(p.cats)));if(p.min_certainty!=null&&isFinite(+p.min_certainty))q.push("min_certainty="+(+p.min_certainty));return q.join("&");}
// Allow-list a palette color string from an untrusted server manifest before it
// reaches the DOM/CSS: hex, rgb(a), hsl(a), or a bare CSS color keyword. Anything
// else falls back to a neutral grey. Used for both the legend style="" and the
// THREE.Color in the streaming tile sink (Prompt 22/23).
function safeColor(c){return typeof c==="string"&&/^#[0-9a-fA-F]{3,8}$|^rgba?\([\d.,\s%]+\)$|^hsla?\([\d.,\s%]+\)$|^[a-zA-Z]+$/.test(c.trim())?c.trim():"#90a4ae";}

// InlineSource — the offline blob. Renders all of `D`; serves the streaming
// DataSource interface from the in-memory scene byte-identically to the offline
// render path. nearest() here is a *positional* cosine (the inline file has no
// raw embeddings) — a local stand-in for the server's ANN over the original
// vectors.
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
// (browser); the memory tier alone suffices for tests and locked-down embeds
// (pass `indexedDB:null` to disable the persistent tier).
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

// ServerSource — streams from the sphereql-vis-server HTTP API. Endpoints match
// the server exactly: GET /manifest, /category_stats, /diagnostics,
// /tiles?<query>; POST /points {rows}→.points, /nearest {row|vector,k}→.neighbors,
// /reproject {projection}→fresh manifest. Tiles arrive as binary SQT1 and are
// decoded (off-thread via a Worker-backed `decode`, else inline); blobs can be
// cached. The injectable `fetch` keeps it testable. All non-ok responses throw a
// human-readable message.
class ServerSource{
  constructor(baseUrl,opts){opts=opts||{};this.base=String(baseUrl||"").replace(/\/+$/,"");
    this._fetch=opts.fetch||(typeof fetch!=="undefined"?fetch.bind(typeof globalThis!=="undefined"?globalThis:null):null);
    this.cache=opts.cache||null;this.decode=opts.decode||decodeTile;this._gen=0;}
  async manifest(){return this._json("/manifest");}
  async categoryStats(){return this._json("/category_stats");}
  async diagnostics(){return this._json("/diagnostics");}
  // Live "tune": ask the server to re-project the corpus with a different kind;
  // returns the fresh manifest. Bumping `_gen` busts the tile-cache key (the old
  // positions are stale), so re-streaming after this never serves a pre-reproject
  // tile from cache.
  async reproject(projection){const m=await this._post("/reproject",{projection});this._gen++;return m;}
  async tiles(params){
    const urlQ=tileQuery(params);
    const key="/tiles?"+urlQ+(this._gen?"@"+this._gen:"");
    let buf=this.cache?await this.cache.get(key):null;
    if(buf==null){const res=await this._fetch(this.base+"/tiles?"+urlQ);if(!res.ok)throw new Error("tiles → "+res.status);buf=await res.arrayBuffer();if(this.cache)await this.cache.put(key,buf);}
    // Decode a throwaway copy when caching: a worker-backed `decode` transfers
    // (detaches) the buffer it is handed, which would corrupt the retained cache
    // entry and break every subsequent cache hit. The cache keeps the pristine
    // blob; the copy is what gets transferred. (Suite 10 regression-tests this.)
    return this.decode(this.cache?buf.slice(0):buf);
  }
  async pointMeta(rows){return(await this._post("/points",{rows:rows||[]})).points||[];}
  async nearest(q,k){const body={k:k||10};if(q&&q.row!=null)body.row=q.row;if(q&&Array.isArray(q.vector))body.vector=q.vector;return(await this._post("/nearest",body)).neighbors||[];}
  async _json(path){const res=await this._fetch(this.base+path);if(!res.ok)throw new Error(path+" → "+res.status);return res.json();}
  async _post(path,body){const res=await this._fetch(this.base+path,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});if(!res.ok)throw new Error(path+" → "+res.status);return res.json();}
}

// Off-thread tile decode: a tiny Worker built from inlined source (so the file
// stays self-contained — no external src=) that runs decodeTile and transfers
// the typed arrays back. Returns an async decode(buf); falls back to inline
// decode when Workers/Blob/URL are unavailable (Node tests, locked-down embeds).
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

// ── Streaming render: camera motion → a bounded working set of tile meshes ──
// TileStreamer maps camera distance→LOD and camera direction→a request cone,
// keeps one persistent coarse "base" tile (whole sphere, LOD 0) plus an
// LRU-bounded set of detail tiles, dedups identical viewports, and drops loads
// that resolve after their entry was evicted/cleared. Source + sink are
// injected (DataSource seam + tileMeshSink) so the orchestration is headless-
// testable. The actual THREE per-tile rendering lives in the sink.
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
  // moves: an identical viewport key dedups to a touch (no refetch); a load
  // that resolves after its entry was evicted/cleared is dropped (returns null).
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
// carries the streamed positions (server-final — NO client transform), palette
// colours (by cat id), point sizes, and the per-point pick id baked from the
// GLOBAL row (so id-buffer picking resolves a row across tiles).
function tileMeshSink(group,palette,material){
  const colors=(palette||[]).map(c=>new THREE.Color(safeColor(c.color)));
  const meshes=new Map();
  function _removeTileGeom(key){const m=meshes.get(key);if(m){group.remove(m);if(m.geometry&&m.geometry.dispose)m.geometry.dispose();meshes.delete(key);}}
  function addTile(key,data){
    if(meshes.has(key))_removeTileGeom(key);
    const n=data.count|0,geo=new THREE.BufferGeometry();
    const col=new Float32Array(n*3),size=new Float32Array(n),pick=new Float32Array(n*3);
    for(let i=0;i<n;i++){const c=colors[data.cats[i]]||new THREE.Color(0x90a4ae);col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;size[i]=DEF.size;
      const pc=pickEncodeId(data.rows[i]);pick[i*3]=pc[0];pick[i*3+1]=pc[1];pick[i*3+2]=pc[2];}
    geo.setAttribute("position",new THREE.BufferAttribute(data.positions,3));
    geo.setAttribute("color",new THREE.BufferAttribute(col,3));
    geo.setAttribute("size",new THREE.BufferAttribute(size,1));
    geo.setAttribute("aPickColor",new THREE.BufferAttribute(pick,3));
    const mesh=new THREE.Points(geo,material);mesh.frustumCulled=true;
    mesh.userData={rows:data.rows}; // global rows, for CPU picking → inspector
    group.add(mesh);meshes.set(key,mesh);
  }
  // Dispose only the per-tile geometry — the material is SHARED across all tiles
  // (disposing it per-tile would free an in-use GPU program and force a shader
  // recompile on every eviction).
  function removeTile(key){_removeTileGeom(key);}
  function clear(){for(const k of[...meshes.keys()])_removeTileGeom(k);}
  return {addTile,removeTile,clear,count:()=>meshes.size,meshAt:k=>meshes.get(k)};
}

// Shared material for streamed tiles (solid-disc points, palette-coloured). No
// sphTransform uniform — the server already projected the positions, so streamed
// tiles render with NO client transform (unlike the inline pointsMat, which
// carries sphTransform in lockstep with CPU curPos/transformPos).
let _streamColorMat=null;
function streamColorMaterial(){
  if(_streamColorMat)return _streamColorMat;
  _streamColorMat=new THREE.ShaderMaterial({vertexColors:true,transparent:true,depthWrite:false,uniforms:{opacity:{value:1.0}},
    vertexShader:`attribute float size;varying vec3 vc;void main(){vc=color;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=size*330.0/(-mv.z);gl_Position=projectionMatrix*mv;}`,
    fragmentShader:`uniform float opacity;varying vec3 vc;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;float a=smoothstep(0.5,0.44,d)*opacity;float core=smoothstep(0.32,0.0,d);gl_FragColor=vec4(mix(vc,vec3(1.0),core*0.4),a);}`});
  return _streamColorMat;
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
  // Settings-pane + tools controls (offline chrome). Resolved once; every use is
  // `if(el)`-guarded so a template missing a control degrades instead of throwing.
  const scaleS=q("scale","scale"),scaleVal=q("scale-val","scale-val");
  const zoomS=q("zoom","zoom"),zoomVal=q("zoom-val","zoom-val");
  const radialS=q("radial","radial"),radialVal=q("radial-val","radial-val");
  const spreadS=q("spread","spread"),spreadVal=q("spread-val","spread-val");
  const psizeS=q("psize","psize"),sizeVal=q("size-val","size-val");
  const uiS=q("ui","ui"),uiVal=q("ui-val","ui-val");
  const schemeSel=q("scheme","scheme");
  const globeCb=q("globe","globe");
  const autorotCb=q("autorot","autorot");
  const densityCb=q("density","density");
  const searchInput=q("search","search-input");
  const pinsDiv=q("pins","pins");
  const dropzone=q("dropzone","dropzone");
  const rulerReadout=q("ruler-readout","ruler-readout");

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
  // Persistent tool groups (like queryGroup): the great-circle ruler arc/markers
  // and the pin annotation markers. Cleared on scene swap, scaled with the scene.
  const rulerGroup=new THREE.Group();scene.add(rulerGroup);
  const pinGroup=new THREE.Group();scene.add(pinGroup);
  const raycaster=new THREE.Raycaster();
  const mouse=new THREE.Vector2();
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
  let baseSize=DEF.size,curScale=DEF.scale,spreadF=DEF.spread,radialG=DEF.radial,uiScale=DEF.ui;
  let selectedIdx=-1,hoveredIdx=-1;
  // Offline tools state: great-circle ruler (picks + last measurement) and pins
  // ((θ,φ) annotation markers + their DOM labels).
  let rulerOn=false,rulerPicks=[],rulerLast=null;
  let pins=[],pinEls=[],pinOn=false;
  let tgtTween=null,pendingTransform=false;
  let zoomLocked=false; // set by #embed compare host
  // drawChain animation callbacks: each returns true when the animation is done.
  const chainAnimations=[];

  // ── Streaming (out-of-core) instance state ───────────────────────────────
  // Declared once here; the streaming render path (connectToServer / the tile
  // sink) and the debugger UI (legend / inspect / tune / filter) below close
  // over these. dataSource is the active DataSource — an InlineSource for the
  // offline blob (installed in rebuild), a ServerSource once connected.
  let dataSource=null;          // active DataSource (InlineSource offline, ServerSource when streaming)
  let streamGroup=null;         // THREE.Group holding the streamed tile meshes
  let streamStreamer=null;      // the active TileStreamer
  let _streamOnMove=null;       // the controls "change" listener (released via cleanups)
  let _streamTimer=null;        // throttle timer for camera→request
  let _streamHoverPos=null;     // [x,y,z] of the hovered streamed point (reticle)
  let _streamFilterOff=new Set();// category NAMES toggled OFF (filter UI)
  let _streamPalette=[];        // the connected manifest's palette (cat → color/count)
  let _streamSelectedRow=null;  // global row of the currently-inspected streamed point

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
      if(bc&&om>=1e-4){
        if(om>Math.PI-1e-4){const h=Math.abs(dx)<0.9?[1,0,0]:[0,1,0];const hd=h[0]*dx+h[1]*dy+h[2]*dz;let px=h[0]-hd*dx,py=h[1]-hd*dy,pz=h[2]-hd*dz;const pm=Math.hypot(px,py,pz)||1;px/=pm;py/=pm;pz/=pm;const th=(1-spreadF)*om,c2=Math.cos(th),s2=Math.sin(th),ndx=dx*c2+px*s2,ndy=dy*c2+py*s2,ndz=dz*c2+pz*s2;dx=ndx;dy=ndy;dz=ndz;}
        else{const s=Math.sin(om),w1=Math.sin((1-spreadF)*om)/s,w2=Math.sin(spreadF*om)/s;
          const nx=bc[0]*w1+dx*w2,ny=bc[1]*w1+dy*w2,nz=bc[2]*w1+dz*w2,nm=Math.hypot(nx,ny,nz)||1;dx=nx/nm;dy=ny/nm;dz=nz/nm;}
      }
    }
    const nmag=Math.max(0.02,SR+(mag-SR)*radialG);
    return[dx*nmag,dy*nmag,dz*nmag];
  }

  function applyTransform(){
    if(!pointsMat)return;
    const u=pointsMat.uniforms;
    u.uSpread.value=spreadF;u.uRadial.value=radialG;u.uSR.value=SR;
    u.uMorphT.value=morphTarget?morphT:0;u.uHasMorph.value=morphTarget?1:0;
    // While a morph is active the GPU shader returns before the spread/radial
    // block, so points sit at their morphed/raw positions. The `from` end
    // already follows that via curPos; keep the `to` end (a static shell coord,
    // not a drawn point) raw too so the bridge stays attached instead of being
    // spread-transformed onto a stale position.
    for(const b of bridgeLines){const a=b.fromIndex>=0?curPos(b.fromIndex):transformPos(b.from),c=(morphTarget&&morphT>0)?b.to:transformPos(b.to),pos=b.line.geometry.getAttribute("position");pos.setXYZ(0,a[0],a[1],a[2]);pos.setXYZ(1,c[0],c[1],c[2]);pos.needsUpdate=true;}
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
    if(streamGroup&&streamStreamer)return pickStreamCPU(e); // streaming → global row
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
      if(om<1e-4){}
      else if(om>Math.PI-1e-4){const h=Math.abs(dx)<0.9?[1,0,0]:[0,1,0];const hd=h[0]*dx+h[1]*dy+h[2]*dz;let px=h[0]-hd*dx,py=h[1]-hd*dy,pz=h[2]-hd*dz;const pm=Math.hypot(px,py,pz)||1;px/=pm;py/=pm;pz/=pm;const th=(1-spreadF)*om,c2=Math.cos(th),s2=Math.sin(th),ndx=dx*c2+px*s2,ndy=dy*c2+py*s2,ndz=dz*c2+pz*s2;dx=ndx;dy=ndy;dz=ndz;}
      else{const s=Math.sin(om),w1=Math.sin((1-spreadF)*om)/s,w2=Math.sin(spreadF*om)/s;
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
      else{sa[i]=catVisible[pts[i].cat]?baseSize*0.5:0;ca[i*3]=base.r*0.28;ca[i*3+1]=base.g*0.28;ca[i*3+2]=base.b*0.28;}}
    pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;
    pointsMat.uniforms.opacity.value=0.4;
    while(linesGroup.children.length)linesGroup.remove(linesGroup.children[0]);
    const lm=new THREE.LineBasicMaterial({color:0x5cc8ff,transparent:true,opacity:0.5});
    for(const d of dists){const c=curPos(d.i);linesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(P[0],P[1],P[2]),new THREE.Vector3(c[0],c[1],c[2])]),lm));}
    const myBridges=bridgesByPoint[idx];
    if(myBridges)for(const br of myBridges){const c=(morphTarget&&morphT>0)?br.to:transformPos(br.to);
      linesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(P[0],P[1],P[2]),new THREE.Vector3(c[0],c[1],c[2])]),new THREE.LineBasicMaterial({color:br.color,transparent:true,opacity:0.9})));}
    const p=pts[idx];
    if(sellabel){sellabel.innerHTML=`<span class="sl-dot" style="background:${catColor[p.cat]};color:${catColor[p.cat]}"></span>${escHtml(p.label||"Point "+idx)}`;sellabel.style.display="block";}
    const infoLabel=q("info-label","info-label");if(infoLabel)infoLabel.textContent=p.label||"Point "+idx;
    const infoTag=q("info-cat","info-cat");if(infoTag){infoTag.textContent=p.cat;infoTag.style.color=catColor[p.cat];infoTag.style.background=catColor[p.cat]+"18";}
    const infoCoords=q("info-coords","info-coords");
    if(infoCoords)infoCoords.innerHTML=`<span>θ</span><b>${p.theta.toFixed(4)}</b><span>φ</span><b>${p.phi.toFixed(4)}</b><span>r</span><b>${p.r.toFixed(4)}</b><span>str</span><b>${deriveStrength(p).toFixed(2)}</b>${myBridges?`<span>bridges</span><b>${myBridges.length}</b>`:""}`;
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

  function setAll(v){soloCat=null;catSet.forEach(c=>{catVisible[c]=v;if(legendRows[c])legendRows[c].classList.toggle("dim",!v);});updateVisibility();}
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

  // Dispose the map texture too: Material.dispose() does NOT cascade to its
  // textures, so without this every makeTextSprite CanvasTexture (relation
  // labels) leaks on the GPU across chain clear / overlay teardown / rebuild.
  // `.map` is the only texture slot used in this viewer.
  function disposeObject(o){if(!o)return;o.traverse(c=>{if(c.geometry)c.geometry.dispose();if(c.material){const m=c.material;(Array.isArray(m)?m:[m]).forEach(x=>{if(x){if(x.map&&x.map.dispose)x.map.dispose();if(x.dispose)x.dispose();}});}});}

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
    // Reset any prior point selection (without reverting the camera) so its
    // neighbor/bridge lines + sellabel don't linger over the query highlight,
    // and selectedIdx is cleared so the next updateScene tick's in-place
    // selectPoint re-apply can't overwrite this query's per-point emphasis.
    if(selectedIdx>=0)deselectPoint();
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
    disconnectServer(); // drop any active server stream (idempotent; no-op when offline) — else a scene swap leaves a zombie stream that keeps getHovered/updateHover on the streaming branch
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
    setRuler(false); // fully disarm the ruler (flag + button + picks) on scene swap
    setPinMode(false);clearPins(); // pins annotated the outgoing scene's shell
    selectedIdx=-1;hoveredIdx=-1;tgtTween=null;pendingTransform=false;
  }

  // ── rebuild(sc) ──────────────────────────────────────────────────────────
  // Full scene swap from a parsed Scene object. Resets view settings to
  // defaults and re-frames the camera unless opts.preserveCamera is set.
  function rebuild(sc,_opts){
    const preserveCamera=_opts&&_opts.preserveCamera;
    teardown();
    // Offline render routes through an InlineSource wrapping this scene (the
    // DataSource seam). teardown() above dropped any active stream and nulled
    // dataSource, so install the inline source AFTER it; connectToServer
    // overwrites dataSource with a ServerSource after its own rebuild.
    dataSource=new InlineSource(sc);
    pts=sc.points||[];N=pts.length;overlays=sc.overlays||[];SR=sc.surface_radius||1.0;showAxes=!!sc.show_axes;
    maxR=1;for(const p of pts){const m=Math.hypot(p.x,p.y,p.z);if(m>maxR)maxR=m;}

    baseSize=DEF.size;curScale=DEF.scale;spreadF=DEF.spread;radialG=DEF.radial;
    // Sync the Settings sliders to the freshly-reset view transforms (they may
    // have been dragged for the previous scene). Guarded — controls are optional;
    // zoom/ui/autorot are persistent user prefs and intentionally not reset here.
    if(scaleS)scaleS.value=DEF.scale;if(scaleVal)scaleVal.textContent=DEF.scale.toFixed(1)+"×";
    if(radialS)radialS.value=DEF.radial;if(radialVal)radialVal.textContent=DEF.radial.toFixed(1)+"×";
    if(spreadS)spreadS.value=DEF.spread;if(spreadVal)spreadVal.textContent=DEF.spread.toFixed(1)+"×";
    if(psizeS)psizeS.value=DEF.size;if(sizeVal)sizeVal.textContent=DEF.size.toFixed(1);
    if(schemeSel)schemeSel.value=DEF.palette;
    if(densityCb)densityCb.checked=DEF.density;
    if(globeCb)globeCb.checked=DEF.globe;
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

    scalables=[pointsMesh,linesGroup,globeGroup,rulerGroup,queryGroup,chainGroup,pinGroup,...Object.values(overlayGroups)];

    // ── Legend ───────────────────────────────────────────────────────────
    legendRows={};
    if(legendDiv){
      catSet.forEach(cat=>{
        const row=document.createElement("div");row.className="lrow";
        row.innerHTML=`<span class="ldot" style="background:${catColor[cat]};color:${catColor[cat]}"></span><span class="lbl"></span><span class="lcnt">${catCounts[cat]}</span>`;
        row.querySelector(".lbl").textContent=cat;legendRows[cat]=row;
        row.addEventListener("click",()=>{soloCat=null;catVisible[cat]=!catVisible[cat];row.classList.toggle("dim",!catVisible[cat]);updateVisibility();});
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
      rebuild(sc,{preserveCamera:true}); // resets DEF (and the Settings sliders) but does not call frameCamera
      // Restore view state — scalables are freshly created so applyScale is safe.
      applyScale(savedView.scale);
      baseSize=savedView.size;spreadF=savedView.spread;radialG=savedView.radial;
      controls.zoomSpeed=savedView.zoom;
      applySize(savedView.size);
      // Re-sync the Settings sliders to the restored view (rebuild set them to
      // DEF; the actual transform vars above are savedView — keep the DOM honest).
      if(scaleS)scaleS.value=savedView.scale;if(scaleVal)scaleVal.textContent=savedView.scale.toFixed(1)+"×";
      if(radialS)radialS.value=savedView.radial;if(radialVal)radialVal.textContent=savedView.radial.toFixed(1)+"×";
      if(spreadS)spreadS.value=savedView.spread;if(spreadVal)spreadVal.textContent=savedView.spread.toFixed(1)+"×";
      if(psizeS)psizeS.value=savedView.size;if(sizeVal)sizeVal.textContent=savedView.size.toFixed(1);
      if(zoomS)zoomS.value=savedView.zoom;if(zoomVal)zoomVal.textContent=savedView.zoom.toFixed(2)+"×";
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
    // selectPoint rewrites the whole size buffer itself, so it covers the
    // selected case. With no selection, refresh size from catVisible here: slots
    // recycle across categories on a tick (see the idToIndex rebuild above), so a
    // slot whose cat changed would otherwise keep a stale visible/hidden size.
    if(selectedIdx>=0)selectPoint(selectedIdx,{skipTween:true});
    else{const sa=pointsGeo.getAttribute("size").array;for(let i=0;i<N;i++)sa[i]=catVisible[pts[i].cat]?baseSize:0;pointsGeo.getAttribute("size").needsUpdate=true;}
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

    // grp inherits chainGroup's scale — chainGroup is in `scalables`, so
    // applyScale already drives it by curScale (matching drawChainIntoGroup's
    // overlay groups and highlightByIds' queryGroup). Setting grp.scale here too
    // would apply curScale twice (curScale²), throwing the chain off-screen.
    const grp=new THREE.Group();
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
    if(streamStreamer){ // streaming: idx is a global row; no `pts[]` to read.
      // Don't fetch per-hover (it would spam /points); the reticle (animate loop)
      // is the hover affordance, click → selectStreamRow inspects.
      if(tooltip)tooltip.style.display="none";
      canvas.style.cursor=idx>=0?"crosshair":"grab";
      return;
    }
    if(idx>=0){const p=pts[idx];
      if(tooltip){tooltip.innerHTML=`<div class="tt-lbl">${escHtml(p.label||"Point "+idx)}</div><div class="tt-meta">${escHtml(p.cat)} · θ ${p.theta.toFixed(2)}  φ ${p.phi.toFixed(2)}  r ${p.r.toFixed(2)}  str ${deriveStrength(p).toFixed(2)}</div>`;
        tooltip.style.display="block";tooltip.style.left=(e.clientX+16)+"px";tooltip.style.top=(e.clientY+14)+"px";}
      canvas.style.cursor="crosshair";
    }else{if(tooltip)tooltip.style.display="none";canvas.style.cursor="grab";}
  }

  // ── Streaming debugger UI + connect/disconnect ──────────────────────────
  // Streaming pick: project every loaded tile point, return the GLOBAL row
  // nearest the cursor (within a small radius). O(loaded points), coalesced to
  // one pick/frame by updateHover. Sets _streamHoverPos for the hover reticle.
  function pickStreamCPU(e){
    _streamHoverPos=null;
    if(!streamGroup)return -1;
    const rect=canvas.getBoundingClientRect();
    let best=-1,bestD=14*14;
    for(const mesh of streamGroup.children){
      const pos=mesh.geometry&&mesh.geometry.getAttribute&&mesh.geometry.getAttribute("position");
      const rows=mesh.userData&&mesh.userData.rows;
      if(!pos||!rows)continue;const a=pos.array;
      for(let i=0;i<rows.length;i++){const x=a[i*3],y=a[i*3+1],z=a[i*3+2],sp=projectToScreen([x,y,z]);
        if(!sp.vis)continue;const dx=sp.x-(e.clientX-rect.left),dy=sp.y-(e.clientY-rect.top),d=dx*dx+dy*dy;
        if(d<bestD){bestD=d;best=rows[i];_streamHoverPos=[x,y,z];}}
    }
    return best;
  }

  // Streaming-mode legend from the manifest palette (name · count, in the
  // category colour). Clicking a row toggles that category out of the stream
  // (server-side tile filter), then re-applies the filter + reloads diagnostics.
  // safeColor() guards every server-supplied colour before it reaches a style="".
  function buildStreamLegend(palette){
    if(legendDiv){legendDiv.innerHTML="";}
    legendRows={};catColor={};catVisible={};
    (palette||[]).forEach(c=>{
      const col=safeColor(c.color);catColor[c.name]=col;catVisible[c.name]=!_streamFilterOff.has(c.name);
      if(!legendDiv)return;
      const row=document.createElement("div");row.className="lrow"+(_streamFilterOff.has(c.name)?" dim":"");
      row.innerHTML=`<span class="ldot" style="background:${col};color:${col}"></span><span class="lbl"></span><span class="lcnt">${(+c.count||0).toLocaleString()}</span>`;
      row.querySelector(".lbl").textContent=c.name; // textContent: category name is foreign
      legendRows[c.name]=row;
      row.addEventListener("click",()=>{
        if(_streamFilterOff.has(c.name))_streamFilterOff.delete(c.name);else _streamFilterOff.add(c.name);
        row.classList.toggle("dim",_streamFilterOff.has(c.name));catVisible[c.name]=!_streamFilterOff.has(c.name);
        applyStreamFilter();
      },sig);
      legendDiv.appendChild(row);
    });
  }
  // Recompute the streaming filter (enabled categories + min-certainty), push it
  // to the streamer (server-side tile filter), then refresh diagnostics.
  function applyStreamFilter(){
    if(!streamStreamer)return;
    const cats=[];_streamPalette.forEach((c,i)=>{if(!_streamFilterOff.has(c.name))cats.push(i);});
    const allOn=cats.length===_streamPalette.length;
    // The /tiles `cats` param can't express the empty set (an empty list is
    // dropped → server returns ALL). So when EVERY category is toggled off, hide
    // the streamed tile group client-side instead; otherwise show it and push
    // the server-side filter (allOn → no cats param).
    if(streamGroup)streamGroup.visible=cats.length>0;
    const mcEl=q("mincert","mincert"),mc=mcEl?parseFloat(mcEl.value)||0:0;
    streamStreamer.setFilter({cats:allOn?[]:cats,minCertainty:mc>0?mc:undefined}).then(()=>loadDiagnostics());
  }

  // Diverging bar sparkline of a raw embedding vector into the inspector canvas
  // (+ blue, − warm). Empty/absent vector hides the wrap. The wrap node may be
  // absent (old template), so every ref is guarded.
  function renderVectorSparkline(vec){
    const wrap=q("info-vector-wrap","info-vector-wrap");
    if(!wrap)return;
    if(!Array.isArray(vec)||!vec.length){wrap.style.display="none";return;}
    wrap.style.display="block";
    const cv=q("info-vector","info-vector");
    const ctx=cv&&cv.getContext&&cv.getContext("2d");if(!ctx)return;
    const W=cv.width,H=cv.height,mid=H/2;ctx.clearRect(0,0,W,H);
    let mx=1e-9;for(const v of vec)if(isFinite(+v))mx=Math.max(mx,Math.abs(+v));
    const n=vec.length,bw=W/n;
    for(let i=0;i<n;i++){const v=+vec[i]||0,h=Math.abs(v)/mx*(mid-1);
      ctx.fillStyle=v>=0?"#5cc8ff":"#ff8a65";ctx.fillRect(i*bw,v>=0?mid-h:mid,Math.max(1,bw-0.3),Math.max(1,h));}
    ctx.strokeStyle="rgba(120,160,255,0.25)";ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(W,mid);ctx.stroke();
  }

  // Inspect a streamed point by GLOBAL row: fetch metadata + ANN neighbors from
  // the server, fill the info panel (label, category, θ/φ/r/cert derived from
  // xyz, raw-vector sparkline, clickable cosine neighbors). The OFFLINE
  // selectPoint path is untouched and used for inline scenes.
  async function selectStreamRow(row){
    if(!streamStreamer)return;
    _streamSelectedRow=row;
    const src=streamStreamer.source;
    let meta,nbrs=[];
    try{const ms=await src.pointMeta([row]);meta=ms&&ms[0];if(!meta)return;nbrs=await src.nearest({row},6);}
    catch(err){console.warn("SphereQL: inspect fetch failed",err);return;}
    const lab=q("info-label","info-label");if(lab)lab.textContent=meta.label||("point #"+row);
    const tag=q("info-cat","info-cat"),col=catColor[meta.category]||"#5cc8ff";
    if(tag){tag.textContent=meta.category||"";tag.style.color=col;tag.style.background=col+"18";}
    const f=v=>isFinite(+v)?(+v).toFixed(4):"—";
    // The meta carries xyz, not angles — derive r/θ/φ.
    const px=+meta.x,py=+meta.y,pz=+meta.z,r=Math.hypot(px,py,pz);
    let th=Math.atan2(py,px);if(th<0)th+=2*Math.PI;const ph=Math.acos(clamp(r>1e-12?pz/r:0,-1,1));
    const coords=q("info-coords","info-coords");
    if(coords)coords.innerHTML=`<span>θ</span><b>${f(th)}</b><span>φ</span><b>${f(ph)}</b><span>r</span><b>${f(r)}</b><span>cert</span><b>${f(meta.certainty)}</b>`;
    renderVectorSparkline(meta.vector);
    const nb=q("info-neighbors","info-neighbors");
    if(nb){
      const rows=nbrs.map(h=>h.row),lookup={};
      try{for(const m of await src.pointMeta(rows))lookup[m.row]=m;}catch(err){/* labels best-effort */}
      nb.innerHTML=nbrs.map(h=>{const m=lookup[h.row],c=m?(catColor[m.category]||"#5cc8ff"):"#5cc8ff";
        return `<div class="nb" data-row="${h.row}" style="background:${c}22;border-left:2px solid ${c}"><span>${escHtml(m?(m.label||("#"+h.row)):("#"+h.row))}</span><span class="dist">${(+h.similarity).toFixed(3)}</span></div>`;}).join("");
      nb.querySelectorAll(".nb").forEach(el=>el.addEventListener("click",()=>selectStreamRow(parseInt(el.dataset.row)),sig));
    }
    const head=q("info-nb-head","info-nb-head");if(head)head.textContent="Nearest · cosine";
    const info=q("info","info");if(info)info.classList.add("visible");
  }

  // Render the /diagnostics payload (EVR + warnings + certainty/intensity
  // histograms + low-certainty outliers) into the Diag tab. Outliers and
  // warnings are foreign; escape every string. Severity is clamped to the known
  // CSS classes so a crafted severity can't inject a class token.
  const SEVERITY=new Set(["info","warn","critical"]);
  function renderDiagnostics(d){
    const el=q("diag-content","diag-content");if(!el)return;
    if(!d){el.innerHTML='<div class="muted">no diagnostics</div>';return;}
    const evr=clamp((+d.evr||0)*100,0,100);
    const hist=h=>{const bins=(h&&h.bins)||[],mx=Math.max(1,...bins);
      return `<div class="histo">${bins.map(b=>`<i style="height:${(b/mx*100).toFixed(1)}%"></i>`).join("")}</div>`
        +`<div class="histo-cap"><span>${h&&isFinite(h.min)?(+h.min).toFixed(2):""}</span><span>${h&&isFinite(h.max)?(+h.max).toFixed(2):""}</span></div>`;};
    let html=`<div class="srow"><span>projection</span><span class="v hl">${escHtml(d.projection_kind||"?")}</span></div>`;
    html+=`<div class="srow"><span>${escHtml(d.evr_label||"EVR")}</span><span class="v">${evr.toFixed(1)}%</span></div><div class="bar"><i style="width:${evr.toFixed(1)}%"></i></div>`;
    html+=`<div class="srow"><span>points</span><span class="v">${(+d.total_points||0).toLocaleString()}</span></div>`;
    for(const w of d.warnings||[]){const sev=SEVERITY.has(w&&w.severity)?w.severity:"info";html+=`<div class="warn ${sev}">${escHtml((w&&w.message)||"")}</div>`;}
    html+=`<h3 style="margin:13px 0 4px">Certainty</h3>${hist(d.certainty)}`;
    html+=`<h3 style="margin:13px 0 4px">Intensity</h3>${hist(d.intensity)}`;
    if((d.outliers||[]).length){html+='<h3 style="margin:13px 0 5px">Low-certainty outliers</h3>';
      html+=(d.outliers||[]).map(o=>`<div class="nb" data-row="${+o.row}"><span>${escHtml(o.label||("#"+o.row))}</span><span class="dist">${(+o.certainty).toFixed(3)}</span></div>`).join("");}
    el.innerHTML=html;
    el.querySelectorAll(".nb[data-row]").forEach(x=>x.addEventListener("click",()=>selectStreamRow(parseInt(x.dataset.row)),sig));
  }
  async function loadDiagnostics(){
    if(!streamStreamer)return;
    try{renderDiagnostics(await streamStreamer.source.diagnostics());}catch(err){console.warn("SphereQL: diagnostics failed",err);}
  }

  // Point the viewer at a sphereql-vis-server: fetch the manifest, build the
  // scene chrome (globe + overlays + stats + a palette legend) with NO inline
  // points, then stream point tiles by viewport via a TileStreamer + wire the
  // debugger rows (tune/filter/diag). Returns the streamer. Browser-validated.
  async function connectToServer(baseUrl,opts){
    opts=opts||{};
    disconnectServer();
    const source=new ServerSource(baseUrl,{cache:new TileCache(),decode:makeWorkerDecoder()});
    const manifest=await source.manifest();
    _streamFilterOff=new Set();_streamSelectedRow=null;_streamPalette=manifest.palette||[];
    // Chrome via rebuild with zero inline points: the globe (surface_radius),
    // overlays (manifest.overlays), and stats panel populate; the empty
    // pointsMesh costs nothing. The palette legend replaces the per-point one.
    rebuild({title:manifest.title,stats:manifest.stats,overlays:manifest.overlays||[],surface_radius:manifest.surface_radius||1,show_axes:false,points:[]});
    // rebuild installed an InlineSource for the (empty) chrome scene; the active
    // source is the server now — set AFTER rebuild so it isn't clobbered.
    dataSource=source;
    {const el=q("empty","empty");if(el)el.style.display="none";}
    buildStreamLegend(_streamPalette);
    // The tile group is a scalable so the slider scales streamed tiles with the
    // rest of the scene; tiles carry server-final positions and the sink material
    // has no sphTransform, so only the group's uniform scale applies.
    streamGroup=new THREE.Group();scene.add(streamGroup);scalables.push(streamGroup);streamGroup.scale.setScalar(curScale);
    const sink=tileMeshSink(streamGroup,_streamPalette,streamColorMaterial());
    streamStreamer=new TileStreamer(source,sink,opts);
    await streamStreamer.startWith(manifest);
    // Camera → viewport tile updates, throttled (120ms) so a drag doesn't spam
    // requests. dist is in shell units (un-scaled) so lodFor is scale-invariant.
    const camToReq=()=>{const p=camera.position,m=Math.hypot(p.x,p.y,p.z)||1;let th=Math.atan2(p.y,p.x);if(th<0)th+=2*Math.PI;return{theta:th,phi:Math.acos(clamp(p.z/m,-1,1)),dist:m/Math.max(curScale,1e-6)};};
    let pend=false;
    _streamOnMove=()=>{if(pend)return;pend=true;_streamTimer=setTimeout(()=>{pend=false;_streamTimer=null;if(streamStreamer)streamStreamer.update(camToReq());},120);};
    controls.addEventListener("change",_streamOnMove);
    // THREE's EventDispatcher has no AbortSignal support, so register the
    // removal through cleanups[] — dispose() runs these and releases the stream.
    cleanups.push(()=>{if(_streamOnMove&&controls.removeEventListener)controls.removeEventListener("change",_streamOnMove);_streamOnMove=null;});

    // ── Debugger rows: tune (live re-projection) + filter (min-certainty) ──
    {const mc=q("mincert","mincert");if(mc)mc.value=0;const mcv=q("mincert-val","mincert-val");if(mcv)mcv.textContent="0.00";}
    const showRow=id=>{const el=q(id,id);if(el)el.style.display="block";};
    showRow("tune-row");showRow("filter-row");
    // Tune: server projection-name map ⇄ studio enum. Routing studio's projection
    // change through the server avoids the slow WASM UMAP on the demo corpus (S2).
    const _sp2srv={"":"pca",UmapSphere:"umap_sphere",LaplacianEigenmap:"laplacian",KernelPca:"kernel_pca"};
    const _srv2sp={pca:"",umap_sphere:"UmapSphere",laplacian:"LaplacianEigenmap",kernel_pca:"KernelPca"};
    const tune=q("tune-proj","tune-proj");
    const studioProj=(typeof document!=="undefined")?document.getElementById("studio-proj"):null; // studio chrome lives outside rootEl
    const doReproject=async serverProj=>{
      try{
        const newM=await source.reproject(serverProj);
        _streamPalette=newM.palette||_streamPalette;
        const pk=(newM.stats&&newM.stats.projection_kind)||serverProj;
        const pill=q("hdr-pill","hdr-pill");if(pill)pill.textContent=pk;
        if(studioProj)studioProj.value=_srv2sp[pk]||"";
        if(tune)tune.value=pk;
        // A reproject moves every point — old-projection tiles are stale
        // geometry. Clear them before re-streaming (startWith doesn't clear).
        streamStreamer.clear();streamStreamer.tiles=new Map();
        await streamStreamer.startWith(newM); // re-stream from the new projection
        streamStreamer.update(camToReq());
        loadDiagnostics();
      }catch(err){console.warn("SphereQL: reproject failed",err);}
    };
    if(tune)tune.addEventListener("change",()=>doReproject(tune.value),sig);
    // studio.js:96 delegates its projection change here when connected.
    window.__sqServerReproject=studioVal=>doReproject(_sp2srv[studioVal]||"pca");
    if(manifest.stats&&manifest.stats.projection_kind){const pk=manifest.stats.projection_kind;if(studioProj)studioProj.value=_srv2sp[pk]||"";if(tune)tune.value=pk;}
    // Min-certainty slider → server-side filter.
    const mcSlider=q("mincert","mincert");
    if(mcSlider)mcSlider.addEventListener("input",()=>{const v=q("mincert-val","mincert-val");if(v)v.textContent=(+mcSlider.value).toFixed(2);applyStreamFilter();},sig);

    streamStreamer.update(camToReq());
    loadDiagnostics();
    {const el=q("server-url","server-url");if(el)el.value=source.base;}
    {const btn=q("server-connect","server-connect");if(btn)btn.textContent="Disconnect";}
    return streamStreamer;
  }

  // Tear down an active server stream (its tile group + camera listener + the
  // reproject delegate + debugger rows). Idempotent — connectToServer calls it
  // first, dispose() relies on it.
  function disconnectServer(){
    window.__sqServerReproject=null;
    if(_streamOnMove&&controls.removeEventListener)controls.removeEventListener("change",_streamOnMove);
    _streamOnMove=null;
    if(_streamTimer){clearTimeout(_streamTimer);_streamTimer=null;}
    if(streamStreamer){streamStreamer.clear();streamStreamer=null;}
    if(streamGroup){scene.remove(streamGroup);disposeObject(streamGroup);const i=scalables.indexOf(streamGroup);if(i>=0)scalables.splice(i,1);streamGroup=null;}
    dataSource=null;
    _streamSelectedRow=null;_streamHoverPos=null;
    {const tr=q("tune-row","tune-row");if(tr)tr.style.display="none";}
    {const fr=q("filter-row","filter-row");if(fr)fr.style.display="none";}
    {const dc=q("diag-content","diag-content");if(dc)dc.innerHTML='<div class="muted">Connect to a <code>sphereql-vis-server</code> (Settings tab) for live projection diagnostics.</div>';}
    {const btn=q("server-connect","server-connect");if(btn)btn.textContent="Connect";}
  }

  // ── Offline tools: palette / settings / ruler / pins / PNG / scene-load ──
  // Briefly flash a button's label as transient feedback, then restore it.
  function flashButton(id,text,ms){const b=q(id,id);if(!b)return;const t=b.textContent;b.textContent=text;setTimeout(()=>{b.textContent=t;},ms||1500);}

  // Recolor the cloud + legend dots to a named palette (no minimap — it was
  // dropped from the template). Re-applies the active selection in place.
  function applyPalette(name){
    buildCatColor(name);
    if(pointsGeo){const ca=pointsGeo.getAttribute("color").array;
      for(let i=0;i<N;i++){const c=new THREE.Color(catColor[pts[i].cat]);ca[i*3]=c.r;ca[i*3+1]=c.g;ca[i*3+2]=c.b;}
      pointsGeo.getAttribute("color").needsUpdate=true;}
    catSet.forEach(c=>{const dot=legendRows[c]&&legendRows[c].querySelector(".ldot");if(dot){dot.style.background=catColor[c];dot.style.color=catColor[c];}});
    if(pointsMat)pointsMat.uniforms.opacity.value=1.0;
    if(selectedIdx>=0)selectPoint(selectedIdx); // re-apply highlight in place (no info-panel flicker)
  }

  // ── Great-circle ruler ─────────────────────────────────────────────────
  // Click two points; measure the angle between their directions (acos of the
  // clamped dot) and draw the connecting geodesic on the shell via shellArc().
  function clearRuler(){rulerPicks=[];while(rulerGroup.children.length){const c=rulerGroup.children[0];disposeObject(c);rulerGroup.remove(c);}if(rulerReadout)rulerReadout.classList.remove("on");}
  function setRuler(on){
    rulerOn=on;{const b=q("tool-ruler","tool-ruler");if(b)b.classList.toggle("active",on);}
    if(on){if(rulerReadout){const a=rulerReadout.querySelector(".rr-ang");if(a)a.textContent="—";const s=rulerReadout.querySelector(".rr-sub");if(s)s.textContent="click two points · Esc to clear";rulerReadout.classList.add("on");}}
    else clearRuler();
  }
  function rulerMeasure(){
    const a=rulerPicks[0],b=rulerPicks[1];
    const om=Math.acos(clamp(a[0]*b[0]+a[1]*b[1]+a[2]*b[2],-1,1));
    rulerGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(shellArc(a,b)),new THREE.LineBasicMaterial({color:0xffb454,transparent:true,opacity:0.95})));
    const deg=om*180/Math.PI;rulerLast={rad:om,deg:deg,chord:2*Math.sin(om/2)};
    if(rulerReadout){const an=rulerReadout.querySelector(".rr-ang");if(an)an.textContent=deg.toFixed(1)+"°  ·  "+om.toFixed(3)+" rad";const s=rulerReadout.querySelector(".rr-sub");if(s)s.textContent="great-circle · chord "+(2*Math.sin(om/2)).toFixed(3);rulerReadout.classList.add("on");}
  }
  function rulerAddPick(P){
    if(rulerPicks.length>=2)clearRuler();
    const m=Math.hypot(P[0],P[1],P[2])||1;
    rulerPicks.push([P[0]/m,P[1]/m,P[2]/m]);
    rulerGroup.add(marker(P,0xffb454,SR*0.02));
    if(rulerPicks.length===2)rulerMeasure();
    else if(rulerReadout){const a=rulerReadout.querySelector(".rr-ang");if(a)a.textContent="•";const s=rulerReadout.querySelector(".rr-sub");if(s)s.textContent="pick the second point";rulerReadout.classList.add("on");}
  }

  // ── PNG snapshot ────────────────────────────────────────────────────────
  function exportPNG(){
    try{renderer.render(scene,camera);const url=canvas.toDataURL("image/png");const a=document.createElement("a");a.href=url;a.download="sphereql-view.png";a.click();flashButton("tool-png","✓",1200);}
    catch(err){console.warn("SphereQL: PNG export failed:",err);flashButton("tool-png","✗");}
  }

  // ── Pins ((θ,φ) annotation markers + DOM labels) ────────────────────────
  function setPinMode(on){pinOn=on;const b=q("tool-pin","tool-pin");if(b)b.classList.toggle("active",on);}
  function clearPins(){pins=[];renderPins();}
  function renderPins(){
    while(pinGroup.children.length){const c=pinGroup.children[0];disposeObject(c);pinGroup.remove(c);}
    if(pinsDiv)pinsDiv.innerHTML="";pinEls=[];
    for(const pin of pins){
      const sp=Math.sin(pin.phi),dir=[sp*Math.cos(pin.theta),sp*Math.sin(pin.theta),Math.cos(pin.phi)],pos=[dir[0]*SR,dir[1]*SR,dir[2]*SR];
      pinGroup.add(marker(pos,0xffb454,SR*0.018));
      if(!pinsDiv)continue;
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

  // ── Config (TOML) save / load ───────────────────────────────────────────
  function currentSettings(){return{scale:curScale,zoom_speed:controls.zoomSpeed,radial:radialG,domain_spread:spreadF,point_size:baseSize,ui_scale:uiScale,color_scheme:schemeSel?schemeSel.value:DEF.palette,reference_globe:globeCb?!!globeCb.checked:DEF.globe,auto_rotate:autorotCb?!!autorotCb.checked:DEF.autorot,density:densityCb?!!densityCb.checked:DEF.density,pins:btoa(encodeURIComponent(JSON.stringify(pins)))};}
  function toToml(o){return"# SphereQL view settings\n"+Object.entries(o).map(([k,v])=>{const val=typeof v==="string"?`"${v}"`:(typeof v==="boolean"?(v?"true":"false"):v);return `${k} = ${val}`;}).join("\n")+"\n";}
  function parseToml(text){const o={};String(text).split(/\r?\n/).forEach(line=>{line=line.trim();if(!line||line[0]==="#")return;const i=line.indexOf("=");if(i<0)return;const k=line.slice(0,i).trim();let v=line.slice(i+1).trim();if(/^".*"$/.test(v))v=v.slice(1,-1);else if(v==="true")v=true;else if(v==="false")v=false;else{const n=parseFloat(v);if(!isNaN(n))v=n;}o[k]=v;});return o;}
  function applySettings(o){
    if(!o||typeof o!=="object")return;
    // Numeric fields are isFinite-guarded: a malformed/empty value in a hand-
    // edited .toml (e.g. `scale = `) would otherwise coerce via + to 0 and
    // applyScale(0)/applySize(0) → an invisible, zero-scale scene with no error.
    if("scale"in o&&isFinite(+o.scale)){const s=+o.scale;if(scaleS)scaleS.value=s;if(scaleVal)scaleVal.textContent=s.toFixed(1)+"×";applyScale(s);}
    if("zoom_speed"in o&&isFinite(+o.zoom_speed)){const v=+o.zoom_speed;if(zoomS)zoomS.value=v;if(zoomVal)zoomVal.textContent=v.toFixed(2)+"×";controls.zoomSpeed=v;}
    if("radial"in o&&isFinite(+o.radial)){radialG=+o.radial;if(radialS)radialS.value=radialG;if(radialVal)radialVal.textContent=radialG.toFixed(1)+"×";}
    if("domain_spread"in o&&isFinite(+o.domain_spread)){spreadF=+o.domain_spread;if(spreadS)spreadS.value=spreadF;if(spreadVal)spreadVal.textContent=spreadF.toFixed(1)+"×";}
    if("point_size"in o&&isFinite(+o.point_size)){const v=+o.point_size;if(psizeS)psizeS.value=v;if(sizeVal)sizeVal.textContent=v.toFixed(1);applySize(v);}
    if("ui_scale"in o&&isFinite(+o.ui_scale)){uiScale=+o.ui_scale;if(uiS)uiS.value=uiScale;if(uiVal)uiVal.textContent=uiScale.toFixed(2)+"×";if(typeof document!=="undefined"&&document.documentElement)document.documentElement.style.setProperty("--ui",uiScale);}
    if("color_scheme"in o&&PALETTES[o.color_scheme]){if(schemeSel)schemeSel.value=o.color_scheme;applyPalette(o.color_scheme);}
    if("reference_globe"in o){if(globeCb)globeCb.checked=!!o.reference_globe;if(globeGroup)globeGroup.visible=!!o.reference_globe;}
    if("auto_rotate"in o){if(autorotCb)autorotCb.checked=!!o.auto_rotate;controls.autoRotate=!!o.auto_rotate;}
    if("density"in o){if(densityCb)densityCb.checked=!!o.density;if(pointsMat)pointsMat.uniforms.densityOn.value=o.density?1:0;}
    if("pins"in o){try{const arr=JSON.parse(decodeURIComponent(atob(o.pins)));if(Array.isArray(arr)){pins=arr.filter(p=>p&&isFinite(+p.theta)&&isFinite(+p.phi)).map(p=>({theta:+p.theta,phi:+p.phi,label:String(p.label||"")}));renderPins();}}catch(err){console.warn("SphereQL: ignoring bad pins in settings");}}
    applyTransform();
  }

  // ── Reset all view settings to defaults ─────────────────────────────────
  function resetDefaults(){
    if(scaleS)scaleS.value=DEF.scale;if(scaleVal)scaleVal.textContent=DEF.scale.toFixed(1)+"×";applyScale(DEF.scale);
    if(zoomS)zoomS.value=DEF.zoom;if(zoomVal)zoomVal.textContent=DEF.zoom.toFixed(2)+"×";controls.zoomSpeed=DEF.zoom;
    radialG=DEF.radial;if(radialS)radialS.value=DEF.radial;if(radialVal)radialVal.textContent=DEF.radial.toFixed(1)+"×";
    spreadF=DEF.spread;if(spreadS)spreadS.value=DEF.spread;if(spreadVal)spreadVal.textContent=DEF.spread.toFixed(1)+"×";
    applySize(DEF.size);if(psizeS)psizeS.value=DEF.size;if(sizeVal)sizeVal.textContent=DEF.size.toFixed(1);
    uiScale=DEF.ui;if(uiS)uiS.value=DEF.ui;if(uiVal)uiVal.textContent=DEF.ui.toFixed(2)+"×";if(typeof document!=="undefined"&&document.documentElement)document.documentElement.style.setProperty("--ui",DEF.ui);
    if(schemeSel)schemeSel.value=DEF.palette;applyPalette(DEF.palette);
    if(globeCb)globeCb.checked=DEF.globe;if(globeGroup)globeGroup.visible=DEF.globe;
    if(autorotCb)autorotCb.checked=DEF.autorot;controls.autoRotate=DEF.autorot;
    if(densityCb)densityCb.checked=DEF.density;if(pointsMat)pointsMat.uniforms.densityOn.value=DEF.density?1:0;
    setPinMode(false);clearPins();
    labelKindsPresent.forEach(k=>(labelKindOn[k]=true));
    if(labelTogglesDiv)labelTogglesDiv.querySelectorAll("input").forEach(cb=>{cb.checked=true;});
    soloCat=null;setAll(true);
    if(searchInput)searchInput.value="";applyTransform();frameCamera();
  }

  // ── Open / drop a foreign Scene JSON ────────────────────────────────────
  function loadSceneFromText(text){
    let obj;
    try{obj=JSON.parse(text);}catch(err){console.warn("SphereQL: scene is not valid JSON:",err);flashButton("open-scene","✗ not JSON");return;}
    let sc;
    try{sc=parseScene(obj);}catch(err){console.warn("SphereQL: not a Scene:",err.message);flashButton("open-scene","✗ "+err.message,2200);return;}
    rebuild(sc);
    flashButton("open-scene","✓ "+sc.points.length+" points");
  }
  function loadSceneFromFile(f){if(!f)return;const r=new FileReader();r.onload=()=>loadSceneFromText(r.result);r.onerror=()=>flashButton("open-scene","✗ read error");r.readAsText(f);}

  // ── Shareable view link (camera + settings + streaming session if connected) ──
  // Encodes the current view into #v= and copies the resulting URL. The hash is
  // the single source of truth a recipient's applyViewHash() reconstructs from:
  // {cam, set:currentSettings(), tools:{ruler}} always, plus {server, filter,
  // selRow} when streaming.
  function shareLink(){
    const state={cam:[camera.position.x,camera.position.y,camera.position.z,
                      controls.target.x,controls.target.y,controls.target.z],
                 set:currentSettings(),tools:{ruler:rulerOn}};
    if(streamStreamer){
      state.server=streamStreamer.source.base;
      const off=[..._streamFilterOff];
      const mcEl=q("mincert","mincert"),mc=mcEl?parseFloat(mcEl.value)||0:0;
      if(off.length||mc>0){state.filter={};if(off.length)state.filter.off=off;if(mc>0)state.filter.mc=mc;}
      if(_streamSelectedRow!=null)state.selRow=_streamSelectedRow;
    }
    let hash;
    try{hash=btoa(encodeURIComponent(JSON.stringify(state)));}catch(err){return;}
    try{history.replaceState(null,"","#v="+hash);}
    catch(err){try{location.hash="v="+hash;}catch(e2){}}
    if(typeof navigator!=="undefined"&&navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(location.href).catch(()=>{});
    }
  }

  // ── Restore a view from the URL hash (called once after the initial rebuild) ──
  // Reads only validated numbers + known keys, so an attacker-controllable hash
  // has no injection surface (nothing decoded reaches the DOM/innerHTML).
  // Returns a Promise ONLY for a streaming session (state.server present) so
  // callers/tests can await full restoration; returns undefined for the offline
  // path. The async restore is no-op-safe if the viewer is disposed before
  // connectToServer resolves (aniRunning is cleared by dispose()).
  function applyViewHash(){
    if(typeof location==="undefined"||!location.hash)return;
    const m=location.hash.match(/[#&]v=([^&]+)/);if(!m)return;
    let state;
    try{state=JSON.parse(decodeURIComponent(atob(m[1])));}
    catch(err){console.warn("SphereQL: ignoring malformed view hash");return;}
    if(!state||typeof state!=="object")return;
    const validCam=c=>Array.isArray(c)&&c.length===6&&c.every(v=>isFinite(+v));
    const restoreCam=()=>{ if(!validCam(state.cam))return;
      camera.position.set(+state.cam[0],+state.cam[1],+state.cam[2]);
      controls.target.set(+state.cam[3],+state.cam[4],+state.cam[5]);
      controls.update();
    };
    // Settings + tools restore. applySettings validates each field and never
    // touches the camera, so order vs restoreCam is immaterial. Older/empty
    // hashes (set:{}, tools:{}) are no-ops.
    const restoreSettings=()=>{
      if(state.set&&typeof state.set==="object")applySettings(state.set);
      if(state.tools&&typeof state.tools==="object"&&state.tools.ruler)setRuler(true);
    };
    // ── Streaming session ──────────────────────────────────────────────────
    // connectToServer() runs rebuild()→frameCamera() internally, so cam/filter/
    // selRow are applied AFTER it resolves (otherwise frameCamera overwrites cam).
    if(typeof state.server==="string"&&state.server){
      return connectToServer(state.server).then(()=>{
        if(!aniRunning)return;              // disposed mid-connect → no-op
        if(!streamStreamer)return;          // connect succeeded but no live stream
        restoreSettings();
        restoreCam();
        if(streamStreamer&&_streamOnMove)_streamOnMove(); // re-request tiles for the restored viewport
        // Filter: cats are NAME strings (not indices); mc ∈ (0,1].
        if(state.filter&&typeof state.filter==="object"){
          const off=Array.isArray(state.filter.off)
            ?state.filter.off.filter(n=>typeof n==="string"):[];
          _streamFilterOff=new Set(off);
          const mc=+state.filter.mc;
          const mcEl=q("mincert","mincert");
          if(mcEl&&isFinite(mc)&&mc>0&&mc<=1){
            mcEl.value=mc;
            const mcv=q("mincert-val","mincert-val");if(mcv)mcv.textContent=mc.toFixed(2);
          }
          applyStreamFilter(); // reads the slider, translates names→indices, calls setFilter
        }
        // Selected row: 0 is a valid row id — guard with !=null + Number.isInteger.
        if(state.selRow!=null&&Number.isInteger(state.selRow)&&streamStreamer){
          selectStreamRow(state.selRow);
        }
      });
    }
    // ── Offline path (synchronous: settings + camera) ──────────────────────
    restoreSettings();
    restoreCam();
  }

  // ── Event listeners ──────────────────────────────────────────────────────
  const ac=new AbortController();
  const sig={signal:ac.signal};
  // Teardown hooks for listeners that can't take an AbortSignal (e.g. THREE's
  // EventDispatcher). dispose() runs these in addition to ac.abort().
  const cleanups=[];

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
  window.addEventListener("keydown",e=>{if(e.key!=="Escape")return;if(pinOn){setPinMode(false);return;}if(rulerOn&&rulerPicks.length){clearRuler();return;}if(selectedIdx>=0)deselectPoint(true);},sig);

  let _downX=0,_downY=0;
  canvas.addEventListener("pointerdown",e=>{_downX=e.clientX;_downY=e.clientY;},sig);
  canvas.addEventListener("pointerup",e=>{
    if(Math.hypot(e.clientX-_downX,e.clientY-_downY)>=5)return;
    if(pinOn){ // pin mode: drop a marker where the ray meets the globe shell
      const rect=canvas.getBoundingClientRect();
      mouse.x=((e.clientX-rect.left)/rect.width)*2-1;mouse.y=-((e.clientY-rect.top)/rect.height)*2+1;
      raycaster.setFromCamera(mouse,camera);
      const hit=globeGroup&&raycaster.intersectObject(globeGroup,true)[0];
      if(hit){const p=hit.point,m=Math.hypot(p.x,p.y,p.z)||1;let th=Math.atan2(p.y,p.x);if(th<0)th+=2*Math.PI;addPin(th,Math.acos(clamp(p.z/m,-1,1)));}
      return;
    }
    const idx=getHovered(e);
    if(streamStreamer){ // streaming: idx is a GLOBAL row → inspect via the server
      if(rulerOn){if(_streamHoverPos)rulerAddPick(_streamHoverPos);return;} // ruler snaps to the hovered streamed point
      if(idx>=0)selectStreamRow(idx);
      else{const info=q("info","info");if(info)info.classList.remove("visible");}
      return;
    }
    if(rulerOn){if(idx>=0)rulerAddPick(curPos(idx));return;} // ruler snaps to data points
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

  // Share-link button (if present in rootEl).
  (function(){ const sh=q("share","tool-share"); if(sh)sh.addEventListener("click",shareLink,sig); })();

  // Server connect row (Settings tab): the manual entry point to streaming.
  // Toggles connect ↔ disconnect; reads the URL from the #server-url input. The
  // button LABEL is owned by connectToServer/disconnectServer; this wires its
  // CLICK. (The server's inject_auto_connect and the #server= hash are the other
  // two entry points; all route through the same connectToServer.)
  (function(){
    const btn=q("server-connect","server-connect");if(!btn)return;
    btn.addEventListener("click",()=>{
      if(streamStreamer){disconnectServer();return;}
      const inp=q("server-url","server-url");
      const url=inp&&inp.value?String(inp.value).trim():"";
      if(!url)return;
      connectToServer(url).catch(e=>console.warn("SphereQL: connect failed",e));
    },sig);
  })();

  // ── Settings pane + header tools + scene-load + search wiring ────────────
  // Offline chrome. Every control is `if(el)`-guarded; all listeners use `,sig`
  // so dispose() removes them.
  (function(){
    if(scaleS)scaleS.addEventListener("input",e=>{const s=parseFloat(e.target.value);if(scaleVal)scaleVal.textContent=s.toFixed(1)+"×";applyScale(s);},sig);
    if(zoomS)zoomS.addEventListener("input",e=>{const v=parseFloat(e.target.value);if(zoomVal)zoomVal.textContent=v.toFixed(2)+"×";controls.zoomSpeed=v;},sig);
    if(radialS)radialS.addEventListener("input",e=>{radialG=parseFloat(e.target.value);if(radialVal)radialVal.textContent=radialG.toFixed(1)+"×";pendingTransform=true;},sig);
    if(spreadS)spreadS.addEventListener("input",e=>{spreadF=parseFloat(e.target.value);if(spreadVal)spreadVal.textContent=spreadF.toFixed(1)+"×";pendingTransform=true;},sig);
    if(psizeS)psizeS.addEventListener("input",e=>{const v=parseFloat(e.target.value);if(sizeVal)sizeVal.textContent=v.toFixed(1);applySize(v);},sig);
    if(uiS)uiS.addEventListener("input",e=>{uiScale=parseFloat(e.target.value);if(typeof document!=="undefined"&&document.documentElement)document.documentElement.style.setProperty("--ui",uiScale);if(uiVal)uiVal.textContent=uiScale.toFixed(2)+"×";},sig);
    if(schemeSel)schemeSel.addEventListener("change",e=>applyPalette(e.target.value),sig);
    if(globeCb)globeCb.addEventListener("change",e=>{if(globeGroup)globeGroup.visible=e.target.checked;},sig);
    if(autorotCb)autorotCb.addEventListener("change",e=>{controls.autoRotate=e.target.checked;},sig);
    if(densityCb)densityCb.addEventListener("change",e=>{if(pointsMat)pointsMat.uniforms.densityOn.value=e.target.checked?1:0;},sig);
    {const r=q("reset","reset");if(r)r.addEventListener("click",resetDefaults,sig);}
    {const h=q("hud-toggle","hud-toggle");if(h)h.addEventListener("click",()=>{if(typeof document!=="undefined"&&document.body)document.body.classList.toggle("hud-hidden");},sig);}
    // Tools: ruler / PNG / pins (share is wired above).
    {const tr=q("tool-ruler","tool-ruler");if(tr)tr.addEventListener("click",()=>setRuler(!rulerOn),sig);}
    {const tp=q("tool-png","tool-png");if(tp)tp.addEventListener("click",exportPNG,sig);}
    {const tpin=q("tool-pin","tool-pin");if(tpin)tpin.addEventListener("click",()=>setPinMode(!pinOn),sig);}
    // Config save / load (.toml).
    {const scb=q("save-cfg","save-cfg");if(scb)scb.addEventListener("click",()=>{
      try{const blob=new Blob([toToml(currentSettings())],{type:"text/plain"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="sphereql-view.toml";a.click();if(a.href&&URL.revokeObjectURL)URL.revokeObjectURL(a.href);}catch(err){console.warn("SphereQL: save failed",err);}
    },sig);}
    const cfgFile=q("cfg-file","cfg-file");
    {const lc=q("load-cfg","load-cfg");if(lc&&cfgFile)lc.addEventListener("click",()=>cfgFile.click(),sig);}
    if(cfgFile)cfgFile.addEventListener("change",e=>{const f=e.target.files&&e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{try{applySettings(parseToml(r.result));}catch(err){console.warn("SphereQL: failed to load settings:",err);flashButton("load-cfg","✗ bad file");}};r.readAsText(f);},sig);
    // Open Scene JSON.
    const sceneFile=q("scene-file","scene-file");
    {const os=q("open-scene","open-scene");if(os&&sceneFile)os.addEventListener("click",()=>sceneFile.click(),sig);}
    if(sceneFile)sceneFile.addEventListener("change",e=>{loadSceneFromFile(e.target.files&&e.target.files[0]);sceneFile.value="";},sig);
    // Search: highlight matching labels, dim the rest.
    if(searchInput)searchInput.addEventListener("input",()=>{
      if(!pointsGeo)return;
      const query=searchInput.value.toLowerCase(),sa=pointsGeo.getAttribute("size").array,ca=pointsGeo.getAttribute("color").array;
      if(!query){deselectPoint();return;}
      for(let i=0;i<N;i++){const match=(pts[i].label||"").toLowerCase().includes(query),vis=catVisible[pts[i].cat];
        sa[i]=(match&&vis)?baseSize*1.5:(vis?baseSize*0.45:0);const c=new THREE.Color(catColor[pts[i].cat]),f=match?1:0.25;
        ca[i*3]=c.r*f;ca[i*3+1]=c.g*f;ca[i*3+2]=c.b*f;}
      pointsGeo.getAttribute("size").needsUpdate=true;pointsGeo.getAttribute("color").needsUpdate=true;if(pointsMat)pointsMat.uniforms.opacity.value=0.55;
    },sig);
  })();

  // File drag-drop anywhere on the window loads a Scene JSON (offline path).
  (function(){
    const isFileDrag=e=>e.dataTransfer&&Array.from(e.dataTransfer.types||[]).indexOf("Files")>=0;
    let _dragDepth=0;
    window.addEventListener("dragenter",e=>{if(!isFileDrag(e))return;e.preventDefault();_dragDepth++;if(dropzone)dropzone.classList.add("on");},sig);
    window.addEventListener("dragover",e=>{if(!isFileDrag(e))return;e.preventDefault();if(e.dataTransfer)e.dataTransfer.dropEffect="copy";},sig);
    window.addEventListener("dragleave",e=>{if(!isFileDrag(e))return;_dragDepth=Math.max(0,_dragDepth-1);if(_dragDepth===0&&dropzone)dropzone.classList.remove("on");},sig);
    window.addEventListener("drop",e=>{e.preventDefault();_dragDepth=0;if(dropzone)dropzone.classList.remove("on");const f=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0];if(f)loadSceneFromFile(f);},sig);
  })();

  // Tab bar (Domains / Overlays / Settings / Diag). The diag tab hosts the
  // streaming debugger dashboard; without this it is unreachable. Pure DOM, so
  // the ,sig binding is the only teardown needed.
  (function(){
    const tabs=[...rootEl.querySelectorAll(".tab[data-tab]")];
    if(!tabs.length)return;
    const panes=new Map();
    for(const t of tabs){const p=rootEl.querySelector("#tab-"+t.dataset.tab);if(p)panes.set(t,p);}
    const activate=t=>{
      for(const [tab,pane] of panes){
        const on=tab===t;
        tab.classList.toggle("active",on);
        pane.classList.toggle("active",on);
      }
    };
    for(const t of tabs)t.addEventListener("click",()=>activate(t),sig);
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
    // Hover reticle. In streaming mode `origPos` is empty and hoveredIdx is a
    // global row, so use the tile-space position pickStreamCPU stashed.
    const hp=streamStreamer?_streamHoverPos:(hoveredIdx>=0?curPos(hoveredIdx):null);
    if(hp){const sp=projectToScreen(hp);if(sp.vis){if(reticle){reticle.style.display="block";reticle.style.left=sp.x+"px";reticle.style.top=sp.y+"px";}}else if(reticle)reticle.style.display="none";}
    else if(reticle)reticle.style.display="none";
    // Selected-point floating label.
    if(selectedIdx>=0&&sellabel){const sp=projectToScreen(curPos(selectedIdx));if(sp.vis){sellabel.style.display="block";sellabel.style.left=sp.x+"px";sellabel.style.top=(sp.y-16)+"px";}else sellabel.style.display="none";}
    updateLabels();
    updatePinLabels();
    renderer.render(scene,camera);
  }
  animate();

  // ── dispose ──────────────────────────────────────────────────────────────
  function dispose(){
    aniRunning=false;cancelAnimationFrame(aniHandle);
    if(ro)ro.disconnect();
    disconnectServer(); // release any active stream first (group + reproject hook)
    ac.abort(); // removes all our addEventListener(...,{signal}) handlers
    for(const fn of cleanups){try{fn();}catch(e){}} // non-signal listeners (e.g. embed)
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
    // Named so dispose() can remove it — THREE's EventDispatcher has no
    // AbortSignal support, so without the cleanup hook this handler outlives
    // dispose() and keeps posting cam updates from a torn-down viewer.
    const onCamChange=()=>{
      if(applying)return;
      const s=camState();
      if(drift(s,lastSent)<eps())return;
      lastSent=s;
      try{parent.postMessage({type:"sphereql-cam",s},"*");}catch(err){}
    };
    controls.addEventListener("change",onCamChange);
    cleanups.push(()=>controls.removeEventListener("change",onCamChange));
    // ,sig binds this to the AbortController so dispose() removes it; the
    // aniRunning guard neutralizes any message already in flight at dispose time
    // (without it, a post-dispose `sphereql-scene` would rebuild() a disposed GL
    // context, and `sphereql-cam`/`-lock` would mutate dead controls).
    window.addEventListener("message",e=>{
      if(!aniRunning)return;
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
    },sig);
    try{parent.postMessage({type:"sphereql-embed-ready"},"*");}catch(err){}
  })();

  // Test-only seam: when a caller passes opts.expose, hand it live getters/
  // setters for the closure-local state the headless suites assert on. No-op in
  // production (the auto-boot and studio never pass it). This is the shared
  // backbone the js-tests harness reads.
  if(typeof opts.expose==="function"){
    opts.expose({
      // render core
      get N(){return N;}, get pts(){return pts;}, get catSet(){return catSet;},
      get SR(){return SR;}, get catDir(){return catDir;},
      get overlayKinds(){return [...overlayKinds];},
      get curScale(){return curScale;}, get radialG(){return radialG;}, get spreadF(){return spreadF;},
      set radialG(v){radialG=v;}, set spreadF(v){spreadF=v;},
      get uniforms(){return pointsMat?pointsMat.uniforms:null;},
      get queryGroup(){return queryGroup;}, get chainGroup(){return chainGroup;},
      get idCount(){return idToIndex.size;}, get morphT(){return morphT;},
      get controls(){return controls;}, get zoomLocked(){return zoomLocked;},
      get densityArr(){return pointsGeo&&pointsGeo.getAttribute("density")&&pointsGeo.getAttribute("density").array;},
      get densityOn(){return pointsMat?pointsMat.uniforms.densityOn.value:0;},
      curPos,applyTransform,applyScale,pickEncode,pickDecode,getHovered,
      // offline tools (ruler / pins / PNG / TOML / palette) — factory-internal,
      // surfaced for the headless suites.
      setRuler,rulerAddPick,exportPNG,setPinMode,addPin,clearPins,
      currentSettings,applySettings,applyPalette,
      get rulerOn(){return rulerOn;}, get rulerPicks(){return rulerPicks;}, get rulerLast(){return rulerLast;},
      get rulerGroup(){return rulerGroup;}, get pins(){return pins;}, get pinOn(){return pinOn;}, get pinGroup(){return pinGroup;},
      get uiScale(){return uiScale;},
      // streaming internals (instance fields above)
      get dataSource(){return dataSource;},
      get streamStreamer(){return streamStreamer;},
      get _streamFilterOff(){return _streamFilterOff;},
      get _streamSelectedRow(){return _streamSelectedRow;},
    });
  }
  return{rebuild,updateScene,drawChain,highlightByIds,setMorphTarget,applyMorph,clearMorph,dispose,camera,applyViewHash,shareLink,
    connectToServer,disconnectServer,selectStreamRow,renderDiagnostics,buildStreamLegend,applyStreamFilter,renderVectorSparkline,
    get streamStreamer(){return streamStreamer;}};
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
  // Streaming entry points — the server's inject_auto_connect calls
  // connectToServer(url) at top level; studio.js reads window.__sqServerReproject.
  window.connectToServer=(url,opts)=>viewer.connectToServer(url,opts);
  window.disconnectServer=()=>viewer.disconnectServer();
  viewer.rebuild(D);
  // applyViewHash returns a Promise for a streaming #v= session; swallow a
  // rejection (e.g. a shared hash pointing at a dead server) so it can't surface
  // as an unhandled rejection. No-op for the synchronous offline / no-hash path.
  Promise.resolve(viewer.applyViewHash()).catch(err=>console.warn("SphereQL: view-hash restore failed",err));
  // #server=<url> boots straight into streaming (offline-by-default: only a
  // #server= or #v= streaming hash makes a network request).
  {
    const m=(location.hash||"").match(/[#&]server=([^&]+)/);
    if(m){viewer.connectToServer(decodeURIComponent(m[1])).catch(e=>console.warn("SphereQL: #server connect failed",e));}
  }
}
