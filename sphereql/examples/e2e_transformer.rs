use std::collections::HashMap;

use sphereql::core::spherical_to_cartesian;
use sphereql::embed::{
    ConceptPath, Embedding, EmbeddingIndex, GlobResult, PcaProjection, Projection, RadialStrategy,
    SlicingManifold,
};

struct Sentence {
    id: String,
    text: String,
    category: String,
    embedding: Embedding,
}

struct ProjectedPoint {
    id: String,
    text: String,
    category: String,
    r: f64,
    x: f64,
    y: f64,
    z: f64,
    neighbors: Vec<String>,
    slice_dist: f64,
    slice_u: f64,
    slice_v: f64,
}

fn category_color(cat: &str) -> &'static str {
    match cat {
        "science" => "#2196F3",
        "technology" => "#9C27B0",
        "sports" => "#FF5722",
        "cooking" => "#FF9800",
        "arts" => "#4CAF50",
        "nature" => "#009688",
        "history" => "#795548",
        "health" => "#E91E63",
        "philosophy" => "#607D8B",
        "business" => "#FFC107",
        _ => "#FFFFFF",
    }
}

fn main() {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sphereql-embed/tools/embeddings.json".into());
    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "sphere_viz.html".into());

    // ── 1. Load embeddings ──────────────────────────────────────────────
    eprintln!("Loading embeddings from {json_path}...");
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("Cannot read {json_path}: {e}"));
    let data: serde_json::Value = serde_json::from_str(&raw).expect("Invalid JSON");

    let dim = data["dimension"].as_u64().unwrap() as usize;
    let model_name = data["model"].as_str().unwrap_or("unknown");
    let arr = data["sentences"].as_array().expect("missing sentences");

    let sentences: Vec<Sentence> = arr
        .iter()
        .map(|s| {
            let values: Vec<f64> = s["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            Sentence {
                id: s["id"].as_str().unwrap().into(),
                text: s["text"].as_str().unwrap().into(),
                category: s["category"].as_str().unwrap().into(),
                embedding: Embedding::new(values),
            }
        })
        .collect();

    eprintln!(
        "Loaded {} sentences ({dim}-d, model: {model_name})",
        sentences.len()
    );

    // ── 2. Fit volumetric PCA projection ────────────────────────────────
    eprintln!("Fitting volumetric PCA projection ({dim}-d → 3D volume)...");
    let all_emb: Vec<Embedding> = sentences.iter().map(|s| s.embedding.clone()).collect();
    let pca = PcaProjection::fit(&all_emb, RadialStrategy::Magnitude).with_volumetric(true);

    let mut index = EmbeddingIndex::builder(pca.clone())
        .uniform_shells(10, 1.0)
        .theta_divisions(12)
        .phi_divisions(6)
        .build();

    for s in &sentences {
        index.insert(&s.id, &s.embedding);
    }

    // ── 3. Project all points → Cartesian for manifold analysis ─────────
    let k_neighbors = 5;
    let mut cart_points: Vec<[f64; 3]> = Vec::with_capacity(sentences.len());
    let mut projected: Vec<ProjectedPoint> = Vec::with_capacity(sentences.len());

    for s in &sentences {
        let sp = pca.project(&s.embedding);
        let c = spherical_to_cartesian(&sp);
        cart_points.push([c.x, c.y, c.z]);

        let nearest = index.search_nearest(&s.embedding, k_neighbors + 1);
        let neighbors: Vec<String> = nearest
            .iter()
            .filter(|r| r.item.id != s.id)
            .take(k_neighbors)
            .map(|r| r.item.id.clone())
            .collect();

        projected.push(ProjectedPoint {
            id: s.id.clone(),
            text: s.text.clone(),
            category: s.category.clone(),
            r: sp.r,
            x: c.x,
            y: c.y,
            z: c.z,
            neighbors,
            slice_dist: 0.0,
            slice_u: 0.0,
            slice_v: 0.0,
        });
    }

    // ── 4. Fit slicing manifold ─────────────────────────────────────────
    eprintln!("Fitting optimal slicing manifold...");
    let manifold = SlicingManifold::fit(&cart_points);

    for (i, p) in projected.iter_mut().enumerate() {
        p.slice_dist = manifold.distance(&cart_points[i]);
        let (u, v) = manifold.project_2d(&cart_points[i]);
        p.slice_u = u;
        p.slice_v = v;
    }

    // ── 5. Concept paths ────────────────────────────────────────────────
    eprintln!("Computing concept paths...");

    let path_pairs = [
        ("sci-04", "cook-15"),   // Einstein → Maillard reaction
        ("phil-06", "tech-03"),  // Mind-body problem → AI text generation
        ("sport-02", "health-01"), // Marathon → cardiovascular exercise
        ("nat-01", "biz-12"),    // Amazon rainforest → corporate social responsibility
    ];

    let mut paths: Vec<(String, String, ConceptPath)> = Vec::new();
    for &(src, tgt) in &path_pairs {
        if let Some(path) = index.concept_path(src, tgt, 8) {
            paths.push((src.into(), tgt.into(), path));
        }
    }

    // ── 6. Queries ──────────────────────────────────────────────────────
    eprintln!("Running queries...");

    let json_queries = data["queries"].as_array();
    let queries: Vec<(&str, Embedding)> = if let Some(jq) = json_queries {
        jq.iter()
            .map(|q| {
                let text = q["text"].as_str().unwrap();
                let values: Vec<f64> = q["embedding"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                (text, Embedding::new(values))
            })
            .collect()
    } else {
        Vec::new()
    };

    struct QueryResult {
        description: String,
        x: f64,
        y: f64,
        z: f64,
        hits: Vec<(String, f64, String)>,
        local_manifold: SlicingManifold,
    }

    impl AsQueryResult for QueryResult {
        fn as_qr(&self) -> QR<'_> {
            QR {
                desc: &self.description,
                x: self.x,
                y: self.y,
                z: self.z,
                hits: &self.hits,
                lm: &self.local_manifold,
            }
        }
    }

    let mut query_results: Vec<QueryResult> = Vec::new();
    for (desc, emb) in &queries {
        let results = index.search_nearest(emb, 5);
        let sp = pca.project(emb);
        let c = spherical_to_cartesian(&sp);
        let qpt = [c.x, c.y, c.z];
        let local_manifold = SlicingManifold::fit_local(&qpt, &cart_points, 20);
        query_results.push(QueryResult {
            description: desc.to_string(),
            x: c.x,
            y: c.y,
            z: c.z,
            hits: results
                .iter()
                .map(|r| {
                    let cat = sentences
                        .iter()
                        .find(|s| s.id == r.item.id)
                        .map(|s| s.category.clone())
                        .unwrap_or_default();
                    (r.item.id.clone(), r.distance, cat)
                })
                .collect(),
            local_manifold,
        });
    }

    // ── 6b. Concept Globs ───────────────────────────────────────────────
    eprintln!("Detecting concept globs...");
    let all_ids: Vec<String> = sentences.iter().map(|s| s.id.clone()).collect();
    let glob_result = GlobResult::detect(&cart_points, &all_ids, None, 15);

    // ── 7. Terminal output ──────────────────────────────────────────────
    println!("=== SphereQL: End-to-End Transformer Pipeline ===\n");
    println!(
        "Model: {model_name}  |  Corpus: {} sentences  |  Dim: {dim} → 3D (volumetric)\n",
        sentences.len()
    );

    let mut cat_counts: HashMap<&str, usize> = HashMap::new();
    for s in &sentences {
        *cat_counts.entry(&s.category).or_default() += 1;
    }
    println!("Categories:");
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (cat, count) in &cats {
        println!("  {:<14} {} docs  {}", cat, count, category_color(cat));
    }

    println!(
        "\nSlicing manifold: {:.1}% variance captured in 2D plane",
        manifold.variance_ratio * 100.0
    );

    println!(
        "\nConcept Globs: auto-detected k={} (silhouette={:.3})",
        glob_result.k, glob_result.silhouette,
    );
    for g in &glob_result.globs {
        let mut glob_cats: HashMap<&str, usize> = HashMap::new();
        for mid in &g.member_ids {
            if let Some(s) = sentences.iter().find(|s| s.id == *mid) {
                *glob_cats.entry(&s.category).or_default() += 1;
            }
        }
        let mut gc: Vec<_> = glob_cats.iter().collect();
        gc.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        let top: String = gc.iter().take(3).map(|(c, n)| format!("{c}({n})")).collect::<Vec<_>>().join(", ");
        println!("  Glob {:>2}: {:>3} members, radius={:.4}, top: {}", g.id, g.member_ids.len(), g.radius, top);
    }

    for (src, tgt, path) in &paths {
        let src_text = sentences.iter().find(|s| s.id == *src).map(|s| truncate(&s.text, 40)).unwrap_or_default();
        let tgt_text = sentences.iter().find(|s| s.id == *tgt).map(|s| truncate(&s.text, 40)).unwrap_or_default();
        println!(
            "\n--- Concept Path: \"{}\" → \"{}\" ({} hops, {:.4} rad total) ---",
            src_text,
            tgt_text,
            path.steps.len() - 1,
            path.total_distance
        );
        for (i, step) in path.steps.iter().enumerate() {
            let cat = sentences.iter().find(|s| s.id == step.id).map(|s| s.category.as_str()).unwrap_or("?");
            let text = sentences.iter().find(|s| s.id == step.id).map(|s| truncate(&s.text, 50)).unwrap_or_default();
            println!(
                "  {:>2}. [{:<12}] cum={:.4}  \"{}\"",
                i, cat, step.cumulative_distance, text
            );
        }
    }

    for qr in &query_results {
        println!("\n--- Query: \"{}\" ---", qr.description);
        for (i, (id, dist, cat)) in qr.hits.iter().enumerate() {
            let text = sentences
                .iter()
                .find(|s| s.id == *id)
                .map(|s| truncate(&s.text, 55))
                .unwrap_or_default();
            println!(
                "  {}. [{:<12}] {:.4} rad ({:>6.2}°)  \"{}\"",
                i + 1,
                cat,
                dist,
                dist.to_degrees(),
                text,
            );
        }
    }

    // ── 8. Generate visualization ───────────────────────────────────────
    eprintln!("\nGenerating visualization...");
    let data_json = build_viz_json(&projected, &query_results, &paths, &manifold, &glob_result, &sentences);
    let html = VIZ_TEMPLATE.replace("__DATA_PLACEHOLDER__", &data_json);
    std::fs::write(&output_path, &html)
        .unwrap_or_else(|e| panic!("Cannot write {output_path}: {e}"));
    eprintln!("Wrote {output_path} ({} bytes)", html.len());
    println!("\n✓ Visualization: {output_path}");
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn build_viz_json(
    points: &[ProjectedPoint],
    queries: &[impl AsQueryResult],
    paths: &[(String, String, ConceptPath)],
    manifold: &SlicingManifold,
    globs: &GlobResult,
    sentences: &[Sentence],
) -> String {
    let pts: Vec<String> = points
        .iter()
        .map(|p| {
            let nb = p
                .neighbors
                .iter()
                .map(|n| format!("\"{}\"", n))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"id\":\"{}\",\"text\":\"{}\",\"cat\":\"{}\",\"x\":{:.6},\"y\":{:.6},\"z\":{:.6},\"r\":{:.4},\"sd\":{:.6},\"su\":{:.6},\"sv\":{:.6},\"n\":[{}]}}",
                p.id, esc(&p.text), p.category, p.x, p.y, p.z, p.r, p.slice_dist, p.slice_u, p.slice_v, nb
            )
        })
        .collect();

    let qs: Vec<String> = queries
        .iter()
        .map(|q| {
            let qr = q.as_qr();
            let hits: Vec<String> = qr
                .hits
                .iter()
                .map(|(id, d, cat)| format!("{{\"id\":\"{id}\",\"d\":{d:.4},\"cat\":\"{cat}\"}}", ))
                .collect();
            let lm = qr.lm;
            format!(
                "{{\"desc\":\"{}\",\"x\":{:.6},\"y\":{:.6},\"z\":{:.6},\"hits\":[{}],\"lm\":{{\"cx\":{:.6},\"cy\":{:.6},\"cz\":{:.6},\"nx\":{:.6},\"ny\":{:.6},\"nz\":{:.6},\"vr\":{:.4}}}}}",
                esc(qr.desc), qr.x, qr.y, qr.z, hits.join(","),
                lm.centroid[0], lm.centroid[1], lm.centroid[2],
                lm.normal[0], lm.normal[1], lm.normal[2],
                lm.variance_ratio,
            )
        })
        .collect();

    let ps: Vec<String> = paths
        .iter()
        .map(|(src, tgt, path)| {
            let steps: Vec<String> = path
                .steps
                .iter()
                .map(|s| format!("\"{}\"", s.id))
                .collect();
            let src_text = sentences.iter().find(|s| s.id == *src).map(|s| esc(&s.text)).unwrap_or_default();
            let tgt_text = sentences.iter().find(|s| s.id == *tgt).map(|s| esc(&s.text)).unwrap_or_default();
            format!(
                "{{\"src\":\"{src}\",\"tgt\":\"{tgt}\",\"srcText\":\"{src_text}\",\"tgtText\":\"{tgt_text}\",\"dist\":{:.4},\"steps\":[{}]}}",
                path.total_distance,
                steps.join(",")
            )
        })
        .collect();

    let mf = format!(
        "{{\"cx\":{:.6},\"cy\":{:.6},\"cz\":{:.6},\"nx\":{:.6},\"ny\":{:.6},\"nz\":{:.6},\"ux\":{:.6},\"uy\":{:.6},\"uz\":{:.6},\"vx\":{:.6},\"vy\":{:.6},\"vz\":{:.6},\"vr\":{:.4}}}",
        manifold.centroid[0], manifold.centroid[1], manifold.centroid[2],
        manifold.normal[0], manifold.normal[1], manifold.normal[2],
        manifold.basis_u[0], manifold.basis_u[1], manifold.basis_u[2],
        manifold.basis_v[0], manifold.basis_v[1], manifold.basis_v[2],
        manifold.variance_ratio,
    );

    let gs: Vec<String> = globs.globs.iter().map(|g| {
        let members = g.member_ids.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(",");
        format!(
            "{{\"id\":{},\"cx\":{:.6},\"cy\":{:.6},\"cz\":{:.6},\"r\":{:.4},\"members\":[{}]}}",
            g.id, g.centroid[0], g.centroid[1], g.centroid[2], g.radius, members
        )
    }).collect();

    format!(
        "{{\"points\":[{}],\"queries\":[{}],\"paths\":[{}],\"manifold\":{},\"globs\":[{}],\"globK\":{},\"globSil\":{:.4}}}",
        pts.join(","),
        qs.join(","),
        ps.join(","),
        mf,
        gs.join(","),
        globs.k,
        globs.silhouette,
    )
}

// Trait to generalize over the query result type without exposing internals
struct QR<'a> {
    desc: &'a str,
    x: f64,
    y: f64,
    z: f64,
    hits: &'a [(String, f64, String)],
    lm: &'a SlicingManifold,
}

trait AsQueryResult {
    fn as_qr(&self) -> QR<'_>;
}

// ── Visualization HTML ──────────────────────────────────────────────────

const VIZ_TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SphereQL — Semantic Embedding Visualization</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#e0e0e0;font-family:system-ui,-apple-system,sans-serif;overflow:hidden}
#c{position:absolute;top:0;left:0}
#legend{position:absolute;top:16px;left:16px;background:rgba(10,10,30,0.88);padding:14px 18px;border-radius:10px;font-size:13px;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);max-height:45vh;overflow-y:auto}
#legend h3{font-size:14px;margin-bottom:8px;color:#aaa;font-weight:500;letter-spacing:.5px}
.lrow{display:flex;align-items:center;gap:8px;margin:4px 0;cursor:pointer;padding:2px 4px;border-radius:4px;transition:background .15s}
.lrow:hover{background:rgba(255,255,255,0.08)}.lrow.dim{opacity:.25}
.ldot{width:10px;height:10px;border-radius:50%;flex-shrink:0}.lbl{font-size:12px}.lcnt{font-size:11px;color:#888;margin-left:auto}
#info{position:absolute;top:16px;right:16px;width:360px;max-height:calc(100vh - 32px);overflow-y:auto;background:rgba(10,10,30,0.88);padding:18px;border-radius:10px;font-size:13px;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);display:none}
#info.visible{display:block}
#info h3{font-size:15px;margin-bottom:6px;font-weight:600}
#info .cat-tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500;margin-bottom:8px}
#info .txt{font-size:13px;line-height:1.5;color:#ccc;margin-bottom:12px}
#info .nb{padding:6px 8px;margin:3px 0;background:rgba(255,255,255,0.04);border-radius:6px;font-size:12px;cursor:pointer;transition:background .15s}
#info .nb:hover{background:rgba(255,255,255,0.1)}
#info .nb .dist{color:#888;font-size:11px;float:right}
#controls{position:absolute;bottom:16px;left:16px;background:rgba(10,10,30,0.88);padding:14px 18px;border-radius:10px;font-size:13px;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);max-width:460px;max-height:45vh;overflow-y:auto}
#controls h3{font-size:14px;margin-bottom:8px;color:#aaa;font-weight:500;letter-spacing:.5px}
.qbtn,.pbtn{display:block;padding:6px 10px;margin:4px 0;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:#ddd;cursor:pointer;font-size:12px;text-align:left;width:100%;transition:background .15s}
.qbtn:hover,.pbtn:hover{background:rgba(255,255,255,0.12)}
.qbtn.active{background:rgba(100,180,255,0.15);border-color:rgba(100,180,255,0.3)}
.pbtn.active{background:rgba(255,160,60,0.18);border-color:rgba(255,160,60,0.35)}
.tog{display:flex;align-items:center;gap:8px;margin:6px 0;cursor:pointer;font-size:12px;color:#bbb}
.tog input{accent-color:#4CAF50}
#help{position:absolute;bottom:16px;right:16px;font-size:11px;color:#555;text-align:right;pointer-events:none}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="legend"></div>
<div id="info"></div>
<div id="controls"></div>
<div id="help">drag to rotate · scroll to zoom · click a point to inspect</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.163.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.163.0/examples/jsm/"}}
</script>
<script type="module">
import*as THREE from'three';
import{OrbitControls}from'three/addons/controls/OrbitControls.js';
const D=__DATA_PLACEHOLDER__;
const COLORS={science:'#2196F3',technology:'#9C27B0',sports:'#FF5722',cooking:'#FF9800',arts:'#4CAF50',nature:'#009688',history:'#795548',health:'#E91E63',philosophy:'#607D8B',business:'#FFC107'};
const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0a0a1a);
scene.fog=new THREE.FogExp2(0x0a0a1a,0.6);
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,100);
camera.position.set(0,0.4,1.2);
const ctrl=new OrbitControls(camera,canvas);
ctrl.enableDamping=true;ctrl.dampingFactor=0.08;ctrl.minDistance=0.3;ctrl.maxDistance=6;
scene.add(new THREE.AmbientLight(0x404060,2));
const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(3,5,4);scene.add(dl);

// Wireframe reference sphere (unit)
const wg=new THREE.IcosahedronGeometry(0.5,3);
const wm=new THREE.MeshBasicMaterial({color:0x334466,wireframe:true,transparent:true,opacity:0.06});
scene.add(new THREE.Mesh(wg,wm));

// Axes
const ag=new THREE.BufferGeometry();
const av=[];const ac=[];
[[1,0,0,1,0.3,0.3],[0,1,0,0.3,1,0.3],[0,0,1,0.3,0.3,1]].forEach(([x,y,z,r,g,b])=>{
av.push(0,0,0,x*0.6,y*0.6,z*0.6);ac.push(r,g,b,r,g,b);});
ag.setAttribute('position',new THREE.Float32BufferAttribute(av,3));
ag.setAttribute('color',new THREE.Float32BufferAttribute(ac,3));
scene.add(new THREE.LineSegments(ag,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.3})));

// Points
const pg=new THREE.Group();scene.add(pg);
const meshes=[];const idMap={};const idData={};
const sg=new THREE.SphereGeometry(0.008,10,10);
D.points.forEach(pt=>{
const col=new THREE.Color(COLORS[pt.cat]||'#fff');
const mat=new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.5,roughness:0.4});
const m=new THREE.Mesh(sg,mat);
m.position.set(pt.x,pt.y,pt.z);m.userData=pt;
pg.add(m);meshes.push(m);idMap[pt.id]=m;idData[pt.id]=pt;});

// Slice plane
const MF=D.manifold;
const sliceGrp=new THREE.Group();scene.add(sliceGrp);
let sliceVisible=false;
function buildSlicePlane(){
while(sliceGrp.children.length)sliceGrp.remove(sliceGrp.children[0]);
if(!sliceVisible)return;
const plGeo=new THREE.PlaneGeometry(1,1);
const plMat=new THREE.MeshBasicMaterial({color:0x6688aa,transparent:true,opacity:0.12,side:THREE.DoubleSide});
const pl=new THREE.Mesh(plGeo,plMat);
pl.position.set(MF.cx,MF.cy,MF.cz);
const up=new THREE.Vector3(0,0,1);
const normal=new THREE.Vector3(MF.nx,MF.ny,MF.nz).normalize();
const q=new THREE.Quaternion().setFromUnitVectors(up,normal);
pl.setRotationFromQuaternion(q);
sliceGrp.add(pl);
// Ring
const rg=new THREE.RingGeometry(0.48,0.5,64);
const rm=new THREE.MeshBasicMaterial({color:0x88aacc,transparent:true,opacity:0.25,side:THREE.DoubleSide});
const ring=new THREE.Mesh(rg,rm);
ring.position.copy(pl.position);ring.setRotationFromQuaternion(q);
sliceGrp.add(ring);
}

// Glob visualization (translucent spheres + diamond centroids)
const globGrp=new THREE.Group();scene.add(globGrp);
const GLOB_COLORS=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F','#BB8FCE','#85C1E9'];
let globsVisible=false;
function buildGlobs(){
while(globGrp.children.length)globGrp.remove(globGrp.children[0]);
if(!globsVisible||!D.globs)return;
D.globs.forEach((g,i)=>{
const col=new THREE.Color(GLOB_COLORS[i%GLOB_COLORS.length]);
// Translucent sphere at radius
const sg=new THREE.SphereGeometry(g.r,16,16);
const sm=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.08,depthWrite:false});
const sphere=new THREE.Mesh(sg,sm);
sphere.position.set(g.cx,g.cy,g.cz);
globGrp.add(sphere);
// Wireframe ring
const wg=new THREE.SphereGeometry(g.r,12,12);
const wm=new THREE.MeshBasicMaterial({color:col,wireframe:true,transparent:true,opacity:0.15});
const wire=new THREE.Mesh(wg,wm);
wire.position.set(g.cx,g.cy,g.cz);
globGrp.add(wire);
// Diamond centroid marker
const dg=new THREE.OctahedronGeometry(0.015);
const dm=new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.8});
const diamond=new THREE.Mesh(dg,dm);
diamond.position.set(g.cx,g.cy,g.cz);
diamond.userData={isGlobCenter:true,globIdx:i,glob:g};
globGrp.add(diamond);
meshes.push(diamond);
});
}

// Per-query local manifold discs
const lmGrp=new THREE.Group();scene.add(lmGrp);
const lmDiscs=[];
function buildLocalManifold(qi,visible){
if(lmDiscs[qi]){lmDiscs[qi].visible=visible;return;}
const q=D.queries[qi];if(!q||!q.lm)return;
const lm=q.lm;
const plGeo=new THREE.CircleGeometry(0.15,32);
const plMat=new THREE.MeshBasicMaterial({color:0xffff44,transparent:true,opacity:0.15,side:THREE.DoubleSide,depthWrite:false});
const disc=new THREE.Mesh(plGeo,plMat);
disc.position.set(lm.cx,lm.cy,lm.cz);
const normal=new THREE.Vector3(lm.nx,lm.ny,lm.nz).normalize();
const up=new THREE.Vector3(0,0,1);
disc.setRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(up,normal));
disc.visible=visible;
lmGrp.add(disc);
lmDiscs[qi]=disc;
// Edge ring
const rGeo=new THREE.RingGeometry(0.148,0.152,32);
const rMat=new THREE.MeshBasicMaterial({color:0xffff88,transparent:true,opacity:0.3,side:THREE.DoubleSide});
const ring=new THREE.Mesh(rGeo,rMat);
ring.position.copy(disc.position);ring.rotation.copy(disc.rotation);
ring.visible=visible;
lmGrp.add(ring);
}

// Lines group for neighbors / paths / queries
const lineGrp=new THREE.Group();scene.add(lineGrp);
const pathGrp=new THREE.Group();scene.add(pathGrp);
const queryGrp=new THREE.Group();scene.add(queryGrp);
let selectedId=null;

function clearLines(){while(lineGrp.children.length)lineGrp.remove(lineGrp.children[0]);}
function clearPaths(){while(pathGrp.children.length)pathGrp.remove(pathGrp.children[0]);}
function clearQueries(){while(queryGrp.children.length)queryGrp.remove(queryGrp.children[0]);}

function drawNeighborLines(pt){
clearLines();
const o=new THREE.Vector3(pt.x,pt.y,pt.z);
pt.n.forEach(nid=>{const nd=idData[nid];if(!nd)return;
const geo=new THREE.BufferGeometry().setFromPoints([o,new THREE.Vector3(nd.x,nd.y,nd.z)]);
lineGrp.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:0.3})));
});}

function drawPath(steps){
clearPaths();
if(steps.length<2)return;
const pts=steps.map(id=>{const d=idData[id];return d?new THREE.Vector3(d.x,d.y,d.z):null}).filter(Boolean);
// Tube along path
const curve=new THREE.CatmullRomCurve3(pts);
const tGeo=new THREE.TubeGeometry(curve,pts.length*8,0.003,6,false);
const tMat=new THREE.MeshBasicMaterial({color:0xff8800,transparent:true,opacity:0.7});
pathGrp.add(new THREE.Mesh(tGeo,tMat));
// Highlight nodes
steps.forEach(id=>{if(idMap[id])idMap[id].scale.setScalar(2.2);});
}

let globLinesShown=null;
function selectPoint(pt){
if(pt.isGlobCenter){
  // Toggle: click centroid → show lines to all members, click again → hide
  const gi=pt.globIdx;
  if(globLinesShown===gi){clearLines();globLinesShown=null;document.getElementById('info').classList.remove('visible');return;}
  clearLines();clearPaths();clearQueries();globLinesShown=gi;
  const g=pt.glob;const o=new THREE.Vector3(g.cx,g.cy,g.cz);
  g.members.forEach(mid=>{const nd=idData[mid];if(!nd)return;
  const geo=new THREE.BufferGeometry().setFromPoints([o,new THREE.Vector3(nd.x,nd.y,nd.z)]);
  lineGrp.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:GLOB_COLORS[gi%GLOB_COLORS.length],transparent:true,opacity:0.4})));});
  const el=document.getElementById('info');
  let html=`<h3>Glob ${gi}</h3><div style="font-size:12px;color:#aaa;margin-bottom:8px">${g.members.length} members · radius=${g.r.toFixed(4)}</div>`;
  g.members.forEach(mid=>{const nd=idData[mid];if(!nd)return;const nc=COLORS[nd.cat]||'#999';
  html+=`<div class="nb" onclick="window._sel('${mid}')"><span style="color:${nc}">●</span> ${nd.text.substring(0,65)}…</div>`;});
  el.innerHTML=html;el.classList.add('visible');
  return;
}
globLinesShown=null;
selectedId=pt.id;drawNeighborLines(pt);clearPaths();clearQueries();
meshes.forEach(m=>m.scale.setScalar(1));
if(idMap[pt.id])idMap[pt.id].scale.setScalar(2.5);
pt.n.forEach(nid=>{if(idMap[nid])idMap[nid].scale.setScalar(1.8);});
showInfo(pt);
document.querySelectorAll('.qbtn,.pbtn').forEach(b=>b.classList.remove('active'));
}

function showInfo(pt){
const color=COLORS[pt.cat]||'#999';
let html=`<h3>${pt.id}</h3><div class="cat-tag" style="background:${color}33;color:${color}">${pt.cat}</div>
<div class="txt">${pt.text}</div>
<div style="font-size:11px;color:#888;margin-bottom:12px">r=${pt.r.toFixed(4)} slice_dist=${pt.sd.toFixed(4)}</div>
<div class="neighbors"><h4 style="font-size:12px;color:#888;margin-bottom:6px">Nearest neighbors</h4>`;
pt.n.forEach(nid=>{const nd=idData[nid];if(!nd)return;const nc=COLORS[nd.cat]||'#999';
html+=`<div class="nb" onclick="window._sel('${nid}')"><span style="color:${nc}">●</span> ${nd.text.substring(0,70)}…<span class="dist">${nd.cat}</span></div>`;});
html+='</div>';
const el=document.getElementById('info');el.innerHTML=html;el.classList.add('visible');
}
window._sel=id=>{const pt=idData[id];if(pt)selectPoint(pt);};

// Raycaster
const rc=new THREE.Raycaster();const mouse=new THREE.Vector2();let hovered=null;
canvas.addEventListener('mousemove',e=>{mouse.x=(e.clientX/innerWidth)*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;});
canvas.addEventListener('click',()=>{rc.setFromCamera(mouse,camera);const h=rc.intersectObjects(meshes.filter(m=>m.visible));if(h.length)selectPoint(h[0].object.userData);});

// Legend
const catCounts={};D.points.forEach(p=>{catCounts[p.cat]=(catCounts[p.cat]||0)+1;});
const hiddenCats=new Set();
const legEl=document.getElementById('legend');
let legH='<h3>Categories ('+D.points.length+' sentences)</h3>';
Object.entries(COLORS).forEach(([cat,color])=>{if(!catCounts[cat])return;
legH+=`<div class="lrow" data-cat="${cat}"><div class="ldot" style="background:${color}"></div><span class="lbl">${cat}</span><span class="lcnt">${catCounts[cat]}</span></div>`;});
legEl.innerHTML=legH;
legEl.querySelectorAll('.lrow').forEach(row=>row.addEventListener('click',()=>{
const cat=row.dataset.cat;
if(hiddenCats.has(cat)){hiddenCats.delete(cat);row.classList.remove('dim');}
else{hiddenCats.add(cat);row.classList.add('dim');}
meshes.forEach(m=>{m.visible=!hiddenCats.has(m.userData.cat);});
}));

// Controls panel
const ctrlEl=document.getElementById('controls');
let cH='<h3>Analysis</h3>';
cH+=`<label class="tog"><input type="checkbox" id="sliceToggle"> Show slicing manifold (${(MF.vr*100).toFixed(1)}% variance)</label>`;
if(D.globs){cH+=`<label class="tog"><input type="checkbox" id="globToggle"> Show concept globs (k=${D.globK}, sil=${D.globSil.toFixed(3)})</label>`;}
if(D.queries.length){cH+='<h3 style="margin-top:12px">Queries</h3>';
D.queries.forEach((q,i)=>{cH+=`<div style="display:flex;align-items:center;gap:4px"><button class="qbtn" data-qi="${i}" style="flex:1">"${q.desc}"</button>${q.lm?`<label class="tog" style="margin:0;white-space:nowrap"><input type="checkbox" class="lmtog" data-qi="${i}"> manifold</label>`:''}</div>`;});}
if(D.paths.length){cH+='<h3 style="margin-top:12px">Concept Paths</h3>';
D.paths.forEach((p,i)=>{cH+=`<button class="pbtn" data-pi="${i}">${p.src} → ${p.tgt} (${p.steps.length-1} hops)</button>`;});}
ctrlEl.innerHTML=cH;

document.getElementById('sliceToggle').addEventListener('change',e=>{sliceVisible=e.target.checked;buildSlicePlane();
meshes.forEach(m=>{const pt=m.userData;if(!pt)return;if(sliceVisible&&pt.sd!==undefined){const a=Math.max(0.15,1-Math.abs(pt.sd)*8);m.material.opacity=a;m.material.transparent=true;}else{m.material.opacity=1;m.material.transparent=false;}});
});

const globToggle=document.getElementById('globToggle');
if(globToggle)globToggle.addEventListener('change',e=>{globsVisible=e.target.checked;buildGlobs();});

document.querySelectorAll('.lmtog').forEach(cb=>{cb.addEventListener('change',e=>{
const qi=parseInt(e.target.dataset.qi);buildLocalManifold(qi,e.target.checked);
});});

ctrlEl.querySelectorAll('.qbtn').forEach(btn=>btn.addEventListener('click',()=>{
const qi=parseInt(btn.dataset.qi);const q=D.queries[qi];
document.querySelectorAll('.qbtn,.pbtn').forEach(b=>b.classList.remove('active'));
btn.classList.add('active');
meshes.forEach(m=>m.scale.setScalar(1));clearLines();clearPaths();clearQueries();selectedId=null;
q.hits.forEach(h=>{if(idMap[h.id])idMap[h.id].scale.setScalar(2.2);});
// Query marker
const qm=new THREE.Mesh(new THREE.SphereGeometry(0.014,10,10),new THREE.MeshBasicMaterial({color:0xffff00}));
qm.position.set(q.x,q.y,q.z);queryGrp.add(qm);
q.hits.forEach(h=>{const nd=idData[h.id];if(!nd)return;
const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(q.x,q.y,q.z),new THREE.Vector3(nd.x,nd.y,nd.z)]);
queryGrp.add(new THREE.Line(g,new THREE.LineBasicMaterial({color:0xffff44,transparent:true,opacity:0.5})));});
const el=document.getElementById('info');
let rH=`<h3>Query Results</h3><div class="txt" style="font-style:italic">"${q.desc}"</div>`;
q.hits.forEach((h,i)=>{const nd=idData[h.id];const nc=COLORS[h.cat]||'#999';
rH+=`<div class="nb" onclick="window._sel('${h.id}')"><span style="color:${nc}">●</span> <b>${i+1}.</b> ${nd?nd.text.substring(0,60)+'…':h.id}<span class="dist">${(h.d*180/Math.PI).toFixed(1)}°</span></div>`;});
el.innerHTML=rH;el.classList.add('visible');
}));

ctrlEl.querySelectorAll('.pbtn').forEach(btn=>btn.addEventListener('click',()=>{
const pi=parseInt(btn.dataset.pi);const p=D.paths[pi];
document.querySelectorAll('.qbtn,.pbtn').forEach(b=>b.classList.remove('active'));
btn.classList.add('active');
meshes.forEach(m=>m.scale.setScalar(1));clearLines();clearQueries();
drawPath(p.steps);
const el=document.getElementById('info');
let rH=`<h3>Concept Path</h3><div class="txt">${p.srcText.substring(0,60)}… → ${p.tgtText.substring(0,60)}…</div>
<div style="font-size:11px;color:#888;margin-bottom:8px">${p.steps.length-1} hops · ${p.dist.toFixed(4)} rad total</div>`;
p.steps.forEach((id,i)=>{const nd=idData[id];const nc=nd?COLORS[nd.cat]||'#999':'#999';
rH+=`<div class="nb" onclick="window._sel('${id}')"><span style="color:${nc}">●</span> <b>${i}.</b> ${nd?nd.text.substring(0,60)+'…':id}</div>`;});
el.innerHTML=rH;el.classList.add('visible');
}));

function updateHover(){
rc.setFromCamera(mouse,camera);
const h=rc.intersectObjects(meshes.filter(m=>m.visible));
const nw=h.length?h[0].object:null;
if(hovered&&hovered!==nw&&hovered.userData.id!==selectedId)hovered.scale.setScalar(1);
if(nw&&nw.userData.id!==selectedId){nw.scale.setScalar(1.6);canvas.style.cursor='pointer';}
else canvas.style.cursor='grab';
hovered=nw;}

addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
(function animate(){requestAnimationFrame(animate);ctrl.update();updateHover();renderer.render(scene,camera);})();
</script>
</body>
</html>
"##;
