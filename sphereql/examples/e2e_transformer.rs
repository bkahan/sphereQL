use std::collections::HashMap;

use sphereql::embed::{
    Embedding, EmbeddingIndex, PcaProjection, Projection, RadialStrategy,
};

// ---------------------------------------------------------------------------
// Data structures for the JSON corpus
// ---------------------------------------------------------------------------

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
    theta: f64,
    phi: f64,
    r: f64,
    neighbors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Category colors (must match the JS visualization)
// ---------------------------------------------------------------------------

fn category_color(cat: &str) -> &'static str {
    match cat {
        "science" => "#2196F3",
        "technology" => "#9C27B0",
        "sports" => "#FF5722",
        "cooking" => "#FF9800",
        "arts" => "#4CAF50",
        "nature" => "#009688",
        "history" => "#795548",
        _ => "#FFFFFF",
    }
}

fn main() {
    let json_path = std::env::args().nth(1).unwrap_or_else(|| {
        "sphereql-embed/tools/embeddings.json".into()
    });
    let output_path = std::env::args().nth(2).unwrap_or_else(|| {
        "sphere_viz.html".into()
    });

    // -----------------------------------------------------------------------
    // 1. Load embeddings from JSON
    // -----------------------------------------------------------------------
    eprintln!("Loading embeddings from {json_path}...");
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("Cannot read {json_path}: {e}"));
    let data: serde_json::Value = serde_json::from_str(&raw).expect("Invalid JSON");

    let dim = data["dimension"].as_u64().unwrap() as usize;
    let model_name = data["model"].as_str().unwrap_or("unknown");
    let arr = data["sentences"].as_array().expect("missing sentences array");

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

    // -----------------------------------------------------------------------
    // 2. Fit PCA projection and build spatial index
    // -----------------------------------------------------------------------
    eprintln!("Fitting PCA projection ({dim}-d → 3D sphere)...");
    let all_emb: Vec<Embedding> = sentences.iter().map(|s| s.embedding.clone()).collect();
    let pca = PcaProjection::fit(&all_emb, RadialStrategy::Magnitude);

    let mut index = EmbeddingIndex::builder(pca.clone())
        .uniform_shells(5, 5.0)
        .theta_divisions(12)
        .phi_divisions(6)
        .build();

    for s in &sentences {
        index.insert(&s.id, &s.embedding);
    }

    // -----------------------------------------------------------------------
    // 3. Project all points and compute nearest neighbors
    // -----------------------------------------------------------------------
    let k_neighbors = 5;
    let mut projected: Vec<ProjectedPoint> = Vec::with_capacity(sentences.len());

    for s in &sentences {
        let sp = pca.project(&s.embedding);
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
            theta: sp.theta,
            phi: sp.phi,
            r: sp.r,
            neighbors,
        });
    }

    // -----------------------------------------------------------------------
    // 4. Run demo queries and collect results for display
    // -----------------------------------------------------------------------
    eprintln!("Running demo queries...");

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
        // Fallback: synthesize queries by averaging corpus embeddings
        vec![
            ("How does gravity work in space?", avg_embeddings(&sentences, &["sci-4", "sci-7", "sci-3"])),
            ("What recipes should I try for dinner?", avg_embeddings(&sentences, &["cook-2", "cook-3", "cook-4"])),
            ("Tell me about famous musicians and composers", avg_embeddings(&sentences, &["art-1", "art-2", "art-4"])),
        ]
    };

    let mut query_results: Vec<QueryResult> = Vec::new();
    for (desc, emb) in &queries {
        let results = index.search_nearest(emb, 5);
        let sp = pca.project(emb);
        query_results.push(QueryResult {
            description: desc.to_string(),
            theta: sp.theta,
            phi: sp.phi,
            hits: results
                .iter()
                .map(|r| {
                    let cat = sentences
                        .iter()
                        .find(|s| s.id == r.item.id)
                        .map(|s| s.category.clone())
                        .unwrap_or_default();
                    QueryHit {
                        id: r.item.id.clone(),
                        distance_rad: r.distance,
                        category: cat,
                    }
                })
                .collect(),
        });
    }

    // -----------------------------------------------------------------------
    // 5. Print query results to terminal
    // -----------------------------------------------------------------------
    println!("=== SphereQL: End-to-End Transformer Pipeline ===\n");
    println!("Model: {model_name}  |  Corpus: {} sentences  |  Dim: {dim} → 3D\n", sentences.len());

    // Category summary
    let mut cat_counts: HashMap<&str, usize> = HashMap::new();
    for s in &sentences {
        *cat_counts.entry(&s.category).or_default() += 1;
    }
    println!("Categories:");
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (cat, count) in &cats {
        println!("  {:<12} {} docs  {}", cat, count, category_color(cat));
    }

    for qr in &query_results {
        println!("\n--- Query: \"{}\" ---", qr.description);
        for (i, hit) in qr.hits.iter().enumerate() {
            let text = sentences
                .iter()
                .find(|s| s.id == hit.id)
                .map(|s| truncate(&s.text, 60))
                .unwrap_or_default();
            println!(
                "  {}. [{:<12}] {:.4} rad ({:>6.2}°)  \"{}\"",
                i + 1,
                hit.category,
                hit.distance_rad,
                hit.distance_rad.to_degrees(),
                text,
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. Generate the visualization HTML
    // -----------------------------------------------------------------------
    eprintln!("\nGenerating visualization...");

    let data_json = build_viz_json(&projected, &query_results);
    let html = VIZ_TEMPLATE.replace("__DATA_PLACEHOLDER__", &data_json);

    std::fs::write(&output_path, &html)
        .unwrap_or_else(|e| panic!("Cannot write {output_path}: {e}"));

    eprintln!("Wrote {output_path} ({} bytes)", html.len());
    println!("\n✓ Visualization written to {output_path}");
    println!("  Open in a browser to explore the semantic sphere.");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct QueryResult {
    description: String,
    theta: f64,
    phi: f64,
    hits: Vec<QueryHit>,
}

struct QueryHit {
    id: String,
    distance_rad: f64,
    category: String,
}

fn avg_embeddings(sentences: &[Sentence], ids: &[&str]) -> Embedding {
    let dim = sentences[0].embedding.dimension();
    let mut sum = vec![0.0; dim];
    let mut count = 0;
    for s in sentences {
        if ids.contains(&s.id.as_str()) {
            for (i, v) in s.embedding.values.iter().enumerate() {
                sum[i] += v;
            }
            count += 1;
        }
    }
    if count > 0 {
        for v in &mut sum {
            *v /= count as f64;
        }
    }
    Embedding::new(sum)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

fn build_viz_json(points: &[ProjectedPoint], queries: &[QueryResult]) -> String {
    let pts: Vec<String> = points
        .iter()
        .map(|p| {
            let neighbors = p
                .neighbors
                .iter()
                .map(|n| format!("\"{}\"", n))
                .collect::<Vec<_>>()
                .join(",");
            let text_escaped = p.text.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "{{\"id\":\"{}\",\"text\":\"{}\",\"cat\":\"{}\",\"t\":{:.6},\"p\":{:.6},\"r\":{:.4},\"n\":[{}]}}",
                p.id, text_escaped, p.category, p.theta, p.phi, p.r, neighbors
            )
        })
        .collect();

    let qs: Vec<String> = queries
        .iter()
        .map(|q| {
            let hits: Vec<String> = q
                .hits
                .iter()
                .map(|h| {
                    format!(
                        "{{\"id\":\"{}\",\"d\":{:.4},\"cat\":\"{}\"}}",
                        h.id, h.distance_rad, h.category
                    )
                })
                .collect();
            let desc_escaped = q.description.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "{{\"desc\":\"{}\",\"t\":{:.6},\"p\":{:.6},\"hits\":[{}]}}",
                desc_escaped,
                q.theta,
                q.phi,
                hits.join(",")
            )
        })
        .collect();

    format!(
        "{{\"points\":[{}],\"queries\":[{}]}}",
        pts.join(","),
        qs.join(",")
    )
}

// ---------------------------------------------------------------------------
// Visualization HTML template — self-contained with inline Three.js
// ---------------------------------------------------------------------------

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
#legend{position:absolute;top:16px;left:16px;background:rgba(10,10,30,0.85);padding:14px 18px;border-radius:10px;font-size:13px;pointer-events:auto;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08)}
#legend h3{font-size:14px;margin-bottom:8px;color:#aaa;font-weight:500;letter-spacing:0.5px}
.lrow{display:flex;align-items:center;gap:8px;margin:4px 0;cursor:pointer;padding:2px 4px;border-radius:4px;transition:background .15s}
.lrow:hover{background:rgba(255,255,255,0.08)}
.lrow.dim{opacity:0.25}
.ldot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.lbl{font-size:12px}
.lcnt{font-size:11px;color:#888;margin-left:auto}
#info{position:absolute;top:16px;right:16px;width:340px;max-height:calc(100vh - 32px);overflow-y:auto;background:rgba(10,10,30,0.88);padding:18px;border-radius:10px;font-size:13px;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);display:none}
#info.visible{display:block}
#info h3{font-size:15px;margin-bottom:6px;font-weight:600}
#info .cat-tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500;margin-bottom:8px}
#info .txt{font-size:13px;line-height:1.5;color:#ccc;margin-bottom:12px}
#info .neighbors h4{font-size:12px;color:#888;margin-bottom:6px;font-weight:500}
#info .nb{padding:6px 8px;margin:3px 0;background:rgba(255,255,255,0.04);border-radius:6px;font-size:12px;cursor:pointer;transition:background .15s}
#info .nb:hover{background:rgba(255,255,255,0.1)}
#info .nb .dist{color:#888;font-size:11px;float:right}
#queries{position:absolute;bottom:16px;left:16px;background:rgba(10,10,30,0.85);padding:14px 18px;border-radius:10px;font-size:13px;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08)}
#queries h3{font-size:14px;margin-bottom:8px;color:#aaa;font-weight:500;letter-spacing:0.5px}
.qbtn{display:block;padding:6px 10px;margin:4px 0;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:#ddd;cursor:pointer;font-size:12px;text-align:left;width:100%;transition:background .15s}
.qbtn:hover{background:rgba(255,255,255,0.12)}
.qbtn.active{background:rgba(100,180,255,0.15);border-color:rgba(100,180,255,0.3)}
#help{position:absolute;bottom:16px;right:16px;font-size:11px;color:#555;text-align:right;pointer-events:none}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="legend"></div>
<div id="info"></div>
<div id="queries"></div>
<div id="help">drag to rotate · scroll to zoom · click a point to inspect</div>

<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.163.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.163.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const DATA = __DATA_PLACEHOLDER__;

const COLORS = {
    science:'#2196F3', technology:'#9C27B0', sports:'#FF5722',
    cooking:'#FF9800', arts:'#4CAF50', nature:'#009688', history:'#795548'
};

// ---- Scene setup ----
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(0, 0.8, 2.8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.5;
controls.maxDistance = 6;

// ---- Lighting ----
scene.add(new THREE.AmbientLight(0x404060, 1.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(3, 5, 4);
scene.add(dirLight);

// ---- Reference sphere (wireframe) ----
const wireGeo = new THREE.IcosahedronGeometry(1, 4);
const wireMat = new THREE.MeshBasicMaterial({color:0x334466, wireframe:true, transparent:true, opacity:0.08});
scene.add(new THREE.Mesh(wireGeo, wireMat));

// Equator ring
const eqGeo = new THREE.RingGeometry(0.998, 1.002, 128);
const eqMat = new THREE.MeshBasicMaterial({color:0x445588, transparent:true, opacity:0.2, side:THREE.DoubleSide});
const eqRing = new THREE.Mesh(eqGeo, eqMat);
eqRing.rotation.x = Math.PI / 2;
scene.add(eqRing);

// ---- Coordinate conversion (SphereQL physics convention) ----
function sph2cart(theta, phi, r) {
    r = r || 1;
    return new THREE.Vector3(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi),                     // Y-up for Three.js (swap y/z)
        r * Math.sin(phi) * Math.sin(theta)
    );
}

// ---- Build point meshes ----
const pointGroup = new THREE.Group();
scene.add(pointGroup);
const meshes = [];
const idToMesh = {};
const idToData = {};

const sphereGeo = new THREE.SphereGeometry(0.018, 12, 12);

DATA.points.forEach(pt => {
    const color = new THREE.Color(COLORS[pt.cat] || '#ffffff');
    const mat = new THREE.MeshStandardMaterial({color, emissive:color, emissiveIntensity:0.4, roughness:0.5});
    const mesh = new THREE.Mesh(sphereGeo, mat);
    const pos = sph2cart(pt.t, pt.p);
    mesh.position.copy(pos);
    mesh.userData = pt;
    pointGroup.add(mesh);
    meshes.push(mesh);
    idToMesh[pt.id] = mesh;
    idToData[pt.id] = pt;
});

// ---- Neighbor lines (toggled on selection) ----
const lineGroup = new THREE.Group();
scene.add(lineGroup);
let selectedId = null;

function clearLines() {
    while (lineGroup.children.length) lineGroup.remove(lineGroup.children[0]);
}

function drawNeighborLines(pt) {
    clearLines();
    const origin = sph2cart(pt.t, pt.p);
    pt.n.forEach(nid => {
        const nd = idToData[nid];
        if (!nd) return;
        const dest = sph2cart(nd.t, nd.p);
        const geo = new THREE.BufferGeometry().setFromPoints([origin, dest]);
        const mat = new THREE.LineBasicMaterial({color:0xffffff, transparent:true, opacity:0.35});
        lineGroup.add(new THREE.Line(geo, mat));
    });
}

// ---- Query markers ----
const queryGroup = new THREE.Group();
scene.add(queryGroup);

function showQueryOnSphere(q) {
    while (queryGroup.children.length) queryGroup.remove(queryGroup.children[0]);
    // Query point
    const qMat = new THREE.MeshBasicMaterial({color:0xffff00});
    const qMesh = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 12), qMat);
    qMesh.position.copy(sph2cart(q.t, q.p));
    queryGroup.add(qMesh);
    // Lines to hits
    q.hits.forEach(h => {
        const nd = idToData[h.id];
        if (!nd) return;
        const geo = new THREE.BufferGeometry().setFromPoints([sph2cart(q.t, q.p), sph2cart(nd.t, nd.p)]);
        const mat = new THREE.LineBasicMaterial({color:0xffff44, transparent:true, opacity:0.5});
        queryGroup.add(new THREE.Line(geo, mat));
    });
}

// ---- Raycaster ----
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredMesh = null;

canvas.addEventListener('mousemove', e => {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
});

canvas.addEventListener('click', () => {
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(meshes);
    if (hits.length) {
        const pt = hits[0].object.userData;
        selectPoint(pt);
    }
});

function selectPoint(pt) {
    selectedId = pt.id;
    drawNeighborLines(pt);
    // Reset scales
    meshes.forEach(m => m.scale.setScalar(1));
    // Highlight selected + neighbors
    if (idToMesh[pt.id]) idToMesh[pt.id].scale.setScalar(2.5);
    pt.n.forEach(nid => { if (idToMesh[nid]) idToMesh[nid].scale.setScalar(1.8); });
    showInfoPanel(pt);
    // Clear query highlights
    while (queryGroup.children.length) queryGroup.remove(queryGroup.children[0]);
    document.querySelectorAll('.qbtn').forEach(b => b.classList.remove('active'));
}

// ---- Info panel ----
const infoEl = document.getElementById('info');

function showInfoPanel(pt) {
    const color = COLORS[pt.cat] || '#999';
    let nbHtml = '';
    pt.n.forEach(nid => {
        const nd = idToData[nid];
        if (!nd) return;
        const nColor = COLORS[nd.cat] || '#999';
        nbHtml += `<div class="nb" onclick="window._selectById('${nid}')"><span style="color:${nColor}">●</span> ${nd.text.substring(0, 65)}…<span class="dist">${nd.cat}</span></div>`;
    });
    infoEl.innerHTML = `
        <h3>${pt.id}</h3>
        <div class="cat-tag" style="background:${color}33;color:${color}">${pt.cat}</div>
        <div class="txt">${pt.text}</div>
        <div style="font-size:11px;color:#888;margin-bottom:12px">θ=${(pt.t*180/Math.PI).toFixed(1)}°  φ=${(pt.p*180/Math.PI).toFixed(1)}°  r=${pt.r.toFixed(3)}</div>
        <div class="neighbors"><h4>Nearest neighbors</h4>${nbHtml}</div>
    `;
    infoEl.classList.add('visible');
}

window._selectById = function(id) {
    const pt = idToData[id];
    if (pt) selectPoint(pt);
};

// ---- Legend ----
const catCounts = {};
DATA.points.forEach(p => { catCounts[p.cat] = (catCounts[p.cat]||0)+1; });
const hiddenCats = new Set();

const legendEl = document.getElementById('legend');
let legendHtml = '<h3>Categories</h3>';
Object.entries(COLORS).forEach(([cat, color]) => {
    if (!catCounts[cat]) return;
    legendHtml += `<div class="lrow" data-cat="${cat}"><div class="ldot" style="background:${color}"></div><span class="lbl">${cat}</span><span class="lcnt">${catCounts[cat]}</span></div>`;
});
legendEl.innerHTML = legendHtml;

legendEl.querySelectorAll('.lrow').forEach(row => {
    row.addEventListener('click', () => {
        const cat = row.dataset.cat;
        if (hiddenCats.has(cat)) { hiddenCats.delete(cat); row.classList.remove('dim'); }
        else { hiddenCats.add(cat); row.classList.add('dim'); }
        meshes.forEach(m => {
            m.visible = !hiddenCats.has(m.userData.cat);
        });
    });
});

// ---- Query buttons ----
const queriesEl = document.getElementById('queries');
if (DATA.queries.length) {
    let qHtml = '<h3>Demo Queries</h3>';
    DATA.queries.forEach((q, i) => {
        qHtml += `<button class="qbtn" data-qi="${i}">"${q.desc}"</button>`;
    });
    queriesEl.innerHTML = qHtml;

    queriesEl.querySelectorAll('.qbtn').forEach(btn => {
        btn.addEventListener('click', () => {
            const qi = parseInt(btn.dataset.qi);
            const q = DATA.queries[qi];
            document.querySelectorAll('.qbtn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Reset point scales
            meshes.forEach(m => m.scale.setScalar(1));
            clearLines();
            selectedId = null;
            // Highlight hits
            q.hits.forEach(h => { if (idToMesh[h.id]) idToMesh[h.id].scale.setScalar(2.2); });
            showQueryOnSphere(q);
            // Show results in info panel
            let rHtml = `<h3>Query Results</h3><div class="txt" style="font-style:italic">"${q.desc}"</div>`;
            q.hits.forEach((h, i) => {
                const nd = idToData[h.id];
                const nColor = COLORS[h.cat] || '#999';
                rHtml += `<div class="nb" onclick="window._selectById('${h.id}')"><span style="color:${nColor}">●</span> <b>${i+1}.</b> ${nd?nd.text.substring(0,55)+'…':h.id}<span class="dist">${(h.d*180/Math.PI).toFixed(1)}°</span></div>`;
            });
            infoEl.innerHTML = rHtml;
            infoEl.classList.add('visible');
        });
    });
} else {
    queriesEl.style.display = 'none';
}

// ---- Hover highlight ----
function updateHover() {
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(meshes.filter(m => m.visible));
    const newHover = hits.length ? hits[0].object : null;
    if (hoveredMesh && hoveredMesh !== newHover) {
        if (hoveredMesh.userData.id !== selectedId) {
            hoveredMesh.scale.setScalar(hoveredMesh.userData.n && hoveredMesh.userData.n.includes(selectedId) ? 1.8 : 1);
        }
    }
    if (newHover && newHover.userData.id !== selectedId) {
        newHover.scale.setScalar(1.6);
        canvas.style.cursor = 'pointer';
    } else {
        canvas.style.cursor = 'grab';
    }
    hoveredMesh = newHover;
}

// ---- Resize ----
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---- Animate ----
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    updateHover();
    renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"##;
