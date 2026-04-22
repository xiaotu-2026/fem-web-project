// main.js - FEA Web Platform orchestrator
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GpuJacobiSolver } from './gpu_solver.js';
const $ = id => document.getElementById(id);
const log = msg => { $('status').textContent = msg; console.log('[FEA]', msg); };
const canvas = $('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a1628);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(3, 2, 2);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(5, 10, 7); scene.add(dl);
let meshGroup = null, previewGroup = null, axisGroup = null;
let currentView = 'disp', femData = null, analysisRun = false;

function resize(){const c=$('canvas-container');if(!c)return;renderer.setSize(c.clientWidth,c.clientHeight);camera.aspect=c.clientWidth/c.clientHeight;camera.updateProjectionMatrix();}
window.addEventListener('resize', resize); resize();
(function anim(){requestAnimationFrame(anim);controls.update();renderer.render(scene,camera);})();

// --- Axis labels using sprite ---
function makeTextSprite(text, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 128; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'Bold 48px Arial';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(0.3, 0.15, 1);
    return sprite;
}

function buildAxes(size) {
    if (axisGroup) { scene.remove(axisGroup); }
    axisGroup = new THREE.Group();
    const len = size * 0.3;
    const headLen = len * 0.15;
    const headW = len * 0.06;
    // X axis - red
    const arX = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), len, 0xff4444, headLen, headW);
    axisGroup.add(arX);
    const lblX = makeTextSprite('X', '#ff4444');
    lblX.position.set(len + 0.08, 0, 0);
    axisGroup.add(lblX);
    // Y axis - green
    const arY = new THREE.ArrowHelper(new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,0), len, 0x44ff44, headLen, headW);
    axisGroup.add(arY);
    const lblY = makeTextSprite('Y', '#44ff44');
    lblY.position.set(0, len + 0.08, 0);
    axisGroup.add(lblY);
    // Z axis - blue
    const arZ = new THREE.ArrowHelper(new THREE.Vector3(0,0,1), new THREE.Vector3(0,0,0), len, 0x4488ff, headLen, headW);
    axisGroup.add(arZ);
    const lblZ = makeTextSprite('Z', '#4488ff');
    lblZ.position.set(0, 0, len + 0.08);
    axisGroup.add(lblZ);
    scene.add(axisGroup);
}

// --- Preview visualization ---
function getFaceVertices(Lx, Ly, Lz, faceId) {
    // Returns 4 corner vertices of the face and the outward normal direction
    // faceId: 0=X0, 1=XL, 2=Y0, 3=YL, 4=Z0, 5=ZL
    switch(faceId) {
        case 0: return { verts: [[0,0,0],[0,Ly,0],[0,Ly,Lz],[0,0,Lz]], normal: [-1,0,0], center: [0,Ly/2,Lz/2] };
        case 1: return { verts: [[Lx,0,0],[Lx,Ly,0],[Lx,Ly,Lz],[Lx,0,Lz]], normal: [1,0,0], center: [Lx,Ly/2,Lz/2] };
        case 2: return { verts: [[0,0,0],[Lx,0,0],[Lx,0,Lz],[0,0,Lz]], normal: [0,-1,0], center: [Lx/2,0,Lz/2] };
        case 3: return { verts: [[0,Ly,0],[Lx,Ly,0],[Lx,Ly,Lz],[0,Ly,Lz]], normal: [0,1,0], center: [Lx/2,Ly,Lz/2] };
        case 4: return { verts: [[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0]], normal: [0,0,-1], center: [Lx/2,Ly/2,0] };
        case 5: return { verts: [[0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz]], normal: [0,0,1], center: [Lx/2,Ly/2,Lz] };
    }
}

function buildFixedFaceIndicator(group, Lx, Ly, Lz, faceId) {
    const face = getFaceVertices(Lx, Ly, Lz, faceId);
    // Semi-transparent green plane
    // Project face vertices to a plane geometry
    const v = face.verts;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array([
        v[0][0],v[0][1],v[0][2], v[1][0],v[1][1],v[1][2], v[2][0],v[2][1],v[2][2],
        v[0][0],v[0][1],v[0][2], v[2][0],v[2][1],v[2][2], v[3][0],v[3][1],v[3][2]
    ]);
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.computeVertexNormals();
    const mat = new THREE.MeshBasicMaterial({ color: 0x00cc88, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
    group.add(new THREE.Mesh(geom, mat));

    // Draw constraint triangles (ground symbol) along the face edge
    const n = face.normal;
    const maxDim = Math.max(Lx, Ly, Lz);
    const triSize = maxDim * 0.04;
    const numTri = 6;
    for (let i = 0; i < numTri; i++) {
        for (let j = 0; j < numTri; j++) {
            const t1 = (i + 0.5) / numTri;
            const t2 = (j + 0.5) / numTri;
            // Interpolate position on face
            const px = v[0][0]*(1-t1)*(1-t2) + v[1][0]*t1*(1-t2) + v[2][0]*t1*t2 + v[3][0]*(1-t1)*t2;
            const py = v[0][1]*(1-t1)*(1-t2) + v[1][1]*t1*(1-t2) + v[2][1]*t1*t2 + v[3][1]*(1-t1)*t2;
            const pz = v[0][2]*(1-t1)*(1-t2) + v[1][2]*t1*(1-t2) + v[2][2]*t1*t2 + v[3][2]*(1-t1)*t2;
            // Small triangle pointing inward
            const triGeom = new THREE.BufferGeometry();
            // Determine two tangent directions on the face
            let tx = v[1][0]-v[0][0], ty = v[1][1]-v[0][1], tz = v[1][2]-v[0][2];
            let tlen = Math.sqrt(tx*tx+ty*ty+tz*tz); tx/=tlen; ty/=tlen; tz/=tlen;
            const triVerts = new Float32Array([
                px - tx*triSize*0.5, py - ty*triSize*0.5, pz - tz*triSize*0.5,
                px + tx*triSize*0.5, py + ty*triSize*0.5, pz + tz*triSize*0.5,
                px + n[0]*triSize*0.7, py + n[1]*triSize*0.7, pz + n[2]*triSize*0.7
            ]);
            triGeom.setAttribute('position', new THREE.Float32BufferAttribute(triVerts, 3));
            const triMat = new THREE.MeshBasicMaterial({ color: 0x00aa66, side: THREE.DoubleSide });
            group.add(new THREE.Mesh(triGeom, triMat));
        }
    }

    // Add "固定" label
    const lbl = makeTextSprite('固定', '#00cc88');
    lbl.position.set(face.center[0] + n[0]*maxDim*0.12, face.center[1] + n[1]*maxDim*0.12, face.center[2] + n[2]*maxDim*0.12);
    lbl.scale.set(0.4, 0.2, 1);
    group.add(lbl);
}

function buildLoadFaceIndicator(group, Lx, Ly, Lz, faceId) {
    const face = getFaceVertices(Lx, Ly, Lz, faceId);
    const v = face.verts;
    // Semi-transparent red/orange plane
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array([
        v[0][0],v[0][1],v[0][2], v[1][0],v[1][1],v[1][2], v[2][0],v[2][1],v[2][2],
        v[0][0],v[0][1],v[0][2], v[2][0],v[2][1],v[2][2], v[3][0],v[3][1],v[3][2]
    ]);
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.computeVertexNormals();
    const mat = new THREE.MeshBasicMaterial({ color: 0xff6633, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
    group.add(new THREE.Mesh(geom, mat));

    // Arrows pointing inward (pressure direction = -normal)
    const n = face.normal;
    const maxDim = Math.max(Lx, Ly, Lz);
    const arrowLen = maxDim * 0.15;
    const dir = new THREE.Vector3(-n[0], -n[1], -n[2]);
    const numArr = 3;
    for (let i = 0; i < numArr; i++) {
        for (let j = 0; j < numArr; j++) {
            const t1 = (i + 0.5) / numArr;
            const t2 = (j + 0.5) / numArr;
            const px = v[0][0]*(1-t1)*(1-t2) + v[1][0]*t1*(1-t2) + v[2][0]*t1*t2 + v[3][0]*(1-t1)*t2;
            const py = v[0][1]*(1-t1)*(1-t2) + v[1][1]*t1*(1-t2) + v[2][1]*t1*t2 + v[3][1]*(1-t1)*t2;
            const pz = v[0][2]*(1-t1)*(1-t2) + v[1][2]*t1*(1-t2) + v[2][2]*t1*t2 + v[3][2]*(1-t1)*t2;
            // Arrow starts outside the face, points inward
            const origin = new THREE.Vector3(
                px + n[0]*arrowLen*1.2,
                py + n[1]*arrowLen*1.2,
                pz + n[2]*arrowLen*1.2
            );
            const arrow = new THREE.ArrowHelper(dir, origin, arrowLen, 0xff4422, arrowLen*0.3, arrowLen*0.15);
            group.add(arrow);
        }
    }

    // Add "荷载" label
    const lbl = makeTextSprite('荷载', '#ff6633');
    lbl.position.set(
        face.center[0] + n[0]*maxDim*0.25,
        face.center[1] + n[1]*maxDim*0.25,
        face.center[2] + n[2]*maxDim*0.25
    );
    lbl.scale.set(0.4, 0.2, 1);
    group.add(lbl);
}

function buildPreview() {
    if (analysisRun) return; // Don't overwrite analysis results
    if (previewGroup) { scene.remove(previewGroup); previewGroup = null; }

    const Lx = +$('inLx').value || 2;
    const Ly = +$('inLy').value || 0.5;
    const Lz = +$('inLz').value || 0.5;
    const fixFace = +$('inFixFace').value;
    const loadFace = +$('inLoadFace').value;

    previewGroup = new THREE.Group();

    // Wireframe box
    const boxGeom = new THREE.BoxGeometry(Lx, Ly, Lz);
    const edges = new THREE.EdgesGeometry(boxGeom);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x88bbdd, linewidth: 1 });
    const wireframe = new THREE.LineSegments(edges, lineMat);
    wireframe.position.set(Lx/2, Ly/2, Lz/2);
    previewGroup.add(wireframe);

    // Semi-transparent solid box
    const boxMat = new THREE.MeshBasicMaterial({ color: 0x2255aa, transparent: true, opacity: 0.08, side: THREE.DoubleSide });
    const boxMesh = new THREE.Mesh(boxGeom.clone(), boxMat);
    boxMesh.position.set(Lx/2, Ly/2, Lz/2);
    previewGroup.add(boxMesh);

    // Fixed face indicator
    buildFixedFaceIndicator(previewGroup, Lx, Ly, Lz, fixFace);

    // Load face indicator
    buildLoadFaceIndicator(previewGroup, Lx, Ly, Lz, loadFace);

    scene.add(previewGroup);

    // Build axes
    const maxDim = Math.max(Lx, Ly, Lz);
    buildAxes(maxDim);

    // Position camera
    const sz = Math.sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
    controls.target.set(Lx/2, Ly/2, Lz/2);
    camera.position.set(Lx/2 + sz*0.8, Ly/2 + sz*0.5, Lz/2 + sz*0.8);
    controls.update();
}

// --- Colormap and result visualization ---
function jet(t){t=Math.max(0,Math.min(1,t));if(t<0.25)return new THREE.Color(0,4*t,1);if(t<0.5)return new THREE.Color(0,1,1-4*(t-0.25));if(t<0.75)return new THREE.Color(4*(t-0.5),1,0);return new THREE.Color(1,1-4*(t-0.75),0);}
function drawColorbar(lo,hi,title){const cb=$('colorbar'),ctx=cb.getContext('2d');cb.width=24;cb.height=260;for(let y=0;y<260;y++){const c=jet(1-y/259);ctx.fillStyle='rgb('+(c.r*255|0)+','+(c.g*255|0)+','+(c.b*255|0)+')';ctx.fillRect(0,y,24,1);}$('cb-max').textContent=hi.toExponential(2);$('cb-mid').textContent=((lo+hi)/2).toExponential(2);$('cb-min').textContent=lo.toExponential(2);$('legend-title').textContent=title;}
const hexFaces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]];
function avg2nodes(ev,conn,nN,nE){const s=new Float64Array(nN),c=new Uint32Array(nN);for(let e=0;e<nE;e++)for(let i=0;i<8;i++){const n=conn[8*e+i];s[n]+=ev[e];c[n]++;}const r=new Float64Array(nN);for(let i=0;i<nN;i++)r[i]=c[i]>0?s[i]/c[i]:0;return r;}

function buildVis(mode) {
    if(!femData)return;
    // Clear preview if present
    if(previewGroup){scene.remove(previewGroup);previewGroup=null;}
    if(meshGroup){scene.remove(meshGroup);meshGroup=null;}
    meshGroup=new THREE.Group();
    const{coords,conn,nN,nE,disp,stress,strain}=femData; let nv=null,title='',lo=Infinity,hi=-Infinity;
    if(mode==='disp'&&disp){nv=new Float64Array(nN);for(let i=0;i<nN;i++){const u=disp[3*i],v=disp[3*i+1],w=disp[3*i+2];nv[i]=Math.sqrt(u*u+v*v+w*w);}title='|U| 位移 (m)';}
    else if(mode==='stress'&&stress){const vm=new Float64Array(nE);for(let e=0;e<nE;e++){const s=stress.subarray(6*e,6*e+6);vm[e]=Math.sqrt(0.5*((s[0]-s[1])**2+(s[1]-s[2])**2+(s[2]-s[0])**2+6*(s[3]**2+s[4]**2+s[5]**2)));}nv=avg2nodes(vm,conn,nN,nE);title='Von Mises 应力 (Pa)';}
    else if(mode==='strain'&&strain){const eq=new Float64Array(nE);for(let e=0;e<nE;e++){const s=strain.subarray(6*e,6*e+6);eq[e]=Math.sqrt(2/9*((s[0]-s[1])**2+(s[1]-s[2])**2+(s[2]-s[0])**2)+1/3*(s[3]**2+s[4]**2+s[5]**2));}nv=avg2nodes(eq,conn,nN,nE);title='等效应变';}
    if(nv){for(let i=0;i<nN;i++){if(nv[i]<lo)lo=nv[i];if(nv[i]>hi)hi=nv[i];}if(hi-lo<1e-30)hi=lo+1;}
    let ds=0;if(mode!=='mesh'&&disp){let md=0,mx=0;for(let i=0;i<nN;i++){md=Math.max(md,Math.abs(coords[3*i]),Math.abs(coords[3*i+1]),Math.abs(coords[3*i+2]));mx=Math.max(mx,Math.sqrt(disp[3*i]**2+disp[3*i+1]**2+disp[3*i+2]**2));}ds=mx>1e-30?0.1*md/mx:0;}
    const pos=[],col=[],dc=new THREE.Color(0.3,0.5,0.8);
    for(let e=0;e<nE;e++){const en=[];for(let i=0;i<8;i++)en.push(conn[8*e+i]);for(const f of hexFaces){const fn=f.map(k=>en[k]);for(const tri of[[fn[0],fn[1],fn[2]],[fn[0],fn[2],fn[3]]]){for(const ni of tri){let x=coords[3*ni],y=coords[3*ni+1],z=coords[3*ni+2];if(disp&&ds>0){x+=disp[3*ni]*ds;y+=disp[3*ni+1]*ds;z+=disp[3*ni+2]*ds;}pos.push(x,y,z);if(nv){const c=jet((nv[ni]-lo)/(hi-lo));col.push(c.r,c.g,c.b);}else col.push(dc.r,dc.g,dc.b);}}}}
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setAttribute('color',new THREE.Float32BufferAttribute(col,3));g.computeVertexNormals();
    meshGroup.add(new THREE.Mesh(g,new THREE.MeshPhongMaterial({vertexColors:true,side:THREE.DoubleSide,flatShading:true})));
    meshGroup.add(new THREE.LineSegments(new THREE.EdgesGeometry(g,15),new THREE.LineBasicMaterial({color:0x000000,opacity:0.25,transparent:true})));
    scene.add(meshGroup);if(nv)drawColorbar(lo,hi,title);
    const box=new THREE.Box3().setFromObject(meshGroup),cen=box.getCenter(new THREE.Vector3()),sz=box.getSize(new THREE.Vector3()).length();
    controls.target.copy(cen);camera.position.copy(cen).add(new THREE.Vector3(sz*0.8,sz*0.5,sz*0.8));controls.update();
    // Update axes for result view
    const maxDim = Math.max(+$('inLx').value||2, +$('inLy').value||0.5, +$('inLz').value||0.5);
    buildAxes(maxDim);
}

function waitWasm(){return new Promise(function(resolve){if(typeof Module!=='undefined'&&Module.calledRun){resolve(Module);return;}if(typeof Module!=='undefined'){Module.onRuntimeInitialized=function(){resolve(Module);};return;}var t=setInterval(function(){if(typeof Module!=='undefined'){clearInterval(t);if(Module.calledRun)resolve(Module);else Module.onRuntimeInitialized=function(){resolve(Module);};}},50);});}

async function runAnalysis(){
    $('btnRun').disabled=true;
    analysisRun = true;
    // Clear preview
    if(previewGroup){scene.remove(previewGroup);previewGroup=null;}
    log('正在加载 WASM 模块...');
    var M=await waitWasm();
    var Lx=+$('inLx').value,Ly=+$('inLy').value,Lz=+$('inLz').value;
    var nx=+$('inNx').value,ny=+$('inNy').value,nz=+$('inNz').value;
    var E=+$('inE').value,nu=+$('inNu').value,P=+$('inP').value;
    var fixFace=+$('inFixFace').value, loadFace=+$('inLoadFace').value;
    log('正在生成网格...');await new Promise(function(r){setTimeout(r,10);});
    M._generateMesh(Lx,Ly,Lz,nx,ny,nz);
    var nN=M._getNumNodes(),nE=M._getNumElems(),nDof=M._getNumDof();
    log('网格: '+nN+' 节点, '+nE+' 单元, '+nDof+' 自由度');
    femData={coords:new Float64Array(M.HEAPF64.buffer,M._getNodeCoords(),3*nN).slice(),conn:new Int32Array(M.HEAP32.buffer,M._getElemConn(),8*nE).slice(),nN:nN,nE:nE,disp:null,stress:null,strain:null};
    buildVis('mesh');await new Promise(function(r){setTimeout(r,30);});
    log('正在组装刚度矩阵...');await new Promise(function(r){setTimeout(r,10);});
    M._assembleSystem(E,nu,P,fixFace,loadFace);
    var nnz=M._getNnz();
    log('非零元素: '+nnz+'，正在求解...');
    var rp32=new Int32Array(M.HEAP32.buffer,M._getCsrRowPtr(),nDof+1);
    var ci32=new Int32Array(M.HEAP32.buffer,M._getCsrColIdx(),nnz);
    var vf64=new Float64Array(M.HEAPF64.buffer,M._getCsrValues(),nnz);
    var rf64=new Float64Array(M.HEAPF64.buffer,M._getRhsVec(),nDof);
    var rp=new Uint32Array(nDof+1),ci=new Uint32Array(nnz),vals=new Float64Array(nnz),rhs=new Float64Array(nDof);
    for(var i=0;i<=nDof;i++)rp[i]=rp32[i];
    for(var i=0;i<nnz;i++){ci[i]=ci32[i];vals[i]=vf64[i];}
    for(var i=0;i<nDof;i++)rhs[i]=rf64[i];
    var solver=new GpuJacobiSolver();var gpu=await solver.init();
    var useGpu = gpu && $('inSolver').value === 'gpu';
    var gpuBadge = $('gpu-badge');
    if (gpuBadge) {
        gpuBadge.textContent = useGpu ? 'GPU' : 'CPU';
        gpuBadge.className = 'gpu-badge ' + (useGpu ? 'gpu-on' : 'gpu-off');
    }
    log('正在求解（' + (useGpu ? 'WebGPU' : 'CPU') + ' PCG 预条件共轭梯度法）...');
    var t0=performance.now();
    var sol = useGpu
        ? await solver.solveF64(rp,ci,vals,rhs,nDof,20000,1e-8)
        : solver.solveCPU_PCG(rp,ci,vals,rhs,nDof,20000,1e-8);
    var dt=((performance.now()-t0)/1000).toFixed(2);
    var solverLabel = useGpu && solver.gpuUsed ? 'WebGPU' : 'CPU';
    if (gpuBadge) {
        var actualGpu = useGpu && solver.gpuUsed;
        gpuBadge.textContent = actualGpu ? 'GPU' : 'CPU';
        gpuBadge.className = 'gpu-badge ' + (actualGpu ? 'gpu-on' : 'gpu-off');
    }
    for(var i=0;i<nDof;i++)M._setSolution(i,sol[i]);
    M._computeResults(E,nu);
    var d=new Float64Array(nDof);for(var i=0;i<nDof;i++)d[i]=sol[i];
    femData.disp=d;
    femData.stress=new Float64Array(M.HEAPF64.buffer,M._getStressData(),6*nE).slice();
    femData.strain=new Float64Array(M.HEAPF64.buffer,M._getStrainData(),6*nE).slice();
    currentView='disp';updBtns();buildVis('disp');
    var mx=0;for(var i=0;i<nN;i++)mx=Math.max(mx,Math.sqrt(d[3*i]**2+d[3*i+1]**2+d[3*i+2]**2));
    log('完成（' + solverLabel + '）。最大位移: '+mx.toExponential(4)+' m，求解耗时: '+dt+' 秒');
    $('btnRun').disabled=false;
}

function updBtns(){document.querySelectorAll('.view-btns button').forEach(function(b){b.classList.remove('active');});var m={disp:'btnDisp',strain:'btnStrain',stress:'btnStress',mesh:'btnMesh'};if(m[currentView])$(m[currentView]).classList.add('active');}
$('btnDisp').onclick=function(){currentView='disp';updBtns();buildVis('disp');};
$('btnStrain').onclick=function(){currentView='strain';updBtns();buildVis('strain');};
$('btnStress').onclick=function(){currentView='stress';updBtns();buildVis('stress');};
$('btnMesh').onclick=function(){currentView='mesh';updBtns();buildVis('mesh');};
$('btnRun').onclick=function(){analysisRun=false;femData=null;runAnalysis().catch(function(e){log('错误: '+e.message);console.error(e);$('btnRun').disabled=false;});};

// Listen for parameter changes to update preview
['inLx','inLy','inLz','inFixFace','inLoadFace'].forEach(function(id){
    $(id).addEventListener('input', function(){ if(!analysisRun) buildPreview(); });
    $(id).addEventListener('change', function(){ if(!analysisRun) buildPreview(); });
});

// Initial preview on load
buildPreview();
if (!navigator.gpu) {
    var sel = $('inSolver');
    if (sel) {
        sel.querySelector('option[value="gpu"]').disabled = true;
        sel.value = 'cpu';
    }
    var badge = $('gpu-badge');
    if (badge) { badge.textContent = '无GPU'; badge.className = 'gpu-badge gpu-off'; }
}
log('就绪。设置参数后点击"运行分析"。调整参数可实时预览模型。');
