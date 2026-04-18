// gen_main.js — generates public/main.js
const fs = require('fs');
const D = '$';

const src = `// main.js - FEA Web Platform orchestrator
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GpuJacobiSolver } from './gpu_solver.js';
const ${D} = id => document.getElementById(id);
const log = msg => { ${D}('status').textContent = msg; console.log('[FEA]', msg); };
const canvas = ${D}('viewport');
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
scene.add(new THREE.AxesHelper(0.3));
let meshGroup = null, currentView = 'disp', femData = null;
function resize(){const c=${D}('canvas-container');if(!c)return;renderer.setSize(c.clientWidth,c.clientHeight);camera.aspect=c.clientWidth/c.clientHeight;camera.updateProjectionMatrix();}
window.addEventListener('resize', resize); resize();
(function anim(){requestAnimationFrame(anim);controls.update();renderer.render(scene,camera);})();

function jet(t){t=Math.max(0,Math.min(1,t));if(t<0.25)return new THREE.Color(0,4*t,1);if(t<0.5)return new THREE.Color(0,1,1-4*(t-0.25));if(t<0.75)return new THREE.Color(4*(t-0.5),1,0);return new THREE.Color(1,1-4*(t-0.75),0);}
function drawColorbar(lo,hi,title){const cb=${D}('colorbar'),ctx=cb.getContext('2d');cb.width=24;cb.height=260;for(let y=0;y<260;y++){const c=jet(1-y/259);ctx.fillStyle='rgb('+(c.r*255|0)+','+(c.g*255|0)+','+(c.b*255|0)+')';ctx.fillRect(0,y,24,1);}${D}('cb-max').textContent=hi.toExponential(2);${D}('cb-mid').textContent=((lo+hi)/2).toExponential(2);${D}('cb-min').textContent=lo.toExponential(2);${D}('legend-title').textContent=title;}
const hexFaces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]];
function avg2nodes(ev,conn,nN,nE){const s=new Float64Array(nN),c=new Uint32Array(nN);for(let e=0;e<nE;e++)for(let i=0;i<8;i++){const n=conn[8*e+i];s[n]+=ev[e];c[n]++;}const r=new Float64Array(nN);for(let i=0;i<nN;i++)r[i]=c[i]>0?s[i]/c[i]:0;return r;}
function buildVis(mode) {
    if(!femData)return; if(meshGroup){scene.remove(meshGroup);meshGroup=null;} meshGroup=new THREE.Group();
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
}
function waitWasm(){return new Promise(function(resolve){if(typeof Module!=='undefined'&&Module.calledRun){resolve(Module);return;}if(typeof Module!=='undefined'){Module.onRuntimeInitialized=function(){resolve(Module);};return;}var t=setInterval(function(){if(typeof Module!=='undefined'){clearInterval(t);if(Module.calledRun)resolve(Module);else Module.onRuntimeInitialized=function(){resolve(Module);};}},50);});}

async function runAnalysis(){
    ${D}('btnRun').disabled=true;
    log('正在加载 WASM 模块...');
    var M=await waitWasm();
    var Lx=+${D}('inLx').value,Ly=+${D}('inLy').value,Lz=+${D}('inLz').value;
    var nx=+${D}('inNx').value,ny=+${D}('inNy').value,nz=+${D}('inNz').value;
    var E=+${D}('inE').value,nu=+${D}('inNu').value,P=+${D}('inP').value;
    var fixFace=+${D}('inFixFace').value, loadFace=+${D}('inLoadFace').value;
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
    var rp=new Uint32Array(nDof+1),ci=new Uint32Array(nnz),vf=new Float32Array(nnz),rf=new Float32Array(nDof);
    for(var i=0;i<=nDof;i++)rp[i]=rp32[i];
    for(var i=0;i<nnz;i++){ci[i]=ci32[i];vf[i]=vf64[i];}
    for(var i=0;i<nDof;i++)rf[i]=rf64[i];
    var solver=new GpuJacobiSolver();var gpu=await solver.init();
    log(gpu?'正在 GPU 上求解...':'正在 CPU 上求解（回退模式）...');
    var t0=performance.now();
    var sol=await solver.solve(rp,ci,vf,rf,nDof,10000,1e-6);
    var dt=((performance.now()-t0)/1000).toFixed(2);
    for(var i=0;i<nDof;i++)M._setSolution(i,sol[i]);
    M._computeResults(E,nu);
    var d=new Float64Array(nDof);for(var i=0;i<nDof;i++)d[i]=sol[i];
    femData.disp=d;
    femData.stress=new Float64Array(M.HEAPF64.buffer,M._getStressData(),6*nE).slice();
    femData.strain=new Float64Array(M.HEAPF64.buffer,M._getStrainData(),6*nE).slice();
    currentView='disp';updBtns();buildVis('disp');
    var mx=0;for(var i=0;i<nN;i++)mx=Math.max(mx,Math.sqrt(d[3*i]**2+d[3*i+1]**2+d[3*i+2]**2));
    log('完成。最大位移: '+mx.toExponential(4)+' m，求解耗时: '+dt+' 秒');
    ${D}('btnRun').disabled=false;
}

function updBtns(){document.querySelectorAll('.view-btns button').forEach(function(b){b.classList.remove('active');});var m={disp:'btnDisp',strain:'btnStrain',stress:'btnStress',mesh:'btnMesh'};if(m[currentView])${D}(m[currentView]).classList.add('active');}
${D}('btnDisp').onclick=function(){currentView='disp';updBtns();buildVis('disp');};
${D}('btnStrain').onclick=function(){currentView='strain';updBtns();buildVis('strain');};
${D}('btnStress').onclick=function(){currentView='stress';updBtns();buildVis('stress');};
${D}('btnMesh').onclick=function(){currentView='mesh';updBtns();buildVis('mesh');};
${D}('btnRun').onclick=function(){runAnalysis().catch(function(e){log('错误: '+e.message);console.error(e);${D}('btnRun').disabled=false;});};
log('就绪。设置参数后点击"运行分析"。');
`;

fs.writeFileSync('D:/FEM_Web_Project/public/main.js', src, 'utf8');
console.log('Full main.js written OK, bytes:', src.length);
