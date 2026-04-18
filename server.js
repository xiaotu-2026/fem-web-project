// server.js — Local HTTP server with COOP/COEP headers for WebGPU + SharedArrayBuffer
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8080;
const PUBLIC = path.join(__dirname, 'public');

const MIME = {
    '.html': 'text/html',
    '.js':   'application/javascript',
    '.wasm': 'application/wasm',
    '.css':  'text/css',
    '.json': 'application/json',
    '.wgsl': 'text/plain',
    '.png':  'image/png',
    '.svg':  'image/svg+xml',
};

const server = http.createServer((req, res) => {
    let filePath = path.join(PUBLIC, req.url === '/' ? 'index.html' : req.url);
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME[ext] || 'application/octet-stream';

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('404 Not Found');
            return;
        }
        res.writeHead(200, {
            'Content-Type': contentType,
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(data);
    });
});

server.listen(PORT, () => {
    console.log(`FEA server running at http://localhost:${PORT}`);
    console.log('COOP/COEP headers enabled for WebGPU + SharedArrayBuffer');
});
