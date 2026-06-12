/* Wizard Chess mode: procedural 3D pieces (marble vs obsidian), animated
 * moves, and captures that SHATTER the victim - Harry Potter style.
 *
 * No chess logic lives here: the server's FEN is the source of truth. Every
 * position update is applied as a diff (moved / captured / promoted pieces
 * derived by comparing piece maps), so castling, en passant and promotion
 * animate correctly for free. Unmatched changes fall back to shatter/grow.
 *
 * Public API (window.Wizard):
 *   init(containerEl, onMoveCb)   - build scene into container
 *   setFen(fen, animate)          - apply a position (diff-animated or instant)
 *   setOrientation('white'|'black')
 *   setInteractive(bool)          - allow piece selection / target clicks
 *   resize()
 */
(function () {
    'use strict';

    const TILE = 1.0;
    const BOARD_Y = 0.0;
    const COLORS = {
        light: 0x9c9085, dark: 0x4b4340, rim: 0x322c28,
        white: 0xe9e2d0, black: 0x35323e,
        selected: 0x4f46e5,
    };

    let scene, camera, renderer, raycaster, container, onMove;
    let pieces = {};          // square -> {mesh, char}
    let squares = [];         // tile meshes for picking
    let selected = null;      // square string
    let tweens = [];
    let shards = [];
    let interactive = false;
    let orientationWhite = true;
    let camTheta = 0, camPhi = 1.05, camR = 11.5;
    let shake = 0;
    let animChain = Promise.resolve();

    // ---------------------------------------------------------------- utils

    function sqToXZ(file, rank) {
        return { x: (file - 3.5) * TILE, z: (3.5 - rank) * TILE };
    }

    function parseFen(fen) {
        const map = {};
        const rows = fen.split(' ')[0].split('/');
        for (let r = 0; r < 8; r++) {
            let f = 0;
            for (const ch of rows[r]) {
                if (/\d/.test(ch)) { f += parseInt(ch, 10); continue; }
                map['abcdefgh'[f] + (8 - r)] = ch;
                f++;
            }
        }
        return map;
    }

    function fileOf(sq) { return sq.charCodeAt(0) - 97; }
    function rankOf(sq) { return parseInt(sq[1], 10) - 1; }

    // ------------------------------------------------------------ materials

    function pieceMaterial(isWhite) {
        return new THREE.MeshStandardMaterial({
            color: isWhite ? COLORS.white : COLORS.black,
            roughness: isWhite ? 0.55 : 0.35,
            metalness: isWhite ? 0.05 : 0.25,
        });
    }

    // ------------------------------------------------------- piece geometry
    // Lathe profiles: arrays of [radius, height]. Stylized stone-set look.

    const PROFILES = {
        p: [[0.30, 0], [0.30, 0.06], [0.16, 0.12], [0.13, 0.42], [0.20, 0.50], [0.13, 0.55], [0.17, 0.68], [0.0, 0.80]],
        r: [[0.32, 0], [0.32, 0.08], [0.20, 0.16], [0.18, 0.70], [0.26, 0.74], [0.26, 0.95], [0.20, 0.95], [0.20, 0.88], [0.0, 0.88]],
        b: [[0.31, 0], [0.31, 0.06], [0.16, 0.14], [0.12, 0.62], [0.20, 0.72], [0.12, 0.82], [0.16, 0.94], [0.05, 1.04], [0.0, 1.08]],
        q: [[0.33, 0], [0.33, 0.06], [0.17, 0.16], [0.12, 0.74], [0.22, 0.86], [0.15, 0.96], [0.20, 1.06], [0.0, 1.16]],
        k: [[0.34, 0], [0.34, 0.06], [0.18, 0.16], [0.13, 0.80], [0.24, 0.92], [0.16, 1.02], [0.20, 1.10], [0.04, 1.16], [0.0, 1.18]],
    };

    function latheFrom(profile) {
        const pts = profile.map(p => new THREE.Vector2(p[0], p[1]));
        return new THREE.LatheGeometry(pts, 24);
    }

    function buildPieceMesh(char) {
        const isWhite = char === char.toUpperCase();
        const type = char.toLowerCase();
        const mat = pieceMaterial(isWhite);
        const group = new THREE.Group();

        if (type === 'n') {
            // stylized knight: lathe base + slanted head + muzzle + ears
            const base = new THREE.Mesh(latheFrom([[0.32, 0], [0.32, 0.07], [0.18, 0.16], [0.16, 0.34]]), mat);
            const neck = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.55, 0.30), mat);
            neck.position.set(0, 0.58, -0.02);
            neck.rotation.x = -0.25;
            const head = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.22, 0.46), mat);
            head.position.set(0, 0.86, 0.14);
            head.rotation.x = 0.35;
            const earL = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.14, 6), mat);
            earL.position.set(-0.07, 1.0, -0.02);
            const earR = earL.clone();
            earR.position.x = 0.07;
            group.add(base, neck, head, earL, earR);
        } else {
            group.add(new THREE.Mesh(latheFrom(PROFILES[type]), mat));
            if (type === 'r') {
                for (let i = 0; i < 4; i++) {
                    const tooth = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.10, 0.10), mat);
                    const a = i * Math.PI / 2 + Math.PI / 4;
                    tooth.position.set(Math.cos(a) * 0.20, 0.99, Math.sin(a) * 0.20);
                    group.add(tooth);
                }
            }
            if (type === 'k') {
                const v = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.20, 0.05), mat);
                v.position.y = 1.27;
                const h = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.05, 0.05), mat);
                h.position.y = 1.29;
                group.add(v, h);
            }
            if (type === 'q') {
                for (let i = 0; i < 5; i++) {
                    const a = i * 2 * Math.PI / 5;
                    const orb = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), mat);
                    orb.position.set(Math.cos(a) * 0.17, 1.1, Math.sin(a) * 0.17);
                    group.add(orb);
                }
            }
        }
        group.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });
        group.userData.char = char;
        const s = 0.92;
        group.scale.set(s, s, s);
        if (!isWhite) group.rotation.y = Math.PI;   // knights face the enemy
        return group;
    }

    // --------------------------------------------------------------- scene

    function buildBoard() {
        const board = new THREE.Group();
        const tileGeo = new THREE.BoxGeometry(TILE, 0.12, TILE);
        for (let r = 0; r < 8; r++) {
            for (let f = 0; f < 8; f++) {
                const dark = (f + r) % 2 === 0;
                const mat = new THREE.MeshStandardMaterial({
                    color: dark ? COLORS.dark : COLORS.light, roughness: 0.9,
                });
                const tile = new THREE.Mesh(tileGeo, mat);
                const { x, z } = sqToXZ(f, r);
                tile.position.set(x, BOARD_Y - 0.06, z);
                tile.receiveShadow = true;
                tile.userData.square = 'abcdefgh'[f] + (r + 1);
                tile.userData.baseColor = mat.color.getHex();
                board.add(tile);
                squares.push(tile);
            }
        }
        const rim = new THREE.Mesh(
            new THREE.BoxGeometry(8 * TILE + 0.7, 0.22, 8 * TILE + 0.7),
            new THREE.MeshStandardMaterial({ color: COLORS.rim, roughness: 0.8 }));
        rim.position.y = BOARD_Y - 0.13;
        rim.receiveShadow = true;
        board.add(rim);
        return board;
    }

    function updateCamera() {
        const az = camTheta + (orientationWhite ? 0 : Math.PI);
        camera.position.set(
            Math.sin(az) * Math.sin(camPhi) * camR,
            Math.cos(camPhi) * camR,
            Math.cos(az) * Math.sin(camPhi) * camR);
        camera.lookAt(0, 0.2, 0);
        if (shake > 0) {
            camera.position.x += (Math.random() - 0.5) * shake;
            camera.position.y += (Math.random() - 0.5) * shake;
            camera.position.z += (Math.random() - 0.5) * shake;
        }
    }

    function init(el, onMoveCb) {
        container = el;
        onMove = onMoveCb;
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x17151d);
        scene.fog = new THREE.Fog(0x17151d, 16, 30);

        camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        const amb = new THREE.AmbientLight(0x8d86a8, 0.55);
        const warm = new THREE.DirectionalLight(0xffd9a0, 1.15);
        warm.position.set(6, 10, 4);
        warm.castShadow = true;
        warm.shadow.mapSize.set(2048, 2048);
        warm.shadow.camera.left = warm.shadow.camera.bottom = -6;
        warm.shadow.camera.right = warm.shadow.camera.top = 6;
        const cool = new THREE.DirectionalLight(0x7a8cff, 0.35);
        cool.position.set(-7, 6, -5);
        scene.add(amb, warm, cool, buildBoard());

        raycaster = new THREE.Raycaster();
        bindInput();
        resize();
        requestAnimationFrame(loop);
    }

    function resize() {
        if (!renderer) return;
        const w = container.clientWidth, h = container.clientHeight;
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    // ----------------------------------------------------------- main loop

    function loop(t) {
        requestAnimationFrame(loop);
        const now = performance.now();
        tweens = tweens.filter(tw => {
            const k = Math.min(1, (now - tw.t0) / tw.dur);
            tw.fn(tw.ease ? tw.ease(k) : k);
            if (k >= 1 && tw.done) tw.done();
            return k < 1;
        });
        shards = shards.filter(s => {
            s.vel.y -= 0.0009 * 16;
            s.mesh.position.add(s.vel);
            s.mesh.rotation.x += s.rot.x;
            s.mesh.rotation.z += s.rot.z;
            s.life -= 0.016;
            s.mesh.material.opacity = Math.max(0, s.life / s.life0);
            if (s.life <= 0 || s.mesh.position.y < -2) {
                scene.remove(s.mesh);
                return false;
            }
            return true;
        });
        shake = Math.max(0, shake - 0.012);
        updateCamera();
        renderer.render(scene, camera);
    }

    function tween(dur, fn, done, ease) {
        tweens.push({ t0: performance.now(), dur, fn, done, ease });
    }
    const easeInOut = k => k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;

    // ------------------------------------------------------------ pieces

    function placePiece(sq, char) {
        const mesh = buildPieceMesh(char);
        const { x, z } = sqToXZ(fileOf(sq), rankOf(sq));
        mesh.position.set(x, BOARD_Y, z);
        scene.add(mesh);
        pieces[sq] = { mesh, char };
    }

    function clearAll() {
        for (const sq in pieces) scene.remove(pieces[sq].mesh);
        pieces = {};
        for (const s of shards) scene.remove(s.mesh);
        shards = [];
        tweens = [];
        animChain = Promise.resolve();
    }

    function setInstant(map) {
        clearAll();
        for (const sq in map) placePiece(sq, map[sq]);
    }

    // --------------------------------------------------------- destruction

    function shatter(entry, pos) {
        scene.remove(entry.mesh);
        const isWhite = entry.char === entry.char.toUpperCase();
        const mat0 = pieceMaterial(isWhite);
        const n = 16;
        for (let i = 0; i < n; i++) {
            const size = 0.06 + Math.random() * 0.12;
            const geo = new THREE.TetrahedronGeometry(size);
            const mat = mat0.clone();
            mat.transparent = true;
            const m = new THREE.Mesh(geo, mat);
            m.position.set(pos.x, BOARD_Y + 0.15 + Math.random() * 0.7, pos.z);
            const a = Math.random() * Math.PI * 2;
            const sp = 0.02 + Math.random() * 0.05;
            shards.push({
                mesh: m,
                vel: new THREE.Vector3(Math.cos(a) * sp, 0.04 + Math.random() * 0.05, Math.sin(a) * sp),
                rot: { x: (Math.random() - 0.5) * 0.3, z: (Math.random() - 0.5) * 0.3 },
                life: 1.1 + Math.random() * 0.4, life0: 1.5,
            });
            scene.add(m);
        }
        shake = 0.18;
    }

    // ------------------------------------------------------- move animation

    function animateMove(fromSq, toSq, victim, epVictimSq, newChar) {
        return new Promise(resolve => {
            const entry = pieces[fromSq];
            if (!entry) { resolve(); return; }
            delete pieces[fromSq];
            const mesh = entry.mesh;
            const from = mesh.position.clone();
            const { x, z } = sqToXZ(fileOf(toSq), rankOf(toSq));
            const isKnight = entry.char.toLowerCase() === 'n';
            const hop = isKnight ? 1.1 : 0.35;
            const dur = 620;

            tween(dur, k => {
                mesh.position.x = from.x + (x - from.x) * k;
                mesh.position.z = from.z + (z - from.z) * k;
                mesh.position.y = BOARD_Y + Math.sin(k * Math.PI) * hop;
            }, () => {
                mesh.position.set(x, BOARD_Y, z);
                if (victim) shatter(victim, { x, z });
                if (epVictimSq && pieces[epVictimSq]) {
                    const ep = pieces[epVictimSq];
                    delete pieces[epVictimSq];
                    const p = sqToXZ(fileOf(epVictimSq), rankOf(epVictimSq));
                    shatter(ep, p);
                }
                if (newChar && newChar !== entry.char) {
                    // promotion: swap the mesh with a little pop
                    scene.remove(mesh);
                    placePiece(toSq, newChar);
                    const nm = pieces[toSq].mesh;
                    nm.scale.set(0.01, 0.01, 0.01);
                    tween(260, k => { const s = 0.92 * k; nm.scale.set(s, s, s); });
                } else {
                    pieces[toSq] = entry;
                }
                resolve();
            }, easeInOut);
        });
    }

    // ------------------------------------------------------------ FEN diff

    function applyDiff(newMap) {
        const removed = [], added = [];
        const oldMap = {};
        for (const sq in pieces) oldMap[sq] = pieces[sq].char;
        for (const sq in oldMap) {
            if (newMap[sq] !== oldMap[sq]) removed.push({ sq, char: oldMap[sq] });
        }
        for (const sq in newMap) {
            if (oldMap[sq] !== newMap[sq]) added.push({ sq, char: newMap[sq] });
        }
        const moves = [];
        // pass 1: same piece char moved (castling = two of these)
        for (const add of added) {
            const i = removed.findIndex(r => r.char === add.char);
            if (i >= 0) {
                moves.push({ from: removed[i].sq, to: add.sq, char: add.char });
                removed.splice(i, 1);
                add.matched = true;
            }
        }
        // pass 2: promotion (pawn removed, new piece of same color added)
        for (const add of added.filter(a => !a.matched)) {
            const pawn = add.char === add.char.toUpperCase() ? 'P' : 'p';
            const i = removed.findIndex(r => r.char === pawn);
            if (i >= 0) {
                moves.push({ from: removed[i].sq, to: add.sq, char: add.char });
                removed.splice(i, 1);
                add.matched = true;
            }
        }
        const promises = [];
        for (const mv of moves) {
            const victim = pieces[mv.to];
            if (victim) delete pieces[mv.to];
            // en-passant victim: a leftover removal of the OTHER color pawn
            let epSq = null;
            const victimColorIsWhite = mv.char !== mv.char.toUpperCase();
            const epIdx = removed.findIndex(r =>
                r.char.toLowerCase() === 'p' &&
                (r.char === r.char.toUpperCase()) === victimColorIsWhite &&
                fileOf(r.sq) === fileOf(mv.to));
            if (!victim && epIdx >= 0) {
                epSq = removed[epIdx].sq;
                removed.splice(epIdx, 1);
            }
            promises.push(animateMove(mv.from, mv.to, victim || null, epSq, mv.char));
        }
        // fallbacks: anything still unexplained shatters / grows in
        for (const r of removed) {
            const entry = pieces[r.sq];
            if (entry) {
                delete pieces[r.sq];
                shatter(entry, sqToXZ(fileOf(r.sq), rankOf(r.sq)));
            }
        }
        for (const a of added.filter(x => !x.matched)) placePiece(a.sq, a.char);
        return Promise.all(promises);
    }

    function setFen(fen, animate) {
        const map = parseFen(fen);
        if (!animate) {
            animChain = animChain.then(() => setInstant(map));
        } else {
            animChain = animChain.then(() => applyDiff(map));
        }
        return animChain;
    }

    // -------------------------------------------------------------- input

    function highlight(sq, on) {
        const tile = squares.find(t => t.userData.square === sq);
        if (tile) tile.material.color.setHex(on ? COLORS.selected : tile.userData.baseColor);
        const entry = pieces[sq];
        if (entry) entry.mesh.position.y = on ? 0.18 : BOARD_Y;
    }

    function pick(ev) {
        const rect = renderer.domElement.getBoundingClientRect();
        const ndc = new THREE.Vector2(
            ((ev.clientX - rect.left) / rect.width) * 2 - 1,
            -((ev.clientY - rect.top) / rect.height) * 2 + 1);
        raycaster.setFromCamera(ndc, camera);
        const meshes = [...squares];
        for (const sq in pieces) pieces[sq].mesh.traverse(o => { if (o.isMesh) { o.userData.pieceSq = sq; meshes.push(o); } });
        const hits = raycaster.intersectObjects(meshes, false);
        if (!hits.length) return null;
        const obj = hits[0].object;
        return obj.userData.pieceSq || obj.userData.square || null;
    }

    function bindInput() {
        let down = null, dragged = false;
        const el = renderer.domElement;
        el.addEventListener('pointerdown', ev => { down = { x: ev.clientX, y: ev.clientY, b: ev.button }; dragged = false; });
        el.addEventListener('pointermove', ev => {
            if (!down) return;
            const dx = ev.clientX - down.x, dy = ev.clientY - down.y;
            if (Math.abs(dx) + Math.abs(dy) > 6) dragged = true;
            if (dragged) {
                camTheta -= dx * 0.008;
                camPhi = Math.min(1.35, Math.max(0.35, camPhi - dy * 0.005));
                down = { x: ev.clientX, y: ev.clientY };
            }
        });
        el.addEventListener('pointerup', ev => {
            const wasDrag = dragged;
            down = null; dragged = false;
            if (wasDrag || !interactive) return;
            const sq = pick(ev);
            if (!sq) return;
            const mine = pieces[sq] && (pieces[sq].char === pieces[sq].char.toUpperCase()) === (window.Wizard._playerWhite);
            if (selected === null) {
                if (mine) { selected = sq; highlight(sq, true); }
            } else if (sq === selected) {
                highlight(selected, false); selected = null;
            } else if (mine) {
                highlight(selected, false); selected = sq; highlight(sq, true);
            } else {
                const from = selected;
                highlight(selected, false); selected = null;
                onMove(from, sq, pieces[from] ? pieces[from].char : null);
            }
        });
        el.addEventListener('wheel', ev => {
            ev.preventDefault();
            camR = Math.min(18, Math.max(7, camR + ev.deltaY * 0.01));
        }, { passive: false });
    }

    window.Wizard = {
        init, setFen, resize,
        _playerWhite: true,
        setOrientation(color) { orientationWhite = color === 'white'; this._playerWhite = orientationWhite; },
        setInteractive(b) { interactive = b; if (!b && selected) { highlight(selected, false); selected = null; } },
    };
})();
