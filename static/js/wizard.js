/* Wizard Chess mode - dark catacomb edition.
 *
 * Polished marble vs obsidian-glass pieces (MeshPhysicalMaterial + generated
 * environment map), procedural stone textures, flickering corner torches,
 * drifting dust motes, fog and distant columns.
 *
 * Captures play a staged death: impact flash -> the victim TOPPLES over
 * (pivoting on its base edge) -> crumbles into a burst of dust and shards ->
 * slowly SINKS through the board while fading -> an ash scorch + rubble
 * remains on the square for the rest of the game.
 *
 * No chess logic here: server FENs are applied as piece-map diffs, so
 * castling / en passant / promotion animate correctly for free.
 *
 * Public API (window.Wizard): init, setFen, setOrientation, setInteractive,
 * resize - unchanged from v1.
 */
(function () {
    'use strict';

    const TILE = 1.0;
    const BOARD_Y = 0.0;
    const SELECT_COLOR = 0x7c6cff;

    let scene, camera, renderer, raycaster, container, onMove;
    let pieces = {};
    let squares = [];
    let selected = null;
    let tweens = [];
    let particleSystems = [];
    let torches = [];
    let motes = null;
    let fxGroup = null;       // death pivots + flash lights; cleared per game
    let interactive = false;
    let orientationWhite = true;
    let camTheta = 0, camPhi = 1.12, camR = 11.0;
    let shake = 0;
    let animChain = Promise.resolve();
    let matWhite, matBlack, envTex;
    let levitation = null;        // magically activated (selected) piece
    let lastSpark = 0;

    // ---------------------------------------------------------------- utils

    const sqToXZ = (f, r) => ({ x: (f - 3.5) * TILE, z: (3.5 - r) * TILE });
    const fileOf = sq => sq.charCodeAt(0) - 97;
    const rankOf = sq => parseInt(sq[1], 10) - 1;

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

    function tween(dur, fn, done, ease) {
        tweens.push({ t0: performance.now(), dur, fn, done, ease });
    }
    const easeInOut = k => k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    const easeIn = k => k * k * k;
    const easeOut = k => 1 - Math.pow(1 - k, 3);

    // ------------------------------------------------------ canvas textures

    function canvasTex(size, draw) {
        const c = document.createElement('canvas');
        c.width = c.height = size;
        draw(c.getContext('2d'), size);
        const t = new THREE.CanvasTexture(c);
        t.anisotropy = 4;
        return t;
    }

    function stoneTexture(base, blotch, n) {
        return canvasTex(256, (g, s) => {
            g.fillStyle = base; g.fillRect(0, 0, s, s);
            for (let i = 0; i < n; i++) {
                const a = 0.03 + Math.random() * 0.10;
                g.fillStyle = blotch.replace('A', a.toFixed(3));
                const r = 4 + Math.random() * 34;
                g.beginPath();
                g.arc(Math.random() * s, Math.random() * s, r, 0, 7);
                g.fill();
            }
            // cracks
            g.strokeStyle = 'rgba(0,0,0,0.16)';
            for (let i = 0; i < 5; i++) {
                g.beginPath();
                let x = Math.random() * s, y = Math.random() * s;
                g.moveTo(x, y);
                for (let j = 0; j < 6; j++) {
                    x += (Math.random() - 0.5) * 60; y += (Math.random() - 0.5) * 60;
                    g.lineTo(x, y);
                }
                g.lineWidth = 0.6 + Math.random();
                g.stroke();
            }
        });
    }

    function marbleTexture() {
        return canvasTex(256, (g, s) => {
            g.fillStyle = '#e8e0cf'; g.fillRect(0, 0, s, s);
            for (let i = 0; i < 26; i++) {           // soft tonal clouds
                g.fillStyle = `rgba(190,180,160,${0.04 + Math.random() * 0.07})`;
                g.beginPath();
                g.arc(Math.random() * s, Math.random() * s, 20 + Math.random() * 50, 0, 7);
                g.fill();
            }
            for (let i = 0; i < 7; i++) {            // veins
                g.strokeStyle = `rgba(120,110,95,${0.10 + Math.random() * 0.12})`;
                g.lineWidth = 0.5 + Math.random() * 0.9;
                g.beginPath();
                let x = Math.random() * s, y = 0;
                g.moveTo(x, y);
                while (y < s) {
                    x += (Math.random() - 0.5) * 26; y += 8 + Math.random() * 18;
                    g.lineTo(x, y);
                }
                g.stroke();
            }
        });
    }

    function softDot(color) {
        return canvasTex(64, (g, s) => {
            const grad = g.createRadialGradient(s/2, s/2, 1, s/2, s/2, s/2);
            grad.addColorStop(0, color);
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            g.fillStyle = grad;
            g.fillRect(0, 0, s, s);
        });
    }

    // -------------------------------------------------------- environment

    function buildEnvironment() {
        // tiny "room" rendered into a PMREM env map: warm torch strips and a
        // cold moon strip make the obsidian glass actually reflect something
        const env = new THREE.Scene();
        env.background = new THREE.Color(0x06050a);
        const strip = (color, intensity, x, y, z, w, h) => {
            const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
                new THREE.MeshBasicMaterial({ color }));
            m.material.color.multiplyScalar(intensity);
            m.position.set(x, y, z);
            m.lookAt(0, 0, 0);
            env.add(m);
        };
        strip(0xff9a3d, 3.5, 6, 3, 5, 3, 5);
        strip(0xff7a2d, 2.5, -6, 2, -4, 3, 4);
        strip(0x8fa0ff, 1.6, -3, 6, 6, 6, 2);
        strip(0xfff1d6, 1.2, 0, 8, 0, 3, 3);
        const pmrem = new THREE.PMREMGenerator(renderer);
        envTex = pmrem.fromScene(env, 0.05).texture;
        pmrem.dispose();
        scene.environment = envTex;
    }

    // ----------------------------------------------------------- materials

    function makeMaterials() {
        matWhite = new THREE.MeshPhysicalMaterial({
            map: marbleTexture(),
            color: 0xf5ecd8,
            roughness: 0.34, metalness: 0.0,
            clearcoat: 0.55, clearcoatRoughness: 0.35,
            envMapIntensity: 0.7,
        });
        matBlack = new THREE.MeshPhysicalMaterial({
            color: 0x0d0b13,
            roughness: 0.06, metalness: 0.08,
            clearcoat: 1.0, clearcoatRoughness: 0.08,
            transmission: 0.18, thickness: 0.8, ior: 1.45,
            envMapIntensity: 1.6,
        });
    }
    const pieceMaterial = isWhite => (isWhite ? matWhite : matBlack).clone();

    // ------------------------------------------------------- piece geometry
    // Detailed lathe profiles ([radius, height]) with collar rings and beads.

    const PROFILES = {
        p: [[0.30,0],[0.30,0.05],[0.27,0.08],[0.17,0.12],[0.145,0.16],[0.125,0.36],
            [0.175,0.42],[0.19,0.45],[0.115,0.50],[0.105,0.53],[0.165,0.62],[0.155,0.70],[0.09,0.78],[0,0.82]],
        r: [[0.33,0],[0.33,0.05],[0.30,0.09],[0.205,0.14],[0.185,0.18],[0.165,0.58],
            [0.185,0.62],[0.165,0.66],[0.245,0.72],[0.255,0.92],[0.215,0.92],[0.215,0.84],[0.0,0.84]],
        b: [[0.32,0],[0.32,0.05],[0.29,0.08],[0.175,0.13],[0.15,0.17],[0.115,0.52],
            [0.165,0.58],[0.175,0.61],[0.10,0.66],[0.185,0.78],[0.14,0.90],[0.055,0.99],[0.085,1.03],[0.0,1.10]],
        q: [[0.34,0],[0.34,0.05],[0.31,0.09],[0.19,0.14],[0.16,0.18],[0.115,0.62],
            [0.185,0.70],[0.20,0.74],[0.125,0.79],[0.115,0.83],[0.225,0.95],[0.16,1.02],[0.19,1.10],[0.06,1.16],[0,1.20]],
        k: [[0.35,0],[0.35,0.05],[0.32,0.09],[0.20,0.15],[0.165,0.19],[0.125,0.68],
            [0.195,0.76],[0.21,0.80],[0.13,0.85],[0.12,0.89],[0.235,1.00],[0.165,1.08],[0.185,1.14],[0.05,1.20],[0,1.22]],
    };

    const latheFrom = prof => new THREE.LatheGeometry(prof.map(p => new THREE.Vector2(p[0], p[1])), 48);

    function buildPieceMesh(char) {
        const isWhite = char === char.toUpperCase();
        const type = char.toLowerCase();
        const mat = pieceMaterial(isWhite);
        const group = new THREE.Group();
        const add = m => { group.add(m); return m; };

        if (type === 'n') {
            add(new THREE.Mesh(latheFrom([[0.33,0],[0.33,0.05],[0.30,0.09],[0.21,0.14],[0.19,0.20],[0.17,0.34],[0.21,0.38]]), mat));
            // arched neck from stacked slabs
            const seg = [[0, 0.46, 0.02, -0.10], [0, 0.60, 0.05, -0.22], [0, 0.74, 0.09, -0.30]];
            for (const [x, y, z, rx] of seg) {
                const slab = add(new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.20, 0.30), mat));
                slab.position.set(x, y, z);
                slab.rotation.x = rx;
            }
            const head = add(new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.18, 0.46), mat));
            head.position.set(0, 0.88, 0.20);
            head.rotation.x = 0.42;
            const muzzle = add(new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.12, 0.16), mat));
            muzzle.position.set(0, 0.80, 0.40);
            muzzle.rotation.x = 0.42;
            for (const sx of [-0.065, 0.065]) {
                const ear = add(new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.16, 6), mat));
                ear.position.set(sx, 1.03, 0.06);
                ear.rotation.x = -0.2;
            }
            for (let i = 0; i < 4; i++) {            // mane ridges
                const ridge = add(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.16, 0.05), mat));
                ridge.position.set(0, 0.52 + i * 0.13, -0.13 - i * 0.022);
                ridge.rotation.x = -0.25;
            }
        } else {
            add(new THREE.Mesh(latheFrom(PROFILES[type]), mat));
            if (type === 'r') {
                for (let i = 0; i < 5; i++) {
                    const a = i * 2 * Math.PI / 5;
                    const tooth = add(new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.12, 0.09), mat));
                    tooth.position.set(Math.cos(a) * 0.20, 0.97, Math.sin(a) * 0.20);
                    tooth.rotation.y = -a;
                }
            }
            if (type === 'q') {
                for (let i = 0; i < 6; i++) {
                    const a = i * Math.PI / 3;
                    const orb = add(new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 10), mat));
                    orb.position.set(Math.cos(a) * 0.175, 1.04, Math.sin(a) * 0.175);
                }
                add(new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 10), mat)).position.y = 1.22;
            }
            if (type === 'k') {
                const v = add(new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.24, 0.055), mat));
                v.position.y = 1.32;
                const h = add(new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.055, 0.055), mat));
                h.position.y = 1.345;
            }
            if (type === 'b') {
                add(new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 8), mat)).position.y = 1.115;
            }
        }
        group.traverse(o => { if (o.isMesh) o.castShadow = true; });
        group.userData.char = char;
        group.userData.baseRadius = 0.33;
        const s = 0.92;
        group.scale.set(s, s, s);
        if (!isWhite) group.rotation.y = Math.PI;
        if (type !== 'n') group.rotation.y += (Math.random() - 0.5) * 0.3;  // hand-set look
        return group;
    }

    // --------------------------------------------------------------- scene

    function buildBoard() {
        const g = new THREE.Group();
        const texDark = stoneTexture('#2e2823', 'rgba(0,0,0,A)', 60);
        const texLight = stoneTexture('#857867', 'rgba(30,20,12,A)', 60);
        const tileGeo = new THREE.BoxGeometry(TILE * 0.985, 0.14, TILE * 0.985);
        for (let r = 0; r < 8; r++) {
            for (let f = 0; f < 8; f++) {
                const dark = (f + r) % 2 === 0;
                const mat = new THREE.MeshStandardMaterial({
                    map: dark ? texDark : texLight,
                    roughness: 0.85, metalness: 0.05,
                });
                const tile = new THREE.Mesh(tileGeo, mat);
                const { x, z } = sqToXZ(f, r);
                tile.position.set(x, BOARD_Y - 0.07, z);
                tile.receiveShadow = true;
                tile.userData.square = 'abcdefgh'[f] + (r + 1);
                g.add(tile);
                squares.push(tile);
            }
        }
        const rim = new THREE.Mesh(
            new THREE.BoxGeometry(8 * TILE + 0.9, 0.26, 8 * TILE + 0.9),
            new THREE.MeshStandardMaterial({ map: stoneTexture('#1c1714', 'rgba(0,0,0,A)', 80), roughness: 0.9 }));
        rim.position.y = BOARD_Y - 0.16;
        rim.receiveShadow = true;
        g.add(rim);
        return g;
    }

    function buildCatacombs() {
        const g = new THREE.Group();
        const floorMat = new THREE.MeshStandardMaterial({
            map: stoneTexture('#15110f', 'rgba(0,0,0,A)', 90), roughness: 1.0 });
        floorMat.map.wrapS = floorMat.map.wrapT = THREE.RepeatWrapping;
        floorMat.map.repeat.set(8, 8);
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(70, 70), floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.30;
        floor.receiveShadow = true;
        g.add(floor);

        const colMat = new THREE.MeshStandardMaterial({
            map: stoneTexture('#221c18', 'rgba(0,0,0,A)', 70), roughness: 0.95 });
        for (let i = 0; i < 10; i++) {
            const a = i * Math.PI / 5 + 0.3;
            const r = 11 + (i % 3) * 2.5;
            const col = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.85, 11, 10), colMat);
            col.position.set(Math.cos(a) * r, 5.2, Math.sin(a) * r);
            g.add(col);
            const cap = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.5, 2.0), colMat);
            cap.position.set(Math.cos(a) * r, 0.0, Math.sin(a) * r);
            g.add(cap);
        }
        return g;
    }

    function buildTorches() {
        const glowTex = softDot('rgba(255,166,77,0.9)');
        for (const [sx, sz] of [[-5.2, -5.2], [5.2, -5.2], [-5.2, 5.2], [5.2, 5.2]]) {
            const pole = new THREE.Mesh(
                new THREE.CylinderGeometry(0.07, 0.10, 2.6, 8),
                new THREE.MeshStandardMaterial({ color: 0x17120e, roughness: 0.9 }));
            pole.position.set(sx, 1.0, sz);
            const bowl = new THREE.Mesh(
                new THREE.CylinderGeometry(0.16, 0.08, 0.18, 8),
                new THREE.MeshStandardMaterial({ color: 0x241a12, roughness: 0.8 }));
            bowl.position.set(sx, 2.35, sz);
            const light = new THREE.PointLight(0xff8c3a, 1.5, 13, 2.0);
            light.position.set(sx, 2.75, sz);
            const glow = new THREE.Sprite(new THREE.SpriteMaterial({
                map: glowTex, color: 0xffb066, transparent: true,
                blending: THREE.AdditiveBlending, depthWrite: false }));
            glow.position.set(sx, 2.62, sz);
            glow.scale.set(1.5, 1.9, 1);
            scene.add(pole, bowl, light, glow);
            torches.push({ light, glow, phase: Math.random() * 9 });
        }
    }

    function buildMotes() {
        const n = 150;
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
            pos[i*3] = (Math.random() - 0.5) * 16;
            pos[i*3+1] = Math.random() * 6;
            pos[i*3+2] = (Math.random() - 0.5) * 16;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        motes = new THREE.Points(geo, new THREE.PointsMaterial({
            map: softDot('rgba(255,220,170,0.55)'), size: 0.05, transparent: true,
            opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false }));
        scene.add(motes);
    }

    function updateCamera() {
        const az = camTheta + (orientationWhite ? 0 : Math.PI);
        camera.position.set(
            Math.sin(az) * Math.sin(camPhi) * camR,
            Math.cos(camPhi) * camR,
            Math.cos(az) * Math.sin(camPhi) * camR);
        camera.lookAt(0, 0.1, 0);
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
        scene.background = new THREE.Color(0x0a0810);
        scene.fog = new THREE.FogExp2(0x0a0810, 0.040);

        camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.outputEncoding = THREE.sRGBEncoding;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.08;
        container.appendChild(renderer.domElement);

        // CSS vignette for the catacomb feel - cheap "post-processing"
        const vig = document.createElement('div');
        vig.style.cssText = 'position:absolute;inset:0;pointer-events:none;border-radius:inherit;' +
            'background:radial-gradient(ellipse at 50% 42%, transparent 52%, rgba(2,1,6,0.55) 100%);';
        container.style.position = 'relative';
        container.appendChild(vig);

        buildEnvironment();
        makeMaterials();

        scene.add(new THREE.AmbientLight(0x4a4060, 0.7));
        const moon = new THREE.DirectionalLight(0x9aa6ff, 0.5);
        moon.position.set(-6, 11, -4);
        moon.castShadow = true;
        moon.shadow.mapSize.set(2048, 2048);
        moon.shadow.camera.left = moon.shadow.camera.bottom = -7;
        moon.shadow.camera.right = moon.shadow.camera.top = 7;
        scene.add(moon);

        renderer.localClippingEnabled = true;   // disintegration sweep
        scene.add(buildBoard());
        scene.add(buildCatacombs());
        buildTorches();
        buildMotes();
        fxGroup = new THREE.Group();
        scene.add(fxGroup);

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

    function loop(tms) {
        requestAnimationFrame(loop);
        const now = performance.now();
        // Snapshot processing: done() callbacks routinely schedule FOLLOW-UP
        // tweens (topple -> ashify -> disintegrate). A plain
        // `tweens = tweens.filter(...)` drops anything pushed during the
        // filter pass - that bug left un-destroyed "phantom" pieces behind.
        const processing = tweens;
        tweens = [];
        const survivors = [];
        for (const tw of processing) {
            const k = Math.min(1, (now - tw.t0) / tw.dur);
            tw.fn(tw.ease ? tw.ease(k) : k);
            if (k < 1) survivors.push(tw);
            else if (tw.done) tw.done();
        }
        tweens = survivors.concat(tweens);
        const psProcessing = particleSystems;
        particleSystems = [];
        const psSurvivors = psProcessing.filter(ps => ps.update(now));
        particleSystems = psSurvivors.concat(particleSystems);
        for (const t of torches) {
            const fl = Math.sin(now * 0.011 + t.phase) * 0.18 +
                       Math.sin(now * 0.027 + t.phase * 2) * 0.10 +
                       (Math.random() - 0.5) * 0.10;
            t.light.intensity = 1.45 + fl;
            t.glow.material.opacity = 0.75 + fl * 0.3;
            const s = 1.5 + fl * 0.25;
            t.glow.scale.set(s, s * 1.3, 1);
        }
        if (motes) {
            const p = motes.geometry.attributes.position;
            for (let i = 0; i < p.count; i++) {
                let y = p.getY(i) - 0.0012;
                if (y < 0) y = 6;
                p.setY(i, y);
                p.setX(i, p.getX(i) + Math.sin(now * 0.0004 + i) * 0.0012);
            }
            p.needsUpdate = true;
        }
        if (levitation) {
            const lev = levitation;
            // float as if held by a spell: smooth bob + slow yaw sway
            const target = 0.26 + Math.sin(now * 0.0032) * 0.055;
            lev.mesh.position.y += (target - lev.mesh.position.y) * 0.10;
            lev.mesh.rotation.y = lev.baseRotY + Math.sin(now * 0.0017) * 0.07;
            const pulse = 0.55 + Math.sin(now * 0.005) * 0.2;
            lev.glow.material.opacity = pulse;
            const gs = 0.95 + Math.sin(now * 0.005) * 0.1;
            lev.glow.scale.set(gs, gs * 0.55, 1);
            if (now - lastSpark > 320) {
                lastSpark = now;
                sparkle(lev.mesh.position.x, lev.mesh.position.z);
            }
        }
        shake = Math.max(0, shake - 0.010);
        updateCamera();
        renderer.render(scene, camera);
    }

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
        if (fxGroup) fxGroup.clear();   // orphaned mid-death pivots/lights
        if (levitation) { scene.remove(levitation.glow); levitation = null; }
        selected = null;
        tweens = [];
        for (const ps of particleSystems) ps.dispose();
        particleSystems = [];
        animChain = Promise.resolve();
    }

    function setInstant(map) {
        clearAll();
        for (const sq in map) placePiece(sq, map[sq]);
    }

    // ----------------------------------------------------- capture effects

    function dustBurst(pos, isWhite, count, spread, upward) {
        const color = isWhite ? 'rgba(220,210,190,0.85)' : 'rgba(80,70,100,0.85)';
        const n = count;
        const positions = new Float32Array(n * 3);
        const vels = [];
        for (let i = 0; i < n; i++) {
            positions[i*3] = pos.x + (Math.random() - 0.5) * 0.25;
            positions[i*3+1] = BOARD_Y + 0.05 + Math.random() * 0.55;
            positions[i*3+2] = pos.z + (Math.random() - 0.5) * 0.25;
            const a = Math.random() * Math.PI * 2;
            const sp = Math.random() * spread;
            vels.push(new THREE.Vector3(Math.cos(a) * sp, upward * (0.4 + Math.random()), Math.sin(a) * sp));
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({
            map: softDot(color), size: 0.10 + Math.random() * 0.05, transparent: true,
            opacity: 0.9, depthWrite: false });
        const points = new THREE.Points(geo, mat);
        scene.add(points);
        const t0 = performance.now();
        const life = 2100;
        particleSystems.push({
            update(now) {
                const k = (now - t0) / life;
                if (k >= 1) { this.dispose(); return false; }
                const p = points.geometry.attributes.position;
                for (let i = 0; i < n; i++) {
                    const v = vels[i];
                    v.y -= 0.0030;                       // gravity
                    v.multiplyScalar(0.985);             // drag
                    let y = p.getY(i) + v.y * 0.016;
                    if (y < BOARD_Y + 0.015) { y = BOARD_Y + 0.015; v.set(0, 0, 0); }
                    p.setXYZ(i, p.getX(i) + v.x * 0.016, y, p.getZ(i) + v.z * 0.016);
                }
                p.needsUpdate = true;
                mat.opacity = 0.9 * (1 - k * k);
                return true;
            },
            dispose() { scene.remove(points); geo.dispose(); mat.dispose(); },
        });
    }

    function sparkle(x, z) {
        // a few violet motes rising around a levitating piece
        const n = 5;
        const positions = new Float32Array(n * 3);
        const vels = [];
        for (let i = 0; i < n; i++) {
            const a = Math.random() * Math.PI * 2, d = 0.18 + Math.random() * 0.22;
            positions[i*3] = x + Math.cos(a) * d;
            positions[i*3+1] = BOARD_Y + 0.05 + Math.random() * 0.3;
            positions[i*3+2] = z + Math.sin(a) * d;
            vels.push(0.004 + Math.random() * 0.006);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({
            map: softDot('rgba(170,150,255,0.9)'), size: 0.07, transparent: true,
            opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false });
        const points = new THREE.Points(geo, mat);
        scene.add(points);
        const t0 = performance.now(), life = 1100;
        particleSystems.push({
            update(now) {
                const k = (now - t0) / life;
                if (k >= 1) { this.dispose(); return false; }
                const p = points.geometry.attributes.position;
                for (let i = 0; i < n; i++) p.setY(i, p.getY(i) + vels[i]);
                p.needsUpdate = true;
                mat.opacity = 0.9 * (1 - k);
                return true;
            },
            dispose() { scene.remove(points); geo.dispose(); mat.dispose(); },
        });
    }

    function impactFlash(pos) {
        const fl = new THREE.PointLight(0xffc080, 3.2, 6, 2.0);
        fl.position.set(pos.x, 1.0, pos.z);
        fxGroup.add(fl);
        tween(420, k => { fl.intensity = 3.2 * (1 - k); }, () => fxGroup.remove(fl), easeOut);
        shake = 0.20;
    }

    function disintegrate(pivot, mesh, basePoint, dir, height, isWhite, sideAxis) {
        // a clipping plane sweeps from the head of the FALLEN piece toward
        // its base; the piece visibly erodes while dense ash streams off the
        // erosion front and drifts away with the wind. Nothing remains.
        const mats = new Set();
        mesh.traverse(o => { if (o.isMesh) { mats.add(o.material); o.castShadow = false; } });
        const plane = new THREE.Plane(dir.clone().negate(), 1000);
        mats.forEach(m => { m.clippingPlanes = [plane]; m.side = THREE.DoubleSide; });

        const headPoint = basePoint.clone().add(dir.clone().multiplyScalar(height));
        const cStart = dir.dot(headPoint) + 0.12;
        const cEnd = dir.dot(basePoint) - 0.06;

        // streaming dust pool
        const N = 620;
        const positions = new Float32Array(N * 3).fill(-100);
        const vels = new Array(N);
        let next = 0;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({
            map: softDot(isWhite ? 'rgba(216,206,186,0.9)' : 'rgba(110,100,130,0.9)'),
            size: 0.085, transparent: true, opacity: 0.95, depthWrite: false });
        const points = new THREE.Points(geo, mat);
        scene.add(points);
        const wind = sideAxis.clone().multiplyScalar((Math.random() < 0.5 ? -1 : 1) * (0.005 + Math.random() * 0.004));
        wind.y = 0.011;
        const up = new THREE.Vector3(0, 1, 0);

        function spawn(front, radius, count) {
            for (let i = 0; i < count; i++) {
                const j = next % N;
                next++;
                const a = Math.random() * Math.PI * 2;
                const r = Math.random() * radius;
                const off = up.clone().multiplyScalar(Math.cos(a) * r + radius * 0.6)
                    .add(sideAxis.clone().multiplyScalar(Math.sin(a) * r));
                positions[j*3] = front.x + off.x;
                positions[j*3+1] = Math.max(0.03, front.y + off.y);
                positions[j*3+2] = front.z + off.z;
                vels[j] = wind.clone().add(new THREE.Vector3(
                    (Math.random() - 0.5) * 0.006,
                    Math.random() * 0.008,
                    (Math.random() - 0.5) * 0.006));
            }
            geo.attributes.position.needsUpdate = true;
        }

        let fading = false;
        particleSystems.push({
            update() {
                const p = geo.attributes.position;
                for (let j = 0; j < N; j++) {
                    if (!vels[j]) continue;
                    p.setXYZ(j, p.getX(j) + vels[j].x, p.getY(j) + vels[j].y, p.getZ(j) + vels[j].z);
                    vels[j].multiplyScalar(0.992);
                }
                p.needsUpdate = true;
                return !this.dead;
            },
            dispose() { scene.remove(points); geo.dispose(); mat.dispose(); },
        });
        const system = particleSystems[particleSystems.length - 1];

        tween(1600, k => {
            plane.constant = cStart + (cEnd - cStart) * k;
            const front = basePoint.clone().add(dir.clone().multiplyScalar(height * (1 - k)));
            const radius = 0.07 + 0.20 * k;          // pieces widen toward the base
            spawn(front, radius, 6);
        }, () => {
            fxGroup.remove(pivot);
            // let the cloud drift, then breathe out
            tween(1300, () => {}, () => {
                tween(700, k => { mat.opacity = 0.95 * (1 - k); },
                    () => { system.dead = true; system.dispose(); });
            });
        }, easeInOut);
    }

    function destroyPiece(entry, sq, attackDir) {
        // staged death: topple over the base edge -> turn to ash -> the ash
        // sweep disintegrates the piece into drifting dust. No remains.
        const mesh = entry.mesh;
        const isWhite = entry.char === entry.char.toUpperCase();
        const { x, z } = sqToXZ(fileOf(sq), rankOf(sq));
        const dir = attackDir.lengthSq() > 0 ? attackDir.clone().normalize()
                                             : new THREE.Vector3(0, 0, 1);
        const sideAxis = new THREE.Vector3(dir.z, 0, -dir.x);

        // pivot at the base edge in fall direction
        const pivot = new THREE.Group();
        const edge = 0.30;
        pivot.position.set(x + dir.x * edge, BOARD_Y, z + dir.z * edge);
        fxGroup.add(pivot);
        scene.remove(mesh);
        mesh.position.set(-dir.x * edge, 0, -dir.z * edge);
        pivot.add(mesh);

        impactFlash({ x, z });
        dustBurst({ x, z }, isWhite, 40, 0.03, 0.05);

        tween(640, k => {
            // fall past vertical with a hard landing thud
            const ang = k < 0.85 ? (k / 0.85) * 1.45 : 1.45 + Math.sin((k - 0.85) / 0.15 * Math.PI) * 0.06;
            pivot.setRotationFromAxisAngle(sideAxis.clone().negate(), -ang);
        }, () => {
            shake = Math.max(shake, 0.12);
            dustBurst({ x: x + dir.x * 0.6, z: z + dir.z * 0.6 }, isWhite, 45, 0.05, 0.07);

            // ashify: the stone loses its lustre before crumbling
            const mats = new Set();
            mesh.traverse(o => { if (o.isMesh) mats.add(o.material); });
            const ash = new THREE.Color(isWhite ? 0xb6ac9b : 0x4b4555);
            mats.forEach(m => {
                const c0 = m.color.clone();
                const cc0 = m.clearcoat || 0, tr0 = m.transmission || 0, env0 = m.envMapIntensity;
                tween(340, k => {
                    m.color.lerpColors(c0, ash, k);
                    m.roughness = Math.min(1, m.roughness + k);
                    m.clearcoat = cc0 * (1 - k);
                    if (m.transmission !== undefined) m.transmission = tr0 * (1 - k);
                    m.envMapIntensity = env0 * (1 - k * 0.85);
                });
            });

            // the fallen piece now lies along `dir` from the pivot point
            const basePoint = pivot.position.clone();
            basePoint.y = 0.12;
            const lieDir = dir.clone();
            setTimeout(() => disintegrate(pivot, mesh, basePoint, lieDir, 1.15, isWhite, sideAxis), 380);
        }, easeIn);
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
            const hop = isKnight ? 1.15 : 0.30;
            const dur = isKnight ? 700 : 640;
            const attackDir = new THREE.Vector3(x - from.x, 0, z - from.z);

            tween(dur, k => {
                mesh.position.x = from.x + (x - from.x) * k;
                mesh.position.z = from.z + (z - from.z) * k;
                mesh.position.y = BOARD_Y + Math.sin(k * Math.PI) * hop;
                if (!isKnight) {                       // subtle stone-grind sway
                    mesh.rotation.z = Math.sin(k * Math.PI * 2) * 0.03;
                }
            }, () => {
                mesh.position.set(x, BOARD_Y, z);
                mesh.rotation.z = 0;
                // landing squash for weight
                tween(180, k => {
                    const sq_ = 1 - Math.sin(k * Math.PI) * 0.06;
                    mesh.scale.set(0.92 / Math.sqrt(sq_), 0.92 * sq_, 0.92 / Math.sqrt(sq_));
                });
                if (victim) destroyPiece(victim, toSq, attackDir);
                if (epVictimSq && pieces[epVictimSq]) {
                    const ep = pieces[epVictimSq];
                    delete pieces[epVictimSq];
                    destroyPiece(ep, epVictimSq, attackDir);
                }
                if (newChar && newChar !== entry.char) {
                    scene.remove(mesh);
                    placePiece(toSq, newChar);
                    const nm = pieces[toSq].mesh;
                    nm.scale.set(0.01, 0.01, 0.01);
                    dustBurst({ x, z }, newChar === newChar.toUpperCase(), 40, 0.03, 0.3);
                    tween(320, k => { const s = 0.92 * k; nm.scale.set(s, s, s); }, null, easeOut);
                } else {
                    pieces[toSq] = entry;
                }
                // resolve a beat after impact so the death registers visually
                setTimeout(resolve, victim || epVictimSq ? 450 : 60);
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
        for (const add of added) {
            const i = removed.findIndex(r => r.char === add.char);
            if (i >= 0) {
                moves.push({ from: removed[i].sq, to: add.sq, char: add.char });
                removed.splice(i, 1);
                add.matched = true;
            }
        }
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
            let epSq = null;
            const victimIsWhite = mv.char !== mv.char.toUpperCase();
            const epIdx = removed.findIndex(r =>
                r.char.toLowerCase() === 'p' &&
                (r.char === r.char.toUpperCase()) === victimIsWhite &&
                fileOf(r.sq) === fileOf(mv.to));
            if (!victim && epIdx >= 0) {
                epSq = removed[epIdx].sq;
                removed.splice(epIdx, 1);
            }
            promises.push(animateMove(mv.from, mv.to, victim || null, epSq, mv.char));
        }
        for (const r of removed) {
            const entry = pieces[r.sq];
            if (entry) {
                delete pieces[r.sq];
                destroyPiece(entry, r.sq, new THREE.Vector3(0, 0, 1));
            }
        }
        for (const a of added.filter(x => !x.matched)) placePiece(a.sq, a.char);
        return Promise.all(promises);
    }

    function setFen(fen, animate) {
        const map = parseFen(fen);
        animChain = animChain.then(() => animate === false ? setInstant(map) : applyDiff(map));
        return animChain;
    }

    // -------------------------------------------------------------- input

    function highlight(sq, on) {
        const tile = squares.find(t => t.userData.square === sq);
        if (tile) {
            tile.material.emissive = new THREE.Color(on ? SELECT_COLOR : 0x000000);
            tile.material.emissiveIntensity = on ? 0.45 : 0.0;
        }
        const entry = pieces[sq];
        if (!entry) return;
        if (on) {
            // magically activated: levitate (driven per-frame in loop) with a
            // pulsing arcane glow beneath
            const { x, z } = sqToXZ(fileOf(sq), rankOf(sq));
            const glow = new THREE.Sprite(new THREE.SpriteMaterial({
                map: softDot('rgba(150,130,255,0.85)'), color: 0xa595ff,
                transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
            glow.position.set(x, BOARD_Y + 0.07, z);
            glow.scale.set(0.95, 0.5, 1);
            scene.add(glow);
            levitation = { sq, mesh: entry.mesh, glow, baseRotY: entry.mesh.rotation.y };
        } else if (levitation && levitation.sq === sq) {
            const lev = levitation;
            levitation = null;
            scene.remove(lev.glow);
            lev.mesh.rotation.y = lev.baseRotY;
            const y0 = lev.mesh.position.y;
            tween(220, k => { lev.mesh.position.y = y0 * (1 - k); }, null, easeOut);
        }
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
        const el = renderer.domElement;
        el.style.touchAction = 'none';   // browser gestures off: we handle pinch/drag
        const pointers = new Map();      // pointerId -> {x, y}; 1 finger = orbit, 2 = pinch
        let down = null, dragged = false;
        let pinchDist = 0, pinchR = camR;

        const pinchSpan = () => {
            const [a, b] = [...pointers.values()];
            return Math.hypot(a.x - b.x, a.y - b.y);
        };

        el.addEventListener('pointerdown', ev => {
            if (el.setPointerCapture) { try { el.setPointerCapture(ev.pointerId); } catch (e) {} }
            pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
            if (pointers.size === 1) {
                down = { x: ev.clientX, y: ev.clientY };
                dragged = false;
            } else if (pointers.size === 2) {
                pinchDist = pinchSpan();
                pinchR = camR;
                down = null;
                dragged = true;          // a pinch never ends in a tap
            }
        });

        el.addEventListener('pointermove', ev => {
            const p = pointers.get(ev.pointerId);
            if (p) { p.x = ev.clientX; p.y = ev.clientY; }
            if (pointers.size === 2 && pinchDist > 0) {
                const d = pinchSpan();
                if (d > 0) camR = Math.min(18, Math.max(6.5, pinchR * pinchDist / d));
                return;
            }
            if (!down) return;
            const dx = ev.clientX - down.x, dy = ev.clientY - down.y;
            if (Math.abs(dx) + Math.abs(dy) > 6) dragged = true;
            if (dragged) {
                camTheta -= dx * 0.008;
                camPhi = Math.min(1.38, Math.max(0.3, camPhi - dy * 0.005));
                down = { x: ev.clientX, y: ev.clientY };
            }
        });

        const releasePointer = ev => {
            pointers.delete(ev.pointerId);
            if (pointers.size < 2) pinchDist = 0;
            if (pointers.size === 1) {
                // one finger of a pinch lifted: re-anchor the survivor for orbit
                const [a] = [...pointers.values()];
                down = { x: a.x, y: a.y };
                dragged = true;
            }
        };

        el.addEventListener('pointerup', ev => {
            const wasDrag = dragged;
            releasePointer(ev);
            if (pointers.size > 0) return;   // another finger is still down
            down = null; dragged = false;
            if (wasDrag || !interactive) return;
            const sq = pick(ev);
            if (!sq) return;
            const mine = pieces[sq] && (pieces[sq].char === pieces[sq].char.toUpperCase()) === window.Wizard._playerWhite;
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
        el.addEventListener('pointercancel', ev => {
            releasePointer(ev);
            if (pointers.size === 0) { down = null; dragged = false; }
        });
        el.addEventListener('wheel', ev => {
            ev.preventDefault();
            camR = Math.min(18, Math.max(6.5, camR + ev.deltaY * 0.01));
        }, { passive: false });
    }

    window.Wizard = {
        init, setFen, resize,
        _playerWhite: true,
        setOrientation(color) { orientationWhite = color === 'white'; this._playerWhite = orientationWhite; },
        setInteractive(b) { interactive = b; if (!b && selected) { highlight(selected, false); selected = null; } },
    };
})();
