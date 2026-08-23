/*
 * AeroPulse-X WebGL engine digital twin
 * ------------------------------------------------------------
 * Dependency-free WebGL renderer for the offline SIH demo. The
 * model is intentionally procedural: it represents a four-cylinder
 * horizontally opposed UAV aero-piston engine, not an automotive V8.
 */
(function () {
  'use strict';

  const DEG = Math.PI / 180;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const lerp = (a, b, t) => a + (b - a) * t;

  function mat4Identity() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let value = 0;
        for (let k = 0; k < 4; k += 1) value += a[k * 4 + row] * b[column * 4 + k];
        out[column * 4 + row] = value;
      }
    }
    return out;
  }

  function mat4Translation(x, y, z) {
    const out = mat4Identity();
    out[12] = x;
    out[13] = y;
    out[14] = z;
    return out;
  }

  function mat4Scale(x, y, z) {
    const out = mat4Identity();
    out[0] = x;
    out[5] = y;
    out[10] = z;
    return out;
  }

  function mat4RotationX(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
  }

  function mat4RotationY(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
  }

  function mat4RotationZ(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  function compose(position, rotation, scale) {
    let out = mat4Translation(position[0], position[1], position[2]);
    out = mat4Multiply(out, mat4RotationZ(rotation[2]));
    out = mat4Multiply(out, mat4RotationY(rotation[1]));
    out = mat4Multiply(out, mat4RotationX(rotation[0]));
    return mat4Multiply(out, mat4Scale(scale[0], scale[1], scale[2]));
  }

  function mat4Perspective(fieldOfView, aspect, near, far) {
    const f = 1 / Math.tan(fieldOfView / 2);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[14] = 2 * far * near * nf;
    return out;
  }

  function normalize(vector) {
    const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
    return [vector[0] / length, vector[1] / length, vector[2] / length];
  }

  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }

  function mat4LookAt(eye, target, up) {
    const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    const x = normalize(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
      -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
      -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
      1
    ]);
  }

  function transformPoint(matrix, point) {
    const x = point[0];
    const y = point[1];
    const z = point[2];
    const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    return [
      (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w,
      (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w,
      (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) / w
    ];
  }

  function hexColor(hex) {
    const value = Number.parseInt(hex.replace('#', ''), 16);
    return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
  }

  function mixColor(a, b, t) {
    return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  }

  function thermalColor(value, low, high) {
    const t = clamp((value - low) / (high - low), 0, 1);
    if (t < 0.34) return mixColor(hexColor('#28d7f2'), hexColor('#51e58a'), t / 0.34);
    if (t < 0.7) return mixColor(hexColor('#51e58a'), hexColor('#ffc536'), (t - 0.34) / 0.36);
    return mixColor(hexColor('#ffc536'), hexColor('#ff4d5b'), (t - 0.7) / 0.3);
  }

  function makeCube() {
    const positions = [];
    const normals = [];
    const indices = [];
    const faces = [
      [[1, 0, 0], [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]],
      [[-1, 0, 0], [[-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]]],
      [[0, 1, 0], [[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]],
      [[0, -1, 0], [[-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5]]],
      [[0, 0, 1], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]],
      [[0, 0, -1], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]]
    ];
    faces.forEach(([normal, vertices]) => {
      const offset = positions.length / 3;
      vertices.forEach(vertex => {
        positions.push(...vertex);
        normals.push(...normal);
      });
      indices.push(offset, offset + 1, offset + 2, offset, offset + 2, offset + 3);
    });
    return { positions, normals, indices };
  }

  function makeCylinder(segments) {
    const positions = [];
    const normals = [];
    const indices = [];
    for (let index = 0; index <= segments; index += 1) {
      const angle = index / segments * Math.PI * 2;
      const x = Math.cos(angle) * 0.5;
      const y = Math.sin(angle) * 0.5;
      positions.push(x, y, -0.5, x, y, 0.5);
      normals.push(Math.cos(angle), Math.sin(angle), 0, Math.cos(angle), Math.sin(angle), 0);
    }
    for (let index = 0; index < segments; index += 1) {
      const a = index * 2;
      indices.push(a, a + 1, a + 3, a, a + 3, a + 2);
    }
    [-0.5, 0.5].forEach((z, capIndex) => {
      const center = positions.length / 3;
      positions.push(0, 0, z);
      normals.push(0, 0, capIndex ? 1 : -1);
      const ring = positions.length / 3;
      for (let index = 0; index <= segments; index += 1) {
        const angle = index / segments * Math.PI * 2;
        positions.push(Math.cos(angle) * 0.5, Math.sin(angle) * 0.5, z);
        normals.push(0, 0, capIndex ? 1 : -1);
      }
      for (let index = 0; index < segments; index += 1) {
        if (capIndex) indices.push(center, ring + index, ring + index + 1);
        else indices.push(center, ring + index + 1, ring + index);
      }
    });
    return { positions, normals, indices };
  }

  function makeSphere(latitudeBands, longitudeBands) {
    const positions = [];
    const normals = [];
    const indices = [];
    for (let latitude = 0; latitude <= latitudeBands; latitude += 1) {
      const theta = latitude * Math.PI / latitudeBands;
      for (let longitude = 0; longitude <= longitudeBands; longitude += 1) {
        const phi = longitude * Math.PI * 2 / longitudeBands;
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.cos(theta);
        const z = Math.sin(theta) * Math.sin(phi);
        positions.push(x * 0.5, y * 0.5, z * 0.5);
        normals.push(x, y, z);
      }
    }
    for (let latitude = 0; latitude < latitudeBands; latitude += 1) {
      for (let longitude = 0; longitude < longitudeBands; longitude += 1) {
        const first = latitude * (longitudeBands + 1) + longitude;
        const second = first + longitudeBands + 1;
        indices.push(first, second, first + 1, second, second + 1, first + 1);
      }
    }
    return { positions, normals, indices };
  }

  function makeTorus(radialSegments, tubularSegments) {
    const positions = [];
    const normals = [];
    const indices = [];
    const major = 0.34;
    const minor = 0.16;
    for (let radial = 0; radial <= radialSegments; radial += 1) {
      const u = radial / radialSegments * Math.PI * 2;
      for (let tubular = 0; tubular <= tubularSegments; tubular += 1) {
        const v = tubular / tubularSegments * Math.PI * 2;
        const x = (major + minor * Math.cos(v)) * Math.cos(u);
        const y = (major + minor * Math.cos(v)) * Math.sin(u);
        const z = minor * Math.sin(v);
        positions.push(x, y, z);
        normals.push(Math.cos(v) * Math.cos(u), Math.cos(v) * Math.sin(u), Math.sin(v));
      }
    }
    for (let radial = 0; radial < radialSegments; radial += 1) {
      for (let tubular = 0; tubular < tubularSegments; tubular += 1) {
        const a = radial * (tubularSegments + 1) + tubular;
        const b = (radial + 1) * (tubularSegments + 1) + tubular;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    return { positions, normals, indices };
  }

  function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  function createProgram(gl) {
    const vertex = createShader(gl, gl.VERTEX_SHADER, `
      attribute vec3 aPosition;
      attribute vec3 aNormal;
      uniform mat4 uModel;
      uniform mat4 uView;
      uniform mat4 uProjection;
      varying vec3 vNormal;
      varying vec3 vWorld;
      void main(){
        vec4 world = uModel * vec4(aPosition, 1.0);
        vWorld = world.xyz;
        vNormal = normalize(mat3(uModel) * aNormal);
        gl_Position = uProjection * uView * world;
      }
    `);
    const fragment = createShader(gl, gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform vec3 uColor;
      uniform float uAlpha;
      uniform float uGlow;
      uniform float uSelected;
      varying vec3 vNormal;
      varying vec3 vWorld;
      void main(){
        vec3 light = normalize(vec3(0.35, 0.85, 0.65));
        float diffuse = max(dot(normalize(vNormal), light), 0.0);
        float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vec3(0.25, 0.2, 1.0)))), 2.0);
        vec3 color = uColor * (0.28 + 0.72 * diffuse);
        color += uColor * uGlow * 0.65 + vec3(0.08, 0.7, 0.95) * rim * (0.16 + uSelected * 0.85);
        gl_FragColor = vec4(color, uAlpha);
      }
    `);
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  function uploadMesh(gl, source) {
    const position = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, position);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(source.positions), gl.STATIC_DRAW);
    const normal = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normal);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(source.normals), gl.STATIC_DRAW);
    const index = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(source.indices), gl.STATIC_DRAW);
    return { position, normal, index, count: source.indices.length };
  }

  class AeroEngineRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext('webgl', { antialias: true, alpha: true });
      if (!this.gl) throw new Error('WebGL is unavailable');
      this.program = createProgram(this.gl);
      this.locations = {
        position: this.gl.getAttribLocation(this.program, 'aPosition'),
        normal: this.gl.getAttribLocation(this.program, 'aNormal'),
        model: this.gl.getUniformLocation(this.program, 'uModel'),
        view: this.gl.getUniformLocation(this.program, 'uView'),
        projection: this.gl.getUniformLocation(this.program, 'uProjection'),
        color: this.gl.getUniformLocation(this.program, 'uColor'),
        alpha: this.gl.getUniformLocation(this.program, 'uAlpha'),
        glow: this.gl.getUniformLocation(this.program, 'uGlow'),
        selected: this.gl.getUniformLocation(this.program, 'uSelected')
      };
      this.meshes = {
        cube: uploadMesh(this.gl, makeCube()),
        cylinder: uploadMesh(this.gl, makeCylinder(24)),
        sphere: uploadMesh(this.gl, makeSphere(12, 18)),
        torus: uploadMesh(this.gl, makeTorus(24, 12))
      };
      this.telemetry = {
        rpm: 3000,
        throttle: 60,
        cht: 220,
        egt: 1200,
        oilPressure: 60,
        oilTemp: 90,
        fuelFlow: 20,
        vibration: 1.02,
        busVoltage: 28.2,
        health: 98,
        fault: 'none'
      };
      this.mode = 'normal';
      this.paused = false;
      this.xray = false;
      this.exploded = false;
      this.explodeAmount = 0;
      this.crankAngle = 0;
      this.camera = { yaw: -38 * DEG, pitch: 19 * DEG, distance: 12.5 };
      this.selected = 'crankcase';
      this.drag = null;
      this.pickTargets = [];
      this.lastTime = performance.now();
      this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.configureGl();
      this.bindInteractions();
      this.bindControls();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas.parentElement);
      this.resize();
      this.updateInspector();
      this.updateThermalField();
      requestAnimationFrame(time => this.frame(time));
    }

    configureGl() {
      const gl = this.gl;
      gl.useProgram(this.program);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0.018, 0.045, 0.075, 1);
    }

    resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(320, Math.floor(this.canvas.clientWidth * ratio));
      const height = Math.max(300, Math.floor(this.canvas.clientHeight * ratio));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl.viewport(0, 0, width, height);
      }
    }

    resetCamera() {
      this.camera = { yaw: -38 * DEG, pitch: 19 * DEG, distance: 12.5 };
    }

    setMode(mode) {
      this.mode = ['normal', 'thermal', 'vibration'].includes(mode) ? mode : 'normal';
      document.querySelectorAll('[data-engine-mode]').forEach(button => {
        button.classList.toggle('active', button.dataset.engineMode === this.mode);
      });
      const modeLabel = document.getElementById('engineModeLabel');
      if (modeLabel) modeLabel.textContent = this.mode.toUpperCase();
      this.updateThermalField();
    }

    setTelemetry(data) {
      const numeric = ['rpm', 'throttle', 'cht', 'egt', 'oilPressure', 'oilTemp', 'fuelFlow', 'vibration', 'busVoltage', 'health'];
      numeric.forEach(key => {
        const value = Number(data[key]);
        if (Number.isFinite(value)) this.telemetry[key] = value;
      });
      if (data.fault != null) this.telemetry.fault = String(data.fault);
      this.updateHud();
      this.updateInspector();
      this.updateThermalField();
    }

    updateThermalField() {
      const t = this.telemetry;
      const chtLevel = clamp((t.cht - 180) / (315 - 180), 0, 1);
      const egtLevel = clamp((t.egt - 850) / (1500 - 850), 0, 1);
      const level = Math.max(chtLevel, egtLevel * 0.94);
      const color = thermalColor(level, 0, 1).map(channel => Math.round(channel * 255));
      const viewport = this.canvas.closest('.engine-viewport');
      if (viewport) {
        viewport.classList.toggle('thermal-active', this.mode === 'thermal');
        viewport.style.setProperty('--thermal-rgb', color.join(', '));
        viewport.style.setProperty('--thermal-level', level.toFixed(3));
        viewport.style.setProperty('--thermal-outer-opacity', (0.18 + level * 0.26).toFixed(3));
        viewport.style.setProperty('--thermal-core-opacity', (0.16 + level * 0.3).toFixed(3));
        viewport.style.setProperty('--thermal-red-alpha', (level * 0.11).toFixed(3));
        viewport.style.setProperty('--thermal-amber-alpha', (level * 0.42).toFixed(3));
        viewport.style.setProperty('--thermal-core-red-alpha', (level * 0.34).toFixed(3));
        viewport.style.setProperty('--thermal-blur', `${Math.round(8 + level * 8)}px`);
        viewport.style.setProperty('--thermal-pulse-scale', (1.015 + level * 0.025).toFixed(3));
      }
      const value = document.getElementById('engineThermalValue');
      if (value) value.textContent = `${Math.round(t.cht)}°F CHT · ${Math.round(t.egt).toLocaleString()}°F EGT`;
      const marker = document.getElementById('engineThermalMarker');
      if (marker) marker.style.left = `${Math.round(level * 100)}%`;
      const field = document.getElementById('engineThermalField');
      if (field) {
        const state = level >= 0.86 ? 'critical' : level >= 0.7 ? 'hot' : level >= 0.34 ? 'nominal' : 'cool';
        field.dataset.thermalState = state;
      }
    }

    updateHud() {
      const values = {
        engineHudRpm: `${Math.round(this.telemetry.rpm).toLocaleString()} RPM`,
        engineHudCht: `${Math.round(this.telemetry.cht)}°F CHT`,
        engineHudEgt: `${Math.round(this.telemetry.egt)}°F EGT`,
        engineHudOil: `${this.telemetry.oilPressure.toFixed(0)} PSI OIL`,
        engineHudVibration: `${this.telemetry.vibration.toFixed(2)} g RMS`
      };
      Object.entries(values).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
      });
      const state = document.getElementById('engineTwinState');
      if (state) {
        const fault = this.telemetry.fault.toLowerCase();
        state.textContent = fault && fault !== 'none' ? `FAULT FOCUS • ${this.telemetry.fault}` : 'DIGITAL TWIN SYNCHRONIZED';
        state.className = fault && fault !== 'none' ? 'engine-twin-state warn' : 'engine-twin-state';
      }
    }

    componentData(component) {
      const t = this.telemetry;
      const fault = t.fault.toLowerCase();
      const entries = {
        crankcase: ['Central crankcase', `${Math.round(t.rpm)} RPM`, `${t.vibration.toFixed(2)} g`, 'Crankshaft and opposed-cylinder power core'],
        cylinders: ['Opposed cylinder bank', `${Math.round(t.cht)}°F CHT`, `${Math.round(t.egt)}°F EGT`, 'Four-cylinder combustion and thermal state'],
        crankshaft: ['Crankshaft assembly', `${Math.round(t.rpm)} RPM`, `${t.vibration.toFixed(2)} g`, 'Animated power transmission and bearing state'],
        propeller: ['Reduction output shaft', `${Math.round(t.rpm * 0.46)} RPM`, `${Math.round(t.throttle)}% load`, 'Propeller reduction and mission power delivery'],
        turbo: ['Turbocharger', `${Math.max(0.6, 0.55 + t.throttle / 100).toFixed(2)} bar`, `${Math.round(t.egt)}°F`, 'Boost and exhaust energy recovery'],
        lubrication: ['Lubrication circuit', `${t.oilPressure.toFixed(1)} PSI`, `${t.oilTemp.toFixed(1)}°C`, 'Oil pressure, temperature and flow condition'],
        fuel: ['Fuel injection rail', `${t.fuelFlow.toFixed(1)} L/h`, `${Math.round(t.throttle)}% command`, 'FADEC-controlled injection pulse state'],
        electrical: ['Alternator / FADEC', `${t.busVoltage.toFixed(1)} V`, `${Math.round(t.health)}% health`, 'Electrical generation and engine control'],
        sensors: ['Sensor network', `${Math.round(t.health)}% trust`, fault.includes('sensor') ? 'DRIFT' : 'TRUSTED', 'Virtual ECU/FADEC sensor nodes']
      };
      return entries[component] || entries.crankcase;
    }

    updateInspector() {
      const data = this.componentData(this.selected);
      const fault = this.telemetry.fault;
      const ids = {
        selectedComponentName: data[0],
        selectedComponentPrimary: data[1],
        selectedComponentSecondary: data[2],
        selectedComponentDetail: data[3],
        selectedComponentFault: fault && fault.toLowerCase() !== 'none' ? fault : 'No active fault evidence'
      };
      Object.entries(ids).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
      });
    }

    bindControls() {
      const on = (id, event, handler) => {
        const element = document.getElementById(id);
        if (element) element.addEventListener(event, handler);
      };
      on('engineResetCamera', 'click', () => this.resetCamera());
      on('enginePause', 'click', event => {
        this.paused = !this.paused;
        event.currentTarget.classList.toggle('active', this.paused);
        event.currentTarget.textContent = this.paused ? 'Resume motion' : 'Pause motion';
      });
      on('engineXray', 'click', event => {
        this.xray = !this.xray;
        event.currentTarget.classList.toggle('active', this.xray);
      });
      on('engineExplode', 'click', event => {
        this.exploded = !this.exploded;
        event.currentTarget.classList.toggle('active', this.exploded);
      });
      document.querySelectorAll('[data-engine-mode]').forEach(button => {
        button.addEventListener('click', () => this.setMode(button.dataset.engineMode));
      });
    }

    bindInteractions() {
      this.canvas.addEventListener('pointerdown', event => {
        this.canvas.setPointerCapture(event.pointerId);
        this.drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY };
      });
      this.canvas.addEventListener('pointermove', event => {
        if (!this.drag) return;
        const dx = event.clientX - this.drag.x;
        const dy = event.clientY - this.drag.y;
        this.camera.yaw -= dx * 0.007;
        this.camera.pitch = clamp(this.camera.pitch - dy * 0.006, -1.15, 1.15);
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
      });
      this.canvas.addEventListener('pointerup', event => {
        if (!this.drag) return;
        const moved = Math.hypot(event.clientX - this.drag.startX, event.clientY - this.drag.startY);
        if (moved < 6) this.pick(event);
        this.drag = null;
      });
      this.canvas.addEventListener('wheel', event => {
        event.preventDefault();
        this.camera.distance = clamp(this.camera.distance + event.deltaY * 0.012, 7, 20);
      }, { passive: false });
      this.canvas.addEventListener('dblclick', () => this.resetCamera());
    }

    pick(event) {
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let winner = null;
      this.pickTargets.forEach(target => {
        const distance = Math.hypot(target.x - x, target.y - y);
        if (distance < target.radius && (!winner || distance < winner.distance)) winner = { ...target, distance };
      });
      if (winner) {
        this.selected = winner.component;
        this.updateInspector();
      }
    }

    part(mesh, component, label, position, rotation, scale, color, options = {}) {
      return { mesh, component, label, position, rotation, scale, color, alpha: options.alpha ?? 1, glow: options.glow ?? 0, pick: options.pick ?? false };
    }

    buildParts(time) {
      const t = this.telemetry;
      const fault = t.fault.toLowerCase();
      const thermal = this.mode === 'thermal';
      const vibrationMode = this.mode === 'vibration';
      const explosion = this.explodeAmount;
      const metal = hexColor('#3f536b');
      const darkMetal = hexColor('#1d2b3b');
      const aluminium = hexColor('#73859a');
      const cyan = hexColor('#29d3f2');
      const green = hexColor('#51e58a');
      const amber = hexColor('#ffc536');
      const red = hexColor('#ff4d5b');
      const oilColor = fault.includes('lubric') ? red : green;
      const fuelColor = fault.includes('inject') ? amber : cyan;
      const electricColor = fault.includes('elect') || fault.includes('battery') || t.busVoltage < 24 ? red : cyan;
      const cylinderColor = thermal ? thermalColor(t.cht, 180, 315) : aluminium;
      const exhaustColor = thermal ? thermalColor(t.egt, 850, 1500) : hexColor('#7b432c');
      const thermalIntensity = Math.max(
        clamp((t.cht - 180) / (315 - 180), 0, 1),
        clamp((t.egt - 850) / (1500 - 850), 0, 1) * 0.94
      );
      const housingAlpha = this.xray ? 0.22 : 1;
      const parts = [];
      const add = (...args) => parts.push(this.part(...args));

      add('cube', 'crankcase', 'Central crankcase', [0, 0, 0], [0, 0, 0], [4.8, 1.3, 1.55], darkMetal, { alpha: housingAlpha, pick: true });
      add('cube', 'crankcase', 'Upper accessory housing', [0.25, 1.02 + explosion * 0.45, 0], [0, 0, 0], [3.1, 0.62, 1.26], metal, { alpha: housingAlpha });
      add('cube', 'lubrication', 'Oil sump', [0, -0.92 - explosion * 0.35, 0], [0, 0, 0], [3.7, 0.42, 1.25], oilColor, { alpha: this.xray ? 0.5 : 0.9, glow: thermal ? 0.15 : 0.03, pick: true });

      add('cylinder', 'crankshaft', 'Crankshaft', [0, 0.03, 0], [0, Math.PI / 2, 0], [0.38, 0.38, 5.5], aluminium, { glow: vibrationMode ? clamp(t.vibration / 4, 0, 0.8) : 0.03, pick: true });
      [-1.4, 0, 1.4].forEach((x, index) => {
        const phase = this.crankAngle + index * Math.PI * 0.66;
        add('cylinder', 'crankshaft', 'Crank throw', [x, Math.sin(phase) * 0.25, Math.cos(phase) * 0.25], [0, Math.PI / 2, 0], [0.55, 0.55, 0.26], hexColor('#95a4b7'), { glow: 0.05 });
      });

      const cylinderXs = [-1.32, 1.32];
      const phases = [0, Math.PI, Math.PI, 0];
      let cylinderIndex = 0;
      cylinderXs.forEach(x => {
        [-1, 1].forEach(side => {
          const phase = this.crankAngle + phases[cylinderIndex];
          const misfire = fault.includes('misfire') && cylinderIndex === 2;
          const stroke = misfire ? Math.sin(phase * 0.42) * 0.1 : Math.sin(phase) * 0.34;
          const bankOffset = side * explosion * 1.45;
          const barrelZ = side * 1.73 + bankOffset;
          const headZ = side * 2.62 + side * explosion * 2.05;
          const pistonZ = side * (0.84 + stroke) + bankOffset * 0.28;
          const hotFault = (fault.includes('overheat') || fault.includes('thermal')) && cylinderIndex === 1;
          const color = hotFault ? red : cylinderColor;
          add('cylinder', 'cylinders', `Cylinder ${cylinderIndex + 1}`, [x, 0.3, barrelZ], [0, 0, 0], [1.2, 1.2, 1.72], color, { alpha: housingAlpha, glow: hotFault ? 0.95 : thermal ? 0.32 : 0.03, pick: true });
          add('cylinder', 'cylinders', `Cylinder head ${cylinderIndex + 1}`, [x, 0.3, headZ], [0, 0, 0], [1.48, 1.48, 0.55], color, { glow: hotFault ? 1 : thermal ? 0.4 : 0.03 });
          if (thermal) {
            add('sphere', 'cylinders', `Cylinder ${cylinderIndex + 1} thermal envelope`, [x, 0.3, headZ], [0, 0, 0], [1.65, 1.65, 1.05], color, {
              alpha: 0.055 + thermalIntensity * 0.075,
              glow: 0.72 + thermalIntensity * 0.28
            });
          }
          add('cylinder', 'cylinders', `Piston ${cylinderIndex + 1}`, [x, 0.3, pistonZ], [0, 0, 0], [0.78, 0.78, 0.42], hexColor('#c1ccd8'), { glow: misfire ? 0.7 : 0.04 });
          const crankZ = Math.cos(phase) * 0.26;
          const middleZ = (pistonZ + crankZ) / 2;
          const deltaZ = pistonZ - crankZ;
          const length = Math.hypot(deltaZ, 0.3);
          add('cube', 'crankshaft', `Connecting rod ${cylinderIndex + 1}`, [x, 0.16, middleZ], [Math.atan2(deltaZ, 0.3), 0, 0], [0.18, length, 0.18], hexColor('#b3c0cd'), { glow: misfire ? 0.45 : 0 });
          cylinderIndex += 1;
        });
      });

      const propX = -3.1 - explosion * 1.25;
      add('cylinder', 'propeller', 'Reduction output shaft', [propX, 0, 0], [0, Math.PI / 2, 0], [0.52, 0.52, 2.2], aluminium, { pick: true });
      add('sphere', 'propeller', 'Propeller spinner', [propX - 1.02, 0, 0], [0, 0, 0], [0.78, 0.78, 0.78], metal, { glow: 0.06 });
      const propAngle = this.crankAngle * 0.46;
      add('cube', 'propeller', 'Propeller blade', [propX - 1.08, 0, 0], [propAngle, 0, 0], [0.16, 4.4, 0.28], hexColor('#456278'), { glow: 0.04 });
      add('cube', 'propeller', 'Propeller blade', [propX - 1.08, 0, 0], [propAngle + Math.PI / 2, 0, 0], [0.16, 4.4, 0.28], hexColor('#456278'), { glow: 0.04 });

      const turboPosition = [2.8 + explosion * 1.35, 1.34 + explosion * 0.75, -0.45];
      add('torus', 'turbo', 'Turbocharger housing', turboPosition, [0, Math.PI / 2, 0], [1.55, 1.55, 1.55], thermal ? exhaustColor : metal, { glow: thermal ? 0.5 : 0.08, pick: true });
      add('cylinder', 'turbo', 'Turbo turbine', turboPosition, [0, Math.PI / 2, this.crankAngle * 1.8], [0.7, 0.7, 0.42], aluminium, { glow: thermal ? 0.55 : 0.08 });
      if (thermal) {
        add('sphere', 'turbo', 'Turbo thermal envelope', turboPosition, [0, 0, 0], [1.45, 1.45, 1.45], exhaustColor, {
          alpha: 0.06 + thermalIntensity * 0.08,
          glow: 0.78 + thermalIntensity * 0.22
        });
      }
      add('cylinder', 'turbo', 'Intake manifold', [0.4, 1.63 + explosion * 0.5, 0], [0, Math.PI / 2, 0], [0.32, 0.32, 4.7], cyan, { alpha: 0.78, glow: 0.26 });

      [-1, 1].forEach(side => {
        add('cylinder', 'fuel', 'Fuel rail', [0, 1.05 + explosion * 0.4, side * (1.12 + explosion * 1.5)], [0, Math.PI / 2, 0], [0.15, 0.15, 3.9], fuelColor, { glow: 0.65, pick: true });
        add('cylinder', 'turbo', 'Exhaust collector', [0.2, -0.52, side * (2.35 + explosion * 1.7)], [0, Math.PI / 2, 0], [0.32, 0.32, 4.1], exhaustColor, { glow: thermal ? 0.75 : 0.12 });
        add('cylinder', 'lubrication', 'Oil gallery', [0, -0.64 - explosion * 0.2, side * 0.83], [0, Math.PI / 2, 0], [0.13, 0.13, 4.0], oilColor, { glow: 0.7 });
      });

      const alternatorPosition = [2.1 + explosion * 0.9, -1.32 - explosion * 0.6, 0];
      add('cylinder', 'electrical', 'Alternator', alternatorPosition, [0, Math.PI / 2, 0], [1.15, 1.15, 1.3], electricColor, { glow: electricColor === red ? 0.9 : 0.16, pick: true });
      add('cylinder', 'electrical', 'Alternator rotor', alternatorPosition, [0, Math.PI / 2, this.crankAngle], [0.58, 0.58, 1.45], aluminium, { glow: 0.08 });

      const sensorPositions = [[-1.32, 1.02, -2.72], [-1.32, 1.02, 2.72], [1.32, 1.02, -2.72], [1.32, 1.02, 2.72], [0, -1.24, 0]];
      sensorPositions.forEach((position, index) => {
        const sensorFault = fault.includes('sensor') && index === 2;
        const pulse = 0.16 + Math.sin(time * 0.006 + index) * 0.04;
        add('sphere', 'sensors', `Sensor node ${index + 1}`, [position[0], position[1], position[2] + Math.sign(position[2]) * explosion * 2], [0, 0, 0], [pulse, pulse, pulse], sensorFault ? red : green, { glow: sensorFault ? 1 : 0.72, pick: true });
      });

      const flowSpeed = Math.max(0.2, t.oilPressure / 60);
      for (let index = 0; index < 7; index += 1) {
        const x = ((time * 0.0006 * flowSpeed + index / 7) % 1) * 3.5 - 1.75;
        add('sphere', 'lubrication', 'Oil flow particle', [x, -0.65, -0.82], [0, 0, 0], [0.11, 0.11, 0.11], oilColor, { glow: 0.95 });
      }
      return parts;
    }

    drawPart(part) {
      const gl = this.gl;
      const mesh = this.meshes[part.mesh];
      const selected = this.selected === part.component ? 1 : 0;
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.position);
      gl.enableVertexAttribArray(this.locations.position);
      gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.normal);
      gl.enableVertexAttribArray(this.locations.normal);
      gl.vertexAttribPointer(this.locations.normal, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.index);
      gl.uniformMatrix4fv(this.locations.model, false, compose(part.position, part.rotation, part.scale));
      gl.uniform3fv(this.locations.color, part.color);
      gl.uniform1f(this.locations.alpha, part.alpha);
      gl.uniform1f(this.locations.glow, part.glow);
      gl.uniform1f(this.locations.selected, selected);
      if (part.alpha < 0.95) gl.disable(gl.CULL_FACE);
      else gl.enable(gl.CULL_FACE);
      gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
    }

    buildPickTargets(parts, viewProjection) {
      const rect = this.canvas.getBoundingClientRect();
      const seen = new Set();
      this.pickTargets = [];
      parts.filter(part => part.pick).forEach(part => {
        if (seen.has(part.component)) return;
        seen.add(part.component);
        const point = transformPoint(viewProjection, part.position);
        if (point[2] < -1 || point[2] > 1) return;
        this.pickTargets.push({
          component: part.component,
          x: (point[0] * 0.5 + 0.5) * rect.width,
          y: (1 - (point[1] * 0.5 + 0.5)) * rect.height,
          radius: 48
        });
      });
    }

    drawGrid() {
      const gridColor = hexColor('#16354a');
      for (let index = -6; index <= 6; index += 1) {
        this.drawPart(this.part('cube', 'grid', 'Grid', [index, -2.25, 0], [0, 0, 0], [0.018, 0.018, 12], gridColor, { alpha: 0.25 }));
        this.drawPart(this.part('cube', 'grid', 'Grid', [0, -2.25, index], [0, 0, 0], [12, 0.018, 0.018], gridColor, { alpha: 0.25 }));
      }
    }

    frame(time) {
      const delta = Math.min(0.05, (time - this.lastTime) / 1000);
      this.lastTime = time;
      this.resize();
      if (!this.paused && !this.reducedMotion) {
        const revolutionsPerSecond = clamp(this.telemetry.rpm / 60, 5, 95);
        this.crankAngle = (this.crankAngle + delta * revolutionsPerSecond * Math.PI * 0.32) % (Math.PI * 2);
      }
      this.explodeAmount += ((this.exploded ? 1 : 0) - this.explodeAmount) * Math.min(1, delta * 5);

      const gl = this.gl;
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const aspect = this.canvas.width / this.canvas.height;
      const projection = mat4Perspective(42 * DEG, aspect, 0.1, 100);
      const camera = this.camera;
      const horizontal = Math.cos(camera.pitch) * camera.distance;
      const eye = [
        Math.sin(camera.yaw) * horizontal,
        Math.sin(camera.pitch) * camera.distance + 0.35,
        Math.cos(camera.yaw) * horizontal
      ];
      const view = mat4LookAt(eye, [0, 0, 0], [0, 1, 0]);
      gl.useProgram(this.program);
      gl.uniformMatrix4fv(this.locations.view, false, view);
      gl.uniformMatrix4fv(this.locations.projection, false, projection);
      this.drawGrid();

      const vibration = this.mode === 'vibration' ? clamp((this.telemetry.vibration - 0.7) * 0.018, 0, 0.09) : 0;
      const fault = this.telemetry.fault.toLowerCase();
      const faultShake = fault.includes('misfire') ? 0.045 : 0;
      const shift = [Math.sin(time * 0.052) * (vibration + faultShake), Math.cos(time * 0.041) * vibration, 0];
      const parts = this.buildParts(time);
      parts.forEach(part => {
        part.position = [part.position[0] + shift[0], part.position[1] + shift[1], part.position[2] + shift[2]];
      });
      const opaque = parts.filter(part => part.alpha >= 0.95);
      const translucent = parts.filter(part => part.alpha < 0.95);
      opaque.forEach(part => this.drawPart(part));
      translucent.forEach(part => this.drawPart(part));
      this.buildPickTargets(parts, mat4Multiply(projection, view));
      requestAnimationFrame(next => this.frame(next));
    }
  }

  function showFallback(error) {
    const shell = document.querySelector('.engine-viewport');
    if (!shell) return;
    shell.innerHTML = `<div class="engine-fallback"><strong>3D engine fallback active</strong><span>${error.message}</span><span>Telemetry and AI analysis remain available.</span></div>`;
  }

  function init() {
    const canvas = document.getElementById('engineCanvas');
    if (!canvas) return;
    try {
      const renderer = new AeroEngineRenderer(canvas);
      window.AeroPulseEngine3D = {
        setTelemetry: data => renderer.setTelemetry(data || {}),
        setMode: mode => renderer.setMode(mode),
        resetCamera: () => renderer.resetCamera(),
        renderer
      };
      renderer.updateHud();
    } catch (error) {
      console.error('AeroPulse WebGL initialization failed:', error);
      showFallback(error);
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
