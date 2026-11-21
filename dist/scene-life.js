const CLOUD_COLORS = {
  sunrise: ["#fff5d8", "#f6d7d3", "#a9aec8"],
  day: ["#ffffef", "#e5eede", "#c6d7cf"],
  sunset: ["#ffd38b", "#e99884", "#986b9b"],
  night: ["#8b9db7", "#6e829d", "#566a86"],
};
const SHIRTS = ["#bb5944", "#597b7e", "#d1aa52", "#5f704d", "#675b79", "#ddd0ad"];
const SKIN = ["#c99368", "#8c5b48", "#e0ad80", "#af775e"];
const CLOUDS = [
  [.095, .20, 220, 7.3], [.36, .14, 165, 5.1], [.66, .27, 278, 6.9],
  [.17, .48, 184, 6.1], [.78, .53, 231, 7.6], [.46, .70, 156, 5.5],
  [.89, .79, 196, 6.2],
];
const CARS = ["#c66451", "#6e929e", "#e4dfc2", "#c99442", "#789079", "#8b7f9e"];
const FLOCKS = [
  ["great-meteoron", 5, 116, 75], ["roussanou", 5, 145, 125],
  ["holy-trinity", 5, 134, 86], ["st-george-mandilas", 5, 98, 115],
  ["st-stephen", 8, 155, 125], ["adrachti", 8, 135, 138],
];
const mix = (a, b, t) => a + (b-a)*t;

function routeGraph(segments) {
  const exits = new Map();
  let distance = 0;
  const edges = segments.map(([start, end, length, points, visibility, mask = 3, speed]) => {
    distance += length;
    const edge = {start, end, length, points, visibility, mask, speed, cumulative: distance};
    for (const [node, direction, bit] of [[start, 1, 1], [end, -1, 2]]) {
      if (!(mask & bit)) continue;
      if (!exits.has(node)) exits.set(node, []);
      exits.get(node).push({edge, direction});
    }
    return edge;
  });
  return {edges, exits, distance};
}

function routePosition(actor) {
  const {points, length, visibility} = actor.edge;
  const along = actor.position/length*(points.length-1);
  if (visibility[Math.round(along)] !== "1") return null;
  const i = Math.min(points.length-2, Math.floor(along));
  const [a, b] = [points[i], points[i+1]];
  return [mix(a[0],b[0],along-i), mix(a[1],b[1],along-i),
    (b[0]-a[0])*actor.direction, (b[1]-a[1])*actor.direction];
}

function carStamp(color, angle, lights) {
  const sprite = document.createElement("canvas");
  sprite.width = sprite.height = 12;
  const ctx = sprite.getContext("2d");
  ctx.translate(6, 6);
  ctx.rotate(angle);
  ctx.fillStyle = "#283034";
  ctx.fillRect(-3, -2.5, 2, 5);
  ctx.fillRect(1.5, -2.5, 2, 5);
  ctx.fillStyle = color;
  ctx.fillRect(-4, -2, 8, 4);
  ctx.fillStyle = "#475c68";
  ctx.fillRect(-1.5, -1.5, 3.25, 3);
  ctx.fillStyle = lights ? "#fff1ad" : "#e4d8ba";
  ctx.fillRect(3, -1.75, 1, 1);
  ctx.fillRect(3, .75, 1, 1);
  ctx.fillStyle = lights ? "#ff7a60" : "#944f4a";
  ctx.fillRect(-4, -1.75, 1, 1);
  ctx.fillRect(-4, .75, 1, 1);
  return sprite;
}

function cloudStamp(palette, variant) {
  const sprite = document.createElement("canvas");
  sprite.width = 56;
  sprite.height = 22;
  const ctx = sprite.getContext("2d");
  const lobes = [[14,14,13,5],[20,10,10,8],[30,11,13,9],[40,13,12,6],[48,15,7,4]];
  lobes[1][1] += variant * 1.2;
  lobes[2][0] += variant * 2.7;
  for (let y = 0; y < sprite.height; y++) {
    for (let x = 0; x < sprite.width; x++) {
      const distance = Math.min(...lobes.map(([cx,cy,rx,ry]) => ((x-cx)/rx)**2 + ((y-cy)/ry)**2));
      if (distance > 1) continue;
      const shelf = (y + Math.floor(x/6)%3) / sprite.height;
      ctx.fillStyle = palette[shelf < .51 && distance < .96 ? 0 : shelf < .73 ? 1 : 2];
      ctx.fillRect(x,y,1,1);
    }
  }
  return sprite;
}

export class SceneAnimation {
  constructor(canvas, map, view, reducedMotion) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.map = map;
    this.view = view;
    this.reducedMotion = reducedMotion;
    this.flocks = FLOCKS.flatMap(([id, count, span, rise]) => {
      const place = map.places.find(place => place.id === id);
      return place ? [{place, count, span, rise}] : [];
    });
    this.fountain = map.places.find(place => place.id === "town-hall-fountain");
    this.walkers = [];
    this.cars = [];
    this.water = [];
    this.cloudStamps = new Map();
    this.carStamps = new Map();
    this.elapsed = 0;
    this.frame = 0;
    this.last = 0;
    this.onFrame = this.tick.bind(this);
    reducedMotion.addEventListener("change", () => this.refresh());
    document.addEventListener("visibilitychange", () => this.refresh());
  }

  set(enabled, lighting) {
    this.enabled = enabled && !!this.ctx;
    this.lighting = lighting;
    this.canvas.hidden = !this.enabled;
    if (this.enabled && !this.routes) this.routes = this.loadRoutes();
    this.refresh();
  }

  async loadRoutes() {
    try {
      const response = await fetch("./assets/life.json");
      if (!response.ok) throw new Error(`Routes returned HTTP ${response.status}`);
      const {paths, roads = [], water} = await response.json();
      this.water = water;
      const {edges, exits, distance} = routeGraph(paths);
      this.walkExits = exits;
      if (edges.length) {
        const count = Math.min(126, Math.max(36, Math.round(distance/195)));
        this.walkers = Array.from({length: count}, (_, index) => {
          const edge = edges.find(e => e.cumulative >= distance*(index+.24)/count);
          return {edge, direction: index%2 ? 1 : -1, position: edge.length*(.19+index%7*.095),
            speed: 1.18+index%8*.133, rest: 0, phase: index*.19,
            shirt: SHIRTS[index%SHIRTS.length], skin: SKIN[index%SKIN.length]};
        });
      }
      const traffic = routeGraph(roads);
      this.trafficExits = traffic.exits;
      this.carStarts = traffic.edges.filter(edge => edge.length >= 7 && edge.visibility.includes("111"));
      const count = this.carStarts.length ? 20 : 0;
      this.cars = Array.from({length: count}, (_, index) => {
        const car = {color: CARS[index%CARS.length], pace: .78+Math.random()*.36};
        this.spawnCar(car, (index+.45)/count);
        return car;
      });
      this.redraw();
    } catch (error) {
      this.routes = null;
      console.warn("Walking routes unavailable.", error);
    }
  }

  refresh() {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.last = 0;
    if (!this.enabled || document.hidden) return;
    this.redraw();
    if (!this.reducedMotion.matches) this.frame = requestAnimationFrame(this.onFrame);
  }

  tick(now) {
    if (this.last && now-this.last < 30) {
      this.frame = requestAnimationFrame(this.onFrame);
      return;
    }
    const delta = this.last ? Math.min(.085, (now-this.last)/1000) : 0;
    this.last = now;
    this.elapsed += delta;
    for (const walker of this.walkers) this.walk(walker, delta);
    for (const car of this.cars) this.drive(car, delta);
    this.redraw();
    this.frame = requestAnimationFrame(this.onFrame);
  }

  walk(actor, delta) {
    if (actor.rest > 0) {
      actor.rest -= delta;
      return;
    }
    actor.phase += delta*6.5;
    actor.position += actor.speed*delta*actor.direction;
    if (actor.position >= 0 && actor.position <= actor.edge.length) return;
    const node = actor.direction > 0 ? actor.edge.end : actor.edge.start;
    const route = this.walkExits.get(node);
    const onward = route.filter(choice => choice.edge !== actor.edge);
    const choices = onward.length ? onward : route;
    const {edge, direction} = choices[Math.floor(Math.random()*choices.length)];
    Object.assign(actor, {edge, direction, position: direction > 0 ? 0 : edge.length, rest: Math.random()*.55});
  }

  spawnCar(car, start = Math.random()) {
    const edge = this.carStarts[Math.floor(start*this.carStarts.length)];
    const direction = edge.mask === 2 || (edge.mask === 3 && Math.random() < .5) ? -1 : 1;
    Object.assign(car, {edge, direction, position: edge.length*(.2+Math.random()*.6)});
  }

  drive(car, delta) {
    const {edge, direction, pace} = car;
    const position = car.position + edge.speed*pace*delta*direction;
    if (position >= 0 && position <= edge.length) {
      car.position = position;
      return;
    }
    const node = direction > 0 ? edge.end : edge.start;
    const routes = this.trafficExits.get(node) ?? [];
    const onward = routes.filter(choice => choice.edge !== edge);
    const choices = onward.length ? onward : routes;
    if (!choices.length) {
      this.spawnCar(car);
      return;
    }
    const next = choices[Math.floor(Math.random()*choices.length)];
    const distance = direction > 0 ? position-edge.length : -position;
    const entry = Math.min(distance, next.edge.length);
    Object.assign(car, next, {position: next.direction > 0 ? entry : next.edge.length-entry});
  }

  redraw() {
    if (!this.enabled || document.hidden) return;
    const {width, height, x, y, z} = this.view;
    if (!width || !height) return;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(width*ratio), h = Math.round(height*ratio);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.clearRect(0, 0, width, height);
    const left = width/2-x*z, top = height/2-y*z;
    const screen = (px, py) => [Math.round(left+px*z), Math.round(top+py*z)];
    const visible = (px, py, margin = 10) => px >= -margin && py >= -margin && px <= width+margin && py <= height+margin;
    this.drawWater(screen, visible);
    if (z >= .47) this.drawCars(screen, visible);
    if (z >= .59) this.drawWalkers(screen, visible);
    this.drawBirds(screen, visible);
    this.drawClouds(screen, visible);
  }

  drawCars(screen, visible) {
    const unit = Math.max(.64, this.view.z*.70);
    const lights = this.lighting === "sunset" || this.lighting === "night";
    for (const car of this.cars) {
      const position = routePosition(car);
      if (!position) continue;
      const [px, py, dx, dy] = position;
      const length = Math.hypot(dx, dy) || 1;
      // Traffic keeps right on both sides of the projected road centreline.
      const [x, y] = screen(px-dy/length, py+dx/length);
      if (!visible(x, y)) continue;
      const heading = (Math.round(Math.atan2(dy, dx)*8/Math.PI)+16)%16;
      const key = `${car.color}/${heading}/${lights}`;
      if (!this.carStamps.has(key)) {
        this.carStamps.set(key, carStamp(car.color, heading*Math.PI/8, lights));
      }
      const size = Math.round(12*unit);
      this.ctx.drawImage(this.carStamps.get(key), x-Math.floor(size/2), y-Math.floor(size/2), size, size);
    }
  }

  drawWater(screen, visible) {
    const {ctx, elapsed} = this;
    const z = this.view.z;
    ctx.fillStyle = this.lighting === "night" ? "#8dafc8" : "#d3f1e1";
    if (z >= .38) {
      for (let index = 0; index < this.water.length; index += 3) {
        const [x, y] = screen(...this.water[index]);
        if (!visible(x, y)) continue;
        const wave = (elapsed*.73 + index*.37)%1;
        if (wave > .61) continue;
        ctx.globalAlpha = .12 + .5*Math.sin(wave/.61*Math.PI);
        ctx.fillRect(x+Math.round(wave*8*z), y, Math.max(2, 4*z*wave), Math.max(1, z*.65));
      }
    }
    if (this.fountain && z >= 1) {
      const [x, y] = screen(this.fountain.x, this.fountain.y);
      for (let j = 0; j < 3; j++) {
        const p = (elapsed*1.7+j/3)%1;
        ctx.globalAlpha = .63*(1-p);
        ctx.fillRect(x + Math.round((j-1)*z), y - Math.round((1-p)*4*z), Math.max(1,z*.62), Math.max(1,z*.62));
      }
    }
    ctx.globalAlpha = 1;
  }

  drawWalkers(screen, visible) {
    const {ctx, walkers, lighting} = this;
    const unit = Math.max(.72, this.view.z*.55);
    const litPeople = [];
    for (const actor of walkers) {
      const position = routePosition(actor);
      if (!position) continue;
      const [x, y] = screen(position[0], position[1]);
      if (!visible(x, y)) continue;
      litPeople.push({x, y, actor});
    }
    litPeople.sort((a,b) => a.y-b.y);
    const night = lighting === "night";
    for (const {x, y, actor} of litPeople) {
      const step = actor.rest > 0 ? 0 : Math.floor(actor.phase)%4;
      const px = (dx, dy, width, height, color) => {
        ctx.fillStyle = color;
        ctx.fillRect(x+dx*unit, y+dy*unit, width*unit, height*unit);
      };
      px(-1, 0, 3, .8, "#28332c55");
      px(-1, -2, 1, step === 1 ? 1 : 2, night ? "#28374d" : "#333942");
      px(0, -2, 1, step === 3 ? 1 : 2, night ? "#28374d" : "#333942");
      px(-1.25, -4.5, 2.5, 2.6, actor.shirt);
      px(-.7, -5.95, 1.47, 1.55, actor.skin);
    }
  }

  drawBirds(screen, visible) {
    const {ctx, elapsed, flocks} = this;
    const wing = Math.min(2.15, Math.max(.7, this.view.z*.76));
    ctx.fillStyle = this.lighting === "night" ? "#8999b5" : "#39403c";
    ctx.globalAlpha = .85;
    for (let group = 0; group < flocks.length; group++) {
      const {place, count, span, rise} = flocks[group];
      for (let bird = 0; bird < count; bird++) {
        const t = elapsed*(.048+group*.004) + group*2.81 + bird*.11;
        const [x, y] = screen(place.x+Math.cos(t)*span-bird*8,
          place.y-rise+Math.sin(t)*(46+bird*5)+bird*7);
        if (!visible(x, y)) continue;
        const lift = Math.round(Math.sin(elapsed*(4.5+group*.28)+bird*.83)*wing);
        ctx.fillRect(x-wing, y-wing, wing*2, wing);
        ctx.fillRect(x-wing*2.8, y-wing+lift, wing*1.8, wing);
        ctx.fillRect(x+wing, y-wing+lift, wing*1.8, wing);
      }
    }
    ctx.globalAlpha = 1;
  }

  drawClouds(screen, visible) {
    const {ctx, elapsed, lighting, map, view} = this;
    const palette = CLOUD_COLORS[lighting] ?? CLOUD_COLORS.day;
    for (const [index, [fx, fy, width, speed]] of CLOUDS.entries()) {
      const period = map.width+680;
      const worldX = (fx*map.width+elapsed*speed+340)%period-340;
      const [x, y] = screen(worldX, fy*map.height+Math.sin(elapsed*.037+index)*7);
      const w = Math.min(234, width*view.z);
      const h = w*.39;
      if (!visible(x, y, w)) continue;
      const key = `${lighting}/${index%3}`;
      if (!this.cloudStamps.has(key)) this.cloudStamps.set(key, cloudStamp(palette, index%3));
      const sprite = this.cloudStamps.get(key);
      if (lighting !== "night") {
        ctx.globalAlpha = .075;
        ctx.filter = "brightness(0)";
        ctx.drawImage(sprite,x-w*.33,y+h*.90,w*.96,h*.67);
        ctx.filter = "none";
      }
      ctx.globalAlpha = lighting === "night" ? .47 : .68;
      ctx.drawImage(sprite,x-w/2,y-h/2,w,h);
    }
    ctx.globalAlpha = 1;
  }
}
