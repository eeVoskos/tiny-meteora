"""Render an orthographic pixel-art atlas from bundled OSM and elevation data."""

from collections import defaultdict
from pathlib import Path
import gzip
import json
import math
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "dist" / "assets"
WIDTH, HEIGHT = 1536, 1040
GROUND_ROWS, GROUND_CENTER = 1900, 950
CENTER_LAT, CENTER_LON = 39.714, 21.6225
METERS_PER_LAT = 111_195
METERS_PER_LON = METERS_PER_LAT * math.cos(math.radians(CENTER_LAT))
METERS_PER_PIXEL = 3.0
ELEVATION_DATUM = 240
VERTICAL_SCALE, GROUND_SCALE, SCREEN_CENTER = 1.16, 0.655, 548
ANGLE = math.radians(22)
COS, SIN = math.cos(ANGLE), math.sin(ANGLE)
RNG = random.Random(577)
NOISE = np.random.default_rng(577)

LAND = {
    "farmland": 1, "orchard": 1, "vineyard": 1, "grass": 2, "grassland": 2,
    "residential": 3, "industrial": 3, "cemetery": 3, "religious": 3,
    "wood": 4, "forest": 4, "scrub": 5, "bare_rock": 6, "shingle": 7,
    "water": 8,
}
ROADS = {
    "motorway": 8, "trunk": 7, "primary": 5, "secondary": 4,
    "tertiary": 4, "residential": 3, "unclassified": 3,
    "living_street": 3, "pedestrian": 2, "service": 2,
}
PATHS = {"path", "footway", "steps", "track"}


def ground(lat, lon):
    east = (lon - CENTER_LON) * METERS_PER_LON
    north = (lat - CENTER_LAT) * METERS_PER_LAT
    return (WIDTH / 2 + (east * COS + north * SIN) / METERS_PER_PIXEL,
            GROUND_CENTER + (east * SIN - north * COS) / METERS_PER_PIXEL)


def screen_y(row, height):
    return SCREEN_CENTER + (row - GROUND_CENTER) * GROUND_SCALE - (
        height - ELEVATION_DATUM) * VERTICAL_SCALE / METERS_PER_PIXEL


def elevations():
    tiles = {}
    for file in sorted((DATA / "terrain").glob("*.png")):
        x, y = map(int, file.stem.split("-"))
        rgb = np.asarray(Image.open(file).convert("RGB"), dtype=np.float32)
        tiles[x, y] = rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256 - 32768
    x0, y0 = (min(p[i] for p in tiles) for i in (0, 1))
    x1, y1 = (max(p[i] for p in tiles) for i in (0, 1))
    grid = np.empty(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), dtype=np.float32)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            row, col = (y - y0) * 256, (x - x0) * 256
            grid[row:row + 256, col:col + 256] = tiles[x, y]

    u = (np.arange(WIDTH, dtype=np.float32)[None, :] - WIDTH / 2) * METERS_PER_PIXEL
    v = (np.arange(GROUND_ROWS, dtype=np.float32)[:, None] - GROUND_CENTER) * METERS_PER_PIXEL
    lon = CENTER_LON + (u * COS + v * SIN) / METERS_PER_LON
    lat = CENTER_LAT + (u * SIN - v * COS) / METERS_PER_LAT
    mercator = np.arcsinh(np.tan(np.radians(lat.astype(np.float64))))
    sx = (lon.astype(np.float64) + 180) / 360 * 2**21 - x0 * 256 - 0.5
    sy = (1 - mercator / np.pi) / 2 * 2**21 - y0 * 256 - 0.5
    if sx.min() < 0 or sx.max() >= grid.shape[1] - 1 or sy.min() < 0 or sy.max() >= grid.shape[0] - 1:
        raise ValueError("The render extends beyond the bundled elevation tiles")
    ix, iy = sx.astype(int), sy.astype(int)
    fx, fy = sx - ix, sy - iy
    return ((1 - fy) * ((1 - fx) * grid[iy, ix] + fx * grid[iy, ix + 1])
            + fy * ((1 - fx) * grid[iy + 1, ix] + fx * grid[iy + 1, ix + 1])).astype(np.float32)


def read_features():
    with gzip.open(DATA / "kalampaka.osm.gz", "rb") as source:
        root = ET.parse(source).getroot()
    tags = lambda element: {t.get("k"): t.get("v") for t in element.findall("tag")}
    nodes = {node.get("id"): ground(float(node.get("lat")), float(node.get("lon")))
             for node in root.findall("node")}
    ways = {}
    for way in root.findall("way"):
        refs = [node.get("ref") for node in way.findall("nd")]
        segments = [[]]
        for ref in refs:
            if ref in nodes:
                segments[-1].append(nodes[ref])
            elif segments[-1]:
                segments.append([])
        ways[way.get("id")] = (tags(way), [s for s in segments if len(s) > 1])
    peaks = []
    for node in root.findall("node"):
        t = tags(node)
        if t.get("natural") == "peak" and t.get("ele", "").isdigit():
            radius = 3 if t.get("name") == "Αδράχτι" else 11
            peaks.append((*nodes[node.get("id")], int(t["ele"]), radius))
    return ways, peaks


def paint_geometry(ways):
    shape = (WIDTH, GROUND_ROWS)
    land = Image.new("L", shape)
    surface = Image.new("L", shape)
    paths = Image.new("L", shape)
    occupied = Image.new("1", shape)
    paint_land, paint_road, paint_path, paint_occupied = map(ImageDraw.Draw, (land, surface, paths, occupied))
    buildings = []
    areas = []
    roads = []
    for tags, segments in ways.values():
        cover = LAND.get(tags.get("natural"), LAND.get(tags.get("landuse")))
        highway = tags.get("highway", "").removesuffix("_link")
        for points in segments:
            closed = len(points) > 3 and points[0] == points[-1]
            if closed and cover:
                areas.append((cover, points))
            if tags.get("waterway") == "river":
                paint_land.line(points, fill=7, width=20)
                paint_land.line(points, fill=8, width=14)
            if highway in PATHS:
                paint_path.line(points, fill=255, width=2)
            if highway in ROADS:
                roads.append((ROADS[highway], points))
            elif tags.get("railway") == "rail":
                paint_road.line(points, fill=3, width=3)
            if closed and tags.get("building") not in (None, "no"):
                buildings.append((tags, points[:-1]))
                paint_occupied.polygon(points, fill=1)
    for cover, points in sorted(areas, key=lambda area: area[0]):
        paint_land.polygon(points, fill=cover)
    for width, points in sorted(roads, key=lambda road: road[0]):
        paint_road.line(points, fill=1, width=width + 2, joint="curve")
        paint_road.line(points, fill=2, width=width, joint="curve")
        paint_occupied.line(points, fill=1, width=width + 5)
    return map(np.asarray, (land, surface, paths, occupied)), buildings


def sharpen_summits(terrain, peaks):
    for x, row, height, radius in peaks:
        i, j = int(x), int(row)
        if not (radius < i < WIDTH - radius and radius < j < GROUND_ROWS - radius):
            continue
        patch = terrain[j - radius:j + radius + 1, i - radius:i + radius + 1]
        gain = min(68, height - terrain[j, i])
        if gain > 10:
            offset = np.arange(-radius, radius + 1)
            distance = np.hypot(offset[:, None] * 1.05, offset[None, :])
            patch += gain * np.clip((1 - distance / radius) * 2.3, 0, 1)


def terrain_shadow(terrain):
    shadow = np.zeros_like(terrain)
    for distance in (2, 4, 7, 11, 16, 23, 33, 48, 64, 90):
        east, south = distance, max(1, round(distance * 0.43))
        rise = terrain[:-south, :-east] - terrain[south:, east:]
        occlusion = np.clip((rise - distance * 1.7) / 18, 0, 1)
        area = shadow[south:, east:]
        np.maximum(area, occlusion, out=area)
    return shadow


def texture(terrain, land, roads):
    dy, dx = np.gradient(terrain, METERS_PER_PIXEL)
    slope = np.hypot(dx, dy)
    regional = np.asarray(Image.fromarray(NOISE.integers(0, 255, (63, 51), dtype=np.uint8))
                          .resize((WIDTH, GROUND_ROWS), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    detail = NOISE.integers(-3, 4, terrain.shape, dtype=np.int8)
    rock = ((land == 6) & (slope > 0.13)) | ((terrain > 315) & (slope > (0.67 + regional * 0.30)))
    wild = ((terrain > (250 + regional * 78)) | (land == 4) | (land == 5)) & ~rock
    wild &= (land == 0) | (land == 4) | (land == 5)

    base = np.zeros((*terrain.shape, 3), dtype=np.float32)
    base[:] = (159, 167, 102)
    base[wild] = (117, 142, 84)
    base[land == 1] = (177, 173, 99)
    base[land == 2] = (170, 172, 104)
    base[land == 3] = (184, 172, 127)
    base[land == 7] = (182, 169, 128)
    base[rock] = (189, 159, 122)
    base[roads == 1] = (157, 142, 108)
    base[roads == 2] = (202, 188, 157)
    base[roads == 3] = (121, 113, 94)
    base[land == 8] = (95, 151, 147)

    facing = (dx + dy * 0.43) / np.sqrt(1 + dx * dx + dy * dy)
    shading = np.round(facing * 7) * 7.5
    strata = np.floor((terrain + regional * 19) / 8) % 6
    strata = np.where(strata == 0, -15, np.where(strata == 2, 8, 0))
    shading += rock * strata + (regional - 0.5) * 29 + detail
    shading[(roads > 0) & (land != 8)] *= 0.47
    shading[land == 8] = ((regional - 0.5) * 17 + detail)[land == 8]
    occlusion = terrain_shadow(terrain)
    base -= occlusion[:, :, None] * (27, 29, 22)
    colors = np.clip(base + shading[:, :, None], 0, 255).astype(np.uint8)
    return colors, wild, rock, slope, regional, occlusion


ROOF_COLORS = [
    ("#d78c61", "#ba6c4e", "#97513f"), ("#cf8252", "#ad6549", "#915447"),
    ("#d69468", "#c27a59", "#a66b5c"), ("#c88965", "#aa6f58", "#8a5b51"),
    ("#ca7956", "#aa5d46", "#914f3e"), ("#b7a88e", "#9f927f", "#80766a"),
]


def building_commands(buildings, terrain):
    rows = defaultdict(list)
    for tags, points in buildings:
        x = sum(p[0] for p in points) / len(points)
        row = sum(p[1] for p in points) / len(points)
        if not (0 <= x < WIDTH and 0 <= row < GROUND_ROWS):
            continue
        datum = terrain[int(row), int(x)]
        base = [(round(px), round(screen_y(py, datum))) for px, py in points]
        if max(p[0] for p in base) - min(p[0] for p in base) < 2:
            continue
        if tags.get("building") in {"church", "chapel", "monastery"}:
            wall = RNG.randint(4, 7)
        else:
            levels = tags.get("building:levels", "")
            wall = min(11, max(2, int(levels) + 1)) if levels.isdigit() else RNG.choice((2, 3, 3, 4))
        roof = [(px, py - wall) for px, py in base]
        palette = RNG.choice(ROOF_COLORS)
        center = (x, sum(point[1] for point in roof) / len(roof) - 1)
        rows[int(row)].append((base, roof, center, palette))
    return rows


def paint_building(draw, base, roof, center, palette):
    draw.polygon([(x + 2, y + 1) for x, y in base], fill="#777856")
    edges = list(zip(range(len(base)), [*range(1, len(base)), 0]))
    for i, j in sorted(edges, key=lambda edge: base[edge[0]][1] + base[edge[1]][1]):
        wall = "#c5a484" if base[j][0] > base[i][0] else "#ae8b6f"
        draw.polygon((base[i], base[j], roof[j], roof[i]), fill=wall)
    draw.polygon(roof, fill=palette[1], outline=palette[2])
    for i, j in edges:
        corner, next_corner = roof[i], roof[j]
        color = palette[0] if corner[0] > next_corner[0] else palette[1]
        draw.polygon((corner, next_corner, center), fill=color)
    if max(x for x, _ in roof) - min(x for x, _ in roof) > 6:
        draw.point((round(center[0]), round(center[1])), palette[0])


TREE_COLORS = [
    ("#4f6542", "#668253", "#8f9e60"),
    ("#51683d", "#71864b", "#9ea862"),
    ("#5c7047", "#7d9155", "#9ca967"),
    ("#425a3b", "#617849", "#849353"),
    ("#596443", "#728151", "#929b61"),
]


def oak(draw, x, y, size, colors):
    shadow, middle, light = colors
    w = max(1, round(size * 0.58))
    draw.line((x, y - 1, x + w + 1, y + 1), fill="#64704a")
    draw.line((x, y - 4, x, y), fill="#6a5740")
    draw.polygon([(x - w, y - 3), (x - w, y - size + 2), (x - 2, y - size),
                  (x + 1, y - size - 1), (x + w, y - size + 1), (x + w, y - 3),
                  (x + 1, y - 2), (x - 1, y - 2)], fill=shadow)
    draw.polygon([(x - w, y - 4), (x - w, y - size + 2), (x - 1, y - size - 1),
                  (x + 1, y - size - 1), (x + w - 1, y - size + 1),
                  (x + 1, y - 4)], fill=middle)
    draw.line((x - w + 1, y - size + 1, x, y - size), fill=light)


TREE_SHADOW = ("#3d523b", "#4a6647", "#688056")


def tree_commands(land, occupied, paths, wild, rock, slope, regional, occlusion):
    rows = defaultdict(list)
    for row in range(4, GROUND_ROWS - 4, 5):
        for col in range(4, WIDTH - 4, 5):
            x, y = col + RNG.randint(-2, 2), row + RNG.randint(-2, 2)
            if occupied[y, x] or rock[y, x] or paths[y, x] or land[y, x] >= 7 or slope[y, x] > 0.88:
                continue
            density = 0.2 + regional[y, x] * 0.65 if wild[y, x] else 0.023
            if land[y, x] == 4:
                density = 0.85
            if RNG.random() < density:
                size = RNG.choice((4, 5, 6, 6, 7, 8, 9)) if wild[y, x] else RNG.randint(4, 7)
                palette = RNG.choice(TREE_COLORS)
                rows[y].append((x, size, TREE_SHADOW if occlusion[y, x] > 0.45 else palette))
    return rows


def render(terrain, colors, paths, buildings, trees):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#859464")
    trail = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image = ImageDraw.Draw(canvas)
    columns = np.arange(WIDTH)
    positions = np.round(screen_y(np.arange(GROUND_ROWS)[:, None], terrain)).astype(int)
    # Rasterize successive terrain strips from back to front. Walls and foliage share the
    # same painter order as the ground, so foreground slopes can hide distant objects.
    for row in range(1, GROUND_ROWS):
        current, previous = positions[row], positions[row - 1]
        top = np.maximum(0, np.minimum(current, previous))
        bottom = np.minimum(HEIGHT - 1, np.maximum(current, previous))
        visible = (top <= bottom) & (top < HEIGHT) & (bottom >= 0)
        if np.any(visible):
            x = columns[visible]
            y, end = top[visible], bottom[visible]
            row_colors = map(tuple, colors[row, x].tolist())
            for col, start, stop, shade in zip(x.tolist(), y.tolist(), end.tolist(), row_colors):
                image.line((col, start, col, stop), fill=shade)
            path_pixels = paths[row, x]
            for offset in range(int(np.max(end - y)) + 1):
                mask = y + offset <= end
                trail[y[mask] + offset, x[mask]] = path_pixels[mask]
        for x, size, palette in trees.get(row, ()):
            y = positions[row, x]
            if 0 <= y < HEIGHT + size:
                oak(image, x, y, size, palette)
        for args in buildings.get(row, ()):
            paint_building(image, *args)
    return canvas, Image.fromarray(trail)


def main():
    terrain = elevations()
    ways, peaks = read_features()
    sharpen_summits(terrain, peaks)
    (land, roads, paths, occupied), buildings = paint_geometry(ways)
    colors, wild, rock, slope, regional, occlusion = texture(terrain, land, roads)
    trees = tree_commands(land, occupied, paths, wild, rock, slope, regional, occlusion)
    canvas, trail = render(terrain, colors, paths, building_commands(buildings, terrain), trees)
    ASSETS.mkdir(parents=True, exist_ok=True)
    canvas.save(ASSETS / "meteora.webp", lossless=True)
    canvas.resize((384, 260), Image.Resampling.BOX).save(ASSETS / "overview.webp", quality=85)
    canvas.resize((1200, 812), Image.Resampling.BOX).save(ASSETS / "share.jpg", quality=87)
    trail_image = Image.new("RGBA", canvas.size, (246, 217, 125, 0))
    edge = trail.filter(ImageFilter.MaxFilter(3)).point(lambda value: value * 0.73)
    outline = Image.new("RGBA", canvas.size, (55, 62, 35, 0))
    outline.putalpha(edge)
    trail_image.putalpha(trail)
    Image.alpha_composite(outline, trail_image).save(ASSETS / "trails.png", optimize=True)

    places = json.loads((DATA / "places.json").read_text())
    for place in places:
        col, row = ground(place["lat"], place["lon"])
        height = terrain[round(row), round(col)]
        place["x"], place["y"] = round(col, 1), round(float(screen_y(row, height)), 1)
        if place["type"] == "monastery":
            place["y"] -= 4
    config = {
        "width": WIDTH, "height": HEIGHT, "metersPerPixel": METERS_PER_PIXEL,
        "northAngle": math.degrees(math.atan2(SIN, COS * GROUND_SCALE)),
        "places": places,
    }
    (ASSETS / "map.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(f"Rendered {WIDTH}×{HEIGHT} pixels, {len(buildings)} building footprints and {len(places)} places")


if __name__ == "__main__":
    main()
