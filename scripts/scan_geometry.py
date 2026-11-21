"""Local metre coordinates and rasterization for the 2016 Meteora scan."""

from pathlib import Path
import ctypes
import io
import json
import math
import struct
import subprocess

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
LAT, LON = 39.714, 21.628
METRES_NORTH = 111_195
METRES_EAST = METRES_NORTH * math.cos(math.radians(LAT))
# Terrain fit refined against roof colour at mapped footprints in both settlements.
SCAN_OFFSET = (-123.756, 756.188)
SCAN_ROTATION = math.radians(-0.306667)
SCAN_SCALE = .992921
GROUND_BOUNDS = (-4200, -4300, 4200, 4700)
GROUND_STEP = 2.5


def local(lat, lon):
    return (lon - LON) * METRES_EAST, (lat - LAT) * METRES_NORTH


def rotate(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def unpack_scan(path):
    blob = path.read_bytes()
    if blob[:4] != b"glTF":
        raise ValueError("Expected the original binary glTF from Zenodo")
    length = struct.unpack_from("<I", blob, 12)[0]
    document = json.loads(blob[20:20 + length])
    binary = memoryview(blob)[28 + length:]
    sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
    formats = {5123: "<u2", 5125: "<u4", 5126: "<f4"}

    def buffer_view(index):
        view = document["bufferViews"][index]
        offset = view.get("byteOffset", 0)
        return binary[offset:offset + view["byteLength"]], view.get("byteStride")

    def accessor(index):
        obj = document["accessors"][index]
        data, stride = buffer_view(obj["bufferView"])
        dtype = np.dtype(formats[obj["componentType"]])
        size = sizes[obj["type"]]
        return np.ndarray((obj["count"], size), dtype, buffer=data,
                          offset=obj.get("byteOffset", 0),
                          strides=(stride or size * dtype.itemsize, dtype.itemsize))

    image_bytes, _ = buffer_view(document["images"][0]["bufferView"])
    texture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    position_parts, normal_parts, uv_parts, index_parts = [], [], [], []
    offset = 0

    def walk(nodes, parent):
        nonlocal offset
        for node_id in nodes:
            node = document["nodes"][node_id]
            transform = parent @ np.array(node.get("matrix", np.eye(4).ravel())).reshape(4, 4).T
            if "mesh" in node:
                mesh = document["meshes"][node["mesh"]]
                for primitive in mesh["primitives"]:
                    attributes = primitive["attributes"]
                    p = accessor(attributes["POSITION"]) @ transform[:3, :3].T + transform[:3, 3]
                    n = accessor(attributes["NORMAL"]) @ transform[:3, :3].T
                    turn = rotate(SCAN_ROTATION).T
                    en = (p[:, (0, 2)] * [1, -1]) @ turn * SCAN_SCALE + SCAN_OFFSET
                    n_en = (n[:, (0, 2)] * [1, -1]) @ turn
                    position_parts.append(np.column_stack([en, p[:, 1]]))
                    normal_parts.append(np.column_stack([n_en, n[:, 1]]))
                    uv_parts.append(accessor(attributes["TEXCOORD_0"]))
                    index_parts.append(accessor(primitive["indices"]).reshape(-1, 3).astype("u4") + offset)
                    offset += len(p)
            walk(node.get("children", ()), transform)

    walk(document["scenes"][document["scene"]]["nodes"], np.eye(4))
    return (*(np.concatenate(parts) for parts in (position_parts, normal_parts, uv_parts, index_parts)), texture)


def texture_colors(texture, uv):
    x = np.clip((uv[..., 0] * texture.shape[1]).astype(int), 0, texture.shape[1] - 1)
    y = np.clip((uv[..., 1] * texture.shape[0]).astype(int), 0, texture.shape[0] - 1)
    return texture[y, x]


class Raster:
    def __init__(self, temporary):
        library = temporary / "scan-raster.so"
        subprocess.run(["cc", "-O3", "-shared", "-fPIC", str(ROOT / "scripts/scan_raster.c"),
                        "-o", str(library)], check=True)
        self.lib = ctypes.CDLL(str(library))
        contiguous = np.ctypeslib.ndpointer(flags="C_CONTIGUOUS")
        self.lib.raster.argtypes = [ctypes.c_int] * 3 + [contiguous] * 7 + [ctypes.c_int]
        self.lib.raster.restype = None

    def draw(self, size, screen, faces, values=None, material=None):
        width, height = size
        depth = np.full((height, width), -np.inf, np.float32)
        shade_only = values is None
        fragment_size = (1, 1) if shade_only else (height, width)
        fragments = np.zeros((*fragment_size, 8), np.float32)
        surfaces = np.zeros(fragment_size, np.uint8)
        self.lib.raster(width, height, len(faces),
                        np.ascontiguousarray(screen, dtype="f"),
                        np.ascontiguousarray(faces, dtype="u4"),
                        np.ascontiguousarray(values if values is not None else [0], dtype="f"),
                        np.ascontiguousarray(material if material is not None else [0], dtype="u1"),
                        depth, fragments, surfaces, shade_only)
        return depth if shade_only else (depth, fragments, surfaces)


class ElevationTiles:
    """Mapzen Terrarium elevation for the landscape outside the scan."""

    def __init__(self, folder):
        tiles = {}
        for path in folder.glob("*.png"):
            x, y = map(int, path.stem.split("-"))
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            tiles[x, y] = rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256 - 32768
        self.x0, self.y0 = (min(point[i] for point in tiles) for i in (0, 1))
        x1, y1 = (max(point[i] for point in tiles) for i in (0, 1))
        self.grid = np.empty(((y1 - self.y0 + 1) * 256, (x1 - self.x0 + 1) * 256), dtype=np.float32)
        for (x, y), tile in tiles.items():
            left, top = (x - self.x0) * 256, (y - self.y0) * 256
            self.grid[top:top + 256, left:left + 256] = tile

    def sample(self, e, n):
        lon, lat = e / METRES_EAST + LON, n / METRES_NORTH + LAT
        mercator = np.arcsinh(np.tan(np.radians(lat)))
        x = (lon + 180) / 360 * 2**21 - self.x0 * 256 - .5
        y = (1 - mercator / np.pi) / 2 * 2**21 - self.y0 * 256 - .5
        return sample(self.grid, x, y)


def sample(grid, x, y):
    """Bilinearly sample a grid, extending its edges."""
    x = np.clip(x, 0, grid.shape[1] - 1)
    y = np.clip(y, 0, grid.shape[0] - 1)
    ix = np.minimum(x.astype(int), grid.shape[1] - 2)
    iy = np.minimum(y.astype(int), grid.shape[0] - 2)
    fx, fy = x - ix, y - iy
    return ((1 - fy) * ((1 - fx) * grid[iy, ix] + fx * grid[iy, ix + 1])
            + fy * ((1 - fx) * grid[iy + 1, ix] + fx * grid[iy + 1, ix + 1]))


class Ground:
    left, bottom, right, top = GROUND_BOUNDS
    step = GROUND_STEP
    size = tuple(round(length / GROUND_STEP) for length in (right - left, top - bottom))

    def coords(self, en):
        coords = np.array(en, dtype=np.float64, copy=True)
        coords[..., 0] = (coords[..., 0] - self.left) / self.step
        coords[..., 1] = (self.top - coords[..., 1]) / self.step
        return coords

    def pixel(self, e, n):
        return ((e - self.left) / self.step).astype(int).clip(0, self.size[0] - 1), (
            (self.top - n) / self.step).astype(int).clip(0, self.size[1] - 1)

    def height(self, e, n):
        return sample(self.heights, (e - self.left) / self.step - .5, (self.top - n) / self.step - .5)

    def repair_grade(self, scan, normals, center, extent, elevation, grade, feather):
        """Trim captured building relief to the grade measured at the surrounding streets."""
        center, extent, grade = map(np.asarray, (center, extent, grade))

        def correction(en, z):
            offset = en - center
            edge = np.maximum(np.abs(offset) - extent, 0).max(axis=-1)
            t = np.minimum(edge / feather, 1)
            weight = 1 - t*t*(3 - 2*t)
            target = elevation + offset @ grade
            return np.minimum(z, target), weight

        cx, cy = self.coords(center)
        cols, rows = np.ceil((extent + feather) / self.step).astype(int)
        x = slice(int(cx)-cols, int(cx)+cols+2)
        y = slice(int(cy)-rows, int(cy)+rows+2)
        xx, yy = np.meshgrid(np.arange(x.start, x.stop), np.arange(y.start, y.stop))
        en = np.stack([self.left + (xx+.5)*self.step, self.top - (yy+.5)*self.step], axis=-1)
        patch = self.heights[y, x]
        target, weight = correction(en, patch)
        patch += (target - patch) * weight

        target, weight = correction(scan[:, :2], scan[:, 2])
        changed = (weight > 0) & (target < scan[:, 2])
        scan[changed, 2] += (target[changed] - scan[changed, 2]) * weight[changed]
        grade_normal = np.array([-grade[0], -grade[1], 1])
        grade_normal /= np.linalg.norm(grade_normal)
        normals[changed] += (grade_normal - normals[changed]) * weight[changed, None]

    @staticmethod
    def blend(coverage):
        t = np.clip((coverage - .67) / .29, 0, 1)
        return t * t * (3 - 2 * t)

    def __init__(self, raster, scan, faces, uv, texture, elevation):
        screen = np.column_stack([self.coords(scan[:, :2]), scan[:, 2]])
        values = np.zeros((len(scan), 8), np.float32)
        values[:, 6:] = uv
        heights, surface, _ = raster.draw(self.size, screen, faces, values, np.ones(len(faces)))
        known = np.isfinite(heights)
        self.scan_known = known
        self.color = texture_colors(np.asarray(texture), surface[:, :, 6:])
        self.color[~known] = [117, 125, 84]
        width, height = self.size
        e = self.left + (np.arange(width) + .5) * self.step
        n = self.top - (np.arange(height) + .5) * self.step
        dem = elevation.sample(e[None, :], n[:, None])
        offset = np.median(dem[known] - heights[known])
        self.dem = (dem - offset).astype(np.float32)
        mask = Image.fromarray((known * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(15))
        self.coverage = np.asarray(mask, dtype=np.float32) / 255
        self.heights = np.where(known, self.dem + (np.where(known, heights, self.dem) - self.dem)
                                * self.blend(self.coverage), self.dem)
        self.elevation_offset = float(offset)
        noise = np.random.default_rng(881)

        def variation(shape):
            stock = Image.fromarray(noise.integers(0, 255, shape, dtype=np.uint8))
            return np.asarray(stock.resize(self.size, Image.Resampling.BICUBIC), dtype=np.float32)/255 - .5

        coarse = variation((31, 29))
        fine = variation((98, 93))
        self.variation = .75*coarse + .25*fine
        self.woodland = np.clip((self.dem + 335) / 88 + coarse*1.4 + fine*.28, 0, 1)

    def mesh(self, scan, scan_norm, scan_faces, uv):
        px = self.coords(scan[:, :2]) - .5
        weight = self.blend(sample(self.coverage, *px.T))
        dem_height = sample(self.dem, *px.T)
        scan = scan.copy()
        scan[:, 2] = dem_height + (scan[:, 2] - dem_height) * weight
        # An overlapping DEM skirt touches the scan only in the blended edge band.
        rows, cols = self.dem.shape
        yy, xx = np.meshgrid(np.arange(0, rows, 4), np.arange(0, cols, 4), indexing="ij")
        east = self.left + (xx + .5) * self.step
        north = self.top - (yy + .5) * self.step
        z = self.dem[yy, xx]
        dz_n, dz_e = np.gradient(z, self.step * 4)
        apron_norm = np.stack([-dz_e, dz_n, np.ones_like(z)], axis=-1).reshape(-1, 3)
        apron_pos = np.stack([east, north, z - .08], axis=-1).reshape(-1, 3)
        ids = np.arange(yy.size, dtype=np.uint32).reshape(yy.shape)
        remove = self.coverage[yy, xx] > .80
        keep = ~(remove[:-1, :-1] & remove[1:, :-1] & remove[:-1, 1:] & remove[1:, 1:])
        a, b, c, d = ids[:-1, :-1][keep], ids[1:, :-1][keep], ids[:-1, 1:][keep], ids[1:, 1:][keep]
        apron_faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
        return (np.concatenate([scan, apron_pos]),
                np.concatenate([scan_norm, apron_norm]),
                np.concatenate([scan_faces, apron_faces + len(scan)]),
                np.concatenate([uv, np.zeros((len(apron_pos), 2))]),
                np.concatenate([np.ones(len(scan_faces), np.uint8), np.full(len(apron_faces), 2, np.uint8)]))


def signed_area(poly):
    return np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))*.5

def cross2(u,v):
    return u[0]*v[1]-u[1]*v[0]

def triangulate(poly):
    remaining=list(range(len(poly)))
    out=[]
    while len(remaining)>3:
        for i,b in enumerate(remaining):
            a,c=remaining[i-1],remaining[(i+1)%len(remaining)]
            v1,v2,v3=poly[a],poly[b],poly[c]
            if cross2(v2-v1,v3-v2)<=.001:
                continue
            others=poly[[j for j in remaining if j not in (a,b,c)]]
            def left(x,y):
                return (y[0]-x[0])*(others[:,1]-x[1])-(y[1]-x[1])*(others[:,0]-x[0]) > .001
            if np.any(left(v1,v2) & left(v2,v3) & left(v3,v1)):
                continue
            out.append((a,b,c))
            remaining.pop(i)
            break
        else:
            return out
    out.append(tuple(remaining))
    return out


def half_plane(poly, height):
    """Clip a convex roof piece at a roof-plane boundary."""
    out = []
    for point, previous, a, b in zip(poly, np.roll(poly, 1, axis=0), height, np.roll(height, 1)):
        if (a < 0) != (b < 0):
            out.append((a * previous - b * point) / (a - b))
        if a >= 0:
            out.append(point)
    return np.asarray(out)
