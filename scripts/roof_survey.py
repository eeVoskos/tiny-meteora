"""Fit stylized roof geometry to a local RGB, DSM and mask GeoTIFF survey."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scan_geometry import (LAT, LON, METRES_EAST, METRES_NORTH, half_plane,
                           signed_area, triangulate)

RESIDENTIAL = {
    "yes", "house", "residential", "apartments", "detached", "semidetached_house",
    "bungalow", "terrace", "dormitory",
}
SURVEY_ROOF_BASE = 100
MIDTONES = np.array([
    (116, 73, 66), (138, 83, 70), (157, 94, 76), (177, 103, 80), (194, 116, 86), (216, 136, 100),
    (130, 87, 71), (151, 100, 78), (173, 111, 82), (192, 129, 92), (213, 148, 102), (229, 167, 118),
    (106, 79, 74), (127, 90, 78), (151, 105, 85), (176, 119, 92), (194, 141, 102), (218, 166, 120),
    (88, 89, 95), (111, 113, 118), (140, 140, 143), (166, 167, 168), (195, 194, 189), (218, 216, 206),
    (109, 104, 98), (137, 130, 119), (158, 149, 133), (180, 172, 155), (205, 194, 172), (229, 218, 196),
], dtype=float)
LIGHT_STEPS = np.array([.51, .63, .74, .83, .92, 1., 1.08, 1.16, 1.24])
SURVEY_ROOFS = np.clip(np.rint(MIDTONES[:,None,:]*LIGHT_STEPS[None,:,None]), 0, 255).astype("u1")


def utm34(points):
    """Project the map's local east/north coordinates to WGS84 / UTM zone 34N."""
    east, north = np.asarray(points).T
    lat = np.deg2rad(LAT + north / METRES_NORTH)
    lon = np.deg2rad(LON + east / METRES_EAST - 21)
    a, e2, k = 6378137., .00669437999014, .9996
    sin, cos = np.sin(lat), np.cos(lat)
    tangent = np.tan(lat)
    t, c, l = tangent**2, (e2/(1-e2))*cos**2, cos*lon
    nu = a / np.sqrt(1-e2*sin**2)
    m = a * ((1-e2/4-3*e2**2/64-5*e2**3/256)*lat
             - (3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*lat)
             + (15*e2**2/256+45*e2**3/1024)*np.sin(4*lat)
             - (35*e2**3/3072)*np.sin(6*lat))
    x = 500000 + k*nu*(l+(1-t+c)*l**3/6+(5-18*t+t**2+72*c-58*e2/(1-e2))*l**5/120)
    y = k*(m + nu*tangent*(l**2/2+(5-t+9*c+4*c**2)*l**4/24
                            +(61-58*t+t**2+600*c-330*e2/(1-e2))*l**6/720))
    return np.column_stack([x, y])


@dataclass(frozen=True)
class Tile:
    path: Path
    size: tuple
    corner: tuple
    step: tuple

    @classmethod
    def open(cls, path):
        with Image.open(path) as im:
            keys = im.tag_v2.get(34735, ())
            crs = {key: value for key, tag, count, value in zip(*[iter(keys[4:])]*4)
                   if not tag and count == 1}
            transform = im.tag_v2.get(34264)
            if crs.get(3072) != 32634 or not transform or transform[1] or transform[4]:
                raise ValueError(f"Expected a north-up EPSG:32634 survey GeoTIFF in {path.name}")
            if im.mode != "L":
                raise ValueError(f"Expected a raster mask in {path.name}")
            return cls(path, im.size, (transform[3], transform[7]), (transform[0], transform[5]))

    def pixels(self, points):
        return (points - self.corner) / self.step

    def layer(self, name):
        return self.path.with_name(self.path.name.removesuffix("_mask.tif") + f"_{name}.tif")

    def read(self):
        layers = []
        for name, mode in (("rgb", "RGB"), ("dsm", "F"), ("mask", "L")):
            path = self.layer(name)
            with Image.open(path) as image:
                a = image.tag_v2.get(34264)
                if (image.mode != mode or image.size != self.size or not a or
                        (a[3], a[7]) != self.corner or (a[0], a[5]) != self.step):
                    raise ValueError(f"Survey layers are out of alignment at {path.name}")
                layers.append(np.asarray(image))
        rgb, dsm, mask = layers
        # Some libtiff/Pillow builds return big-endian float samples in native
        # array order. Choose the decoding consistent with local elevation.
        def plausible(z):
            with np.errstate(invalid="ignore"):
                return ((z > 100) & (z < 1500)).mean()
        probe = dsm[::16, ::16]
        if plausible(probe.byteswap()) > plausible(probe):
            dsm = dsm.byteswap()
        if plausible(dsm[::16, ::16]) < .30:
            raise ValueError(f"Could not decode surface elevations in {self.layer('dsm').name}")
        return rgb, dsm, mask


@dataclass
class Roof:
    color: int
    shape: str | None
    origin: np.ndarray
    basis: np.ndarray
    planes: np.ndarray | None


def best_color(rgb):
    """Select an atlas palette from the color of lit, masked roof samples."""
    values = rgb.astype(float)
    light = values @ [.25, .58, .17]
    visible = (light >= np.percentile(light, 36)) & (light <= np.percentile(light, 90))
    r, g, b = np.median(values[visible], axis=0)
    warmth = np.clip((r-g-13)/30, 0, 1)
    # The orthophoto is slightly magenta. Warm its clay tiles to fit the atlas
    # palette while keeping concrete and metal neutral.
    color = np.array([r+5*warmth, g+12*warmth, b-15*warmth])
    difference = MIDTONES-color
    return int(np.argmin((difference**2) @ [1., 1.25, .85]))


def roof_planes(outline):
    """Candidate normalized gable and hip surfaces along both footprint axes."""
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    for u, v in ((0, 1), (1, 0)):
        half_width = (hi[v] - lo[v]) / 2
        if half_width < 2.0:
            continue
        cross_axis = np.eye(2)[v]
        long_axis = np.eye(2)[u]
        for offset in (0, -.16, .16):
            rise_run, fall_run = half_width*(1+offset), half_width*(1-offset)
            sides = [[-lo[v]/rise_run, *(cross_axis/rise_run)], [hi[v]/fall_run, *(-cross_axis/fall_run)]]
            yield "gable", np.asarray(sides)
            hip_run = min(half_width*.94, (hi[u]-lo[u])*.49)
            end = long_axis / hip_run
            ends = [[-lo[u]/hip_run, *end], [hi[u]/hip_run, *-end]]
            yield "hip", np.asarray(sides+ends)


def fit_roof(poly, sample, height, color):
    origin = poly.mean(axis=0)
    edges = np.roll(poly, -1, axis=0) - poly
    longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    axis = longest / np.linalg.norm(longest)
    basis = np.array([axis, [-axis[1], axis[0]]])
    roof = Roof(color, None, origin, basis, None)
    if len(height) < 30:
        return roof
    xy = (sample-origin) @ basis.T
    samples = np.column_stack([np.ones(len(xy)), xy])
    z = height - np.median(height)
    spread = np.percentile(z, 90)-np.percentile(z, 10)
    outline = (poly-origin) @ basis.T
    # Compare pitched templates to a constant. Outliers near overhanging trees,
    # chimneys and walls are downweighted with a capped absolute loss.
    best = np.minimum(np.abs(z), 1.5).mean()
    if spread <= .82 and best < .34:
        return Roof(color, "flat", origin, basis, np.zeros((1, 3)))
    for shape, planes in roof_planes(outline):
        features = (samples @ planes.T).min(axis=1)
        keep = np.ones(len(z), bool)
        for _ in range(3):
            x = features[keep]
            y = z[keep]
            x0, y0 = x.mean(), y.mean()
            denom = np.sum((x-x0)**2)
            if denom < .001:
                break
            rise = np.sum((x-x0)*(y-y0))/denom
            residual = np.abs(z - (y0 + rise*(features-x0)))
            keep = residual < max(.46, np.quantile(residual, .75)*1.9)
        else:
            score = np.minimum(residual, 1.5).mean() + (.12 if shape == "hip" else .10)
            if (.65 <= rise <= 5.4 and score < best and np.quantile(residual,.85) < 1.12
                    and rise * np.max(np.linalg.norm(planes[:,1:], axis=1)) <= .90):
                best = score
                roof.shape, roof.planes = shape, rise*planes
    plane = np.linalg.lstsq(samples, z, rcond=None)[0]
    projected = outline @ plane[1:]
    residual = np.abs(z - samples @ plane)
    score = np.minimum(residual, 1.5).mean() + .16
    if (score < best and np.quantile(residual, .85) < .56 and
            .80 <= np.ptp(projected) <= 4.6 and np.linalg.norm(plane[1:]) < .56):
        roof.shape = "shed"
        roof.planes = np.array([[-projected.min(), *plane[1:]]])
    return roof


class RoofSurvey:
    def __init__(self, folder):
        paths = sorted(folder.glob("*_mask.tif"))
        if not paths:
            raise ValueError(f"No *_mask.tif survey layers found in {folder}")
        self.tiles = [Tile.open(p) for p in paths]
        self.read = lru_cache(maxsize=6)(Tile.read)

    def patch(self, tile, poly, in_tile):
        # Request a compact window around the mapped outline.
        left, top = np.floor(in_tile.min(axis=0)).astype(int)-1
        right, bottom = np.ceil(in_tile.max(axis=0)).astype(int)+1
        roi = np.s_[top:bottom, left:right]
        zone = Image.new("L", (right-left, bottom-top))
        points = list(map(tuple, in_tile-[left+.5, top+.5]))
        ImageDraw.Draw(zone).polygon(points, fill=255)
        core = np.asarray(zone.filter(ImageFilter.MinFilter(3))) != 0
        rgb, dsm, mask = self.read(tile)
        rgb, dsm, mask = rgb[roi], dsm[roi], mask[roi]
        valid = (mask != 0) & np.isfinite(dsm) & (dsm > 100) & (dsm < 1500)
        if (valid & core).sum() < max(16, core.sum()*.56):
            return None
        # Keep a buffer from the survey's roof edges as well as the footprint.
        safe = np.asarray(Image.fromarray(valid.astype("u1")*255).filter(ImageFilter.MinFilter(3))) != 0
        safe &= core
        if safe.sum() < 16:
            return None
        row, col = np.nonzero(safe)
        rgb = rgb[safe]
        good = np.any(rgb > 20, axis=1) & np.all(rgb < 254, axis=1)
        if good.sum() < 12:
            return None
        color = best_color(rgb[good])
        # UTM and the scene's local frame differ by a small, nearly affine
        # rotation/scale over a single building. Fit it to this OSM outline.
        target = np.column_stack([in_tile-in_tile.mean(axis=0), np.ones(len(poly))])
        transform = np.linalg.lstsq(target, poly, rcond=None)[0]
        sample = np.column_stack([col+left+.5, row+top+.5]) - in_tile.mean(axis=0)
        sample = np.column_stack([sample, np.ones(len(sample))]) @ transform
        return fit_roof(poly, sample, dsm[safe], color)

    def match(self, buildings, exclude):
        profiles = {}
        by_tile = {}
        for way_id, tags, poly, on_cliff in buildings:
            area = abs(signed_area(poly))
            if (on_cliff or way_id in exclude or tags.get("building") not in RESIDENTIAL
                    or tags.get("amenity") or not 24 <= area <= 1600):
                continue
            utm = utm34(poly)
            options = []
            for tile in self.tiles:
                px = tile.pixels(utm)
                margin = min(px.min(), (np.asarray(tile.size)-px.max(axis=0)).min())
                if margin >= 3:
                    options.append((margin, tile, px))
            if not options:
                continue
            _, tile, px = max(options, key=lambda option: option[0])
            by_tile.setdefault(tile, []).append((way_id, poly, px))
        for tile, footprints in by_tile.items():
            for way_id, poly, px in footprints:
                profile = self.patch(tile, poly, px)
                if profile is not None:
                    profiles[way_id] = profile
        counts = Counter(p.shape or "color only" for p in profiles.values())
        print(f"Roof survey matched {len(profiles)} footprints: " +
              ", ".join(f"{count} {name}" for name,count in sorted(counts.items())), flush=True)
        return profiles


def draw_observed_roof(scene, roof, polygon, eave, wall_material, roof_material):
    """Clip the chosen roof planes to the footprint and close its gable ends."""
    if roof.shape is None:
        return False
    outline = (polygon-roof.origin) @ roof.basis.T
    triangles = triangulate(outline)
    if len(triangles) != len(outline) - 2:
        return False
    lifted = np.column_stack([np.ones(len(outline)), outline])
    planes = roof.planes
    for indices in triangles:
        triangle = lifted[list(indices)]
        for i, plane in enumerate(planes):
            limits = np.delete(planes-plane, i, axis=0)
            face = triangle
            for boundary in limits:
                face = half_plane(face, face @ boundary)
                if len(face) < 3:
                    break
            if len(face) < 3:
                continue
            xy = face[:, 1:]
            points = xy @ roof.basis + roof.origin
            points = np.column_stack([points, eave + face @ plane])
            for j in range(1, len(points)-1):
                idx = [0, j, j+1]
                scene.face(points[idx], xy[idx], roof_material)

    if roof.shape == "flat":
        return True
    for a, b in zip(lifted, np.roll(lifted, -1, axis=0)):
        start, end = planes @ a, planes @ b
        change = end-start
        cuts = [0., 1.]
        for i in range(len(planes)):
            for j in range(i+1, len(planes)):
                slope = change[i]-change[j]
                if abs(slope) > .0001:
                    t = (start[j]-start[i])/slope
                    if .0001 < t < .9999:
                        cuts.append(t)
        cuts = np.asarray(sorted(set(cuts)))
        profile = np.maximum(0, (start+cuts[:, None]*change).min(axis=1))
        length = np.linalg.norm(b-a)
        for i in range(len(cuts)-1):
            t, u = cuts[i:i+2]
            h, k = profile[i:i+2]
            if max(h,k) < .03:
                continue
            p, q = (a[1:]+np.array([[t],[u]])*(b-a)[1:]) @ roof.basis + roof.origin
            base_p, base_q = (*p, eave), (*q, eave)
            end_p, end_q = (*p, eave+h), (*q, eave+k)
            if k < .001:
                scene.face([base_p, base_q, end_p], [(t*length,0),(u*length,0),(t*length,h)], wall_material)
            elif h < .001:
                scene.face([base_p, base_q, end_q], [(t*length,0),(u*length,0),(u*length,k)], wall_material)
            else:
                scene.face([base_p, base_q, end_q, end_p],
                           [(t*length,0),(u*length,0),(u*length,k),(t*length,h)], wall_material)
    return True


def shade_survey_roofs(image, material, light):
    surveyed = (material >= SURVEY_ROOF_BASE) & (material < SURVEY_ROOF_BASE + len(SURVEY_ROOFS))
    if surveyed.any():
        colors = material[surveyed] - SURVEY_ROOF_BASE
        shades = np.clip(np.rint(light[surveyed]*(len(LIGHT_STEPS)-1)), 0, len(LIGHT_STEPS)-1).astype(int)
        image[surveyed] = SURVEY_ROOFS[colors, shades]
