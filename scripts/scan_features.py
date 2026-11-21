"""Project OpenStreetMap features and the hand-drawn civic details."""

import gzip
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from scan_geometry import local

ROADS = {
    "motorway": 12, "trunk": 9, "primary": 8, "secondary": 7, "tertiary": 6,
    "residential": 6, "unclassified": 5, "living_street": 5, "pedestrian": 4, "service": 3,
}
UNPAVED = {"unpaved", "dirt", "earth", "gravel", "fine_gravel", "ground", "grass", "sand", "mud", "compacted"}
PATHS = {"footway", "path", "steps", "track"}
PARKS = {
    "bare_rock": 1, "scrub": 2, "wood": 2, "forest": 2,
    "orchard": 3, "farmland": 3, "vineyard": 3, "grass": 6, "grassland": 6,
    "water": 7, "park": 6, "recreation_ground": 6,
}
TOWN_FOUNTAIN = "4332014836"
FOOTBALL_STADIUM = "800498756"
DORMITION_COURT = "609218809"
# Cliff-top offsets measured against the roof texture in the 2016 scan.
# Each campus keeps the original footprint shapes and relative building positions.
MONASTERIES = {
    "great-meteoron": ((15, -25), (1066256733, 1066256734, 1066256735, 1066256736,
                                  1066256737, 1066256738, 1066256739, 1066256740,
                                  1066256741, 1066256742, 1066256769)),
    "varlaam": ((12.5, -27.5), (1064680426, 1064680429, 1064680430, 1064680431,
                              1064680432, 1064680433)),
    "roussanou": ((7.5, -22.5), (379736574, 1369606372)),
    "st-nicholas": ((10, -20), (1066373214,)),
    "holy-trinity": ((5, -22.5), (1064663873,)),
    "st-stephen": ((0, -15), (1064658382, 1064658383, 1064658386, 1064658387)),
}
MONASTERY_BUILDINGS = {
    str(id_): offset for offset, ids in MONASTERIES.values() for id_ in ids
}


def read_osm(path):
    with gzip.open(path) as file:
        root = ET.parse(file).getroot()

    def tags(element):
        return {tag.get("k"): tag.get("v") for tag in element.findall("tag")}

    nodes = {n.get("id"): (local(float(n.get("lat")), float(n.get("lon"))), tags(n))
             for n in root.findall("node")}
    ways = {}
    for way in root.findall("way"):
        points = []
        segments = []
        for node in way.findall("nd"):
            item = nodes.get(node.get("ref"))
            if item is not None:
                points.append(item[0])
            elif points:
                segments.append(points)
                points = []
        if points:
            segments.append(points)
        ways[way.get("id")] = (tags(way), [segment for segment in segments if len(segment) > 1])
    return nodes, ways


class Pitch:
    def __init__(self, points, sport, stadium):
        self.points = np.asarray(points)
        self.center = self.points.mean(axis=0)
        edges = np.roll(self.points, -1, axis=0) - self.points
        long = edges[np.linalg.norm(edges, axis=1).argmax()]
        self.long = long / np.linalg.norm(long)
        self.short = np.array([-self.long[1], self.long[0]])
        centered = self.points - self.center
        self.length = np.ptp(centered @ self.long) * .5
        self.width = np.ptp(centered @ self.short) * .5
        self.sport = sport
        self.stadium = stadium

    def stadium_perimeter(self):
        radius = np.hypot(self.length * .35, self.width) + 11.0
        angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        speedway = np.column_stack([
            np.cos(angles) * radius + np.sign(np.cos(angles)) * self.length * .65,
            np.sin(angles) * radius,
        ])
        return self.center + speedway @ np.array([self.long, self.short])

    def draw(self, color, world, terrain, sun, shades, palette):

        distance = 1.55 * (self.length + self.width)
        e, n = world[:, :, 0] - self.center[0], world[:, :, 1] - self.center[1]
        interesting = terrain & (np.abs(e) < distance) & (np.abs(n) < distance)
        if not interesting.any():
            return
        xx, yy = e[interesting], n[interesting]
        u = np.abs(self.long[0] * xx + self.long[1] * yy)
        v = np.abs(self.short[0] * xx + self.short[1] * yy)
        light = sun[interesting]
        pixels = color[interesting]
        a, b = self.length, self.width
        field = (u < a) & (v < b)
        if self.stadium:
            # The mapped football rectangle anchors the visually checked oval track.
            straight = a * .65
            # The inside running curve clears the mapped grass rectangle at its corners.
            inside = np.hypot(a - straight, b) + 1.2
            curve = np.hypot(np.maximum(u - straight, 0), v)
            infield = curve < inside
            track = (curve >= inside) & (curve <= inside + 8.5)
            stripe = (np.floor((xx * self.long[0] + yy * self.long[1]) * .12) % 2) * .09
            grass = palette("335743 406c4c 477950 548653 649555 70a45c 84ad65")
            pixels[infield] = shades(grass, .54 + light*.20 + stripe)[infield]
            clay = palette("714e52 8f5b55 a86b60 bb7868 cd826f db907a e3a090")
            lanes = (((curve-inside+.12) % 1.33) < .28) & track
            pixels[track] = shades(clay, .46 + light*.24 + .20*lanes)[track]
            marking = (((u > a-1.75) | (v > b-1.65)) | (u < .65)
                       | (np.abs(np.hypot(u,v)-9.15) < .64)
                       | ((v < 20) & (np.abs(u-(a-16.5)) < .62))
                       | ((u > a-16.5) & (np.abs(v-20) < .62)))
        else:
            court = palette("4f625b 5d7666 6b866a 81936f 95a37e")
            pixels[field] = shades(court, .55 + light*.24)[field]
            marking = (u > a-1.05) | (v > b-1.02) | (u < .38)
            if self.sport == "basketball":
                marking |= (np.abs(np.hypot(u,v)-1.8) < .36)
                marking |= ((u > a-4.0) & (np.abs(v-2.15) < .35))
        lines = marking & field
        pixels[lines] = shades(palette("969f7c b8c29a dce1bd f5efce"), .46 + light*.45)[lines]
        color[interesting] = pixels


def masks_for(ground, ways):
    land, road, trail, occupied = (Image.new("L", ground.size) for _ in range(4))
    paint_land, paint_road, paint_trail, paint_occupied = map(ImageDraw.Draw, (land,road,trail,occupied))
    parcels, buildings, roads, pitches = [], [], [], []
    for way_id, (tags, segments) in ways.items():
        highway = tags.get("highway", "").removesuffix("_link")
        cover = PARKS.get(tags.get("natural"), PARKS.get(tags.get("landuse"), PARKS.get(tags.get("leisure"))))
        is_paved_area = tags.get("area") == "yes" and highway == "pedestrian"
        if tags.get("amenity") == "school" or tags.get("place") == "square" or is_paved_area:
            cover = 4
        if tags.get("leisure") == "pitch":
            cover = 5
        if way_id == DORMITION_COURT:
            cover = 8
        for points in segments:
            px = [tuple(point) for point in ground.coords(points)]
            closed = points[0] == points[-1] and len(set(points)) >= 3
            if closed and cover:
                parcels.append((cover, px))
            if closed and tags.get("building") not in (None, "no"):
                offset = MONASTERY_BUILDINGS.get(way_id)
                footprint = np.asarray(points[:-1])
                if offset is not None:
                    footprint = footprint + offset
                buildings.append((way_id, tags, footprint, offset is not None))
                paint_occupied.polygon([tuple(point) for point in ground.coords(footprint)], fill=255)
            if closed and tags.get("leisure") == "pitch" and tags.get("sport") in (
                    "soccer", "basketball", "volleyball", "tennis"):
                pitches.append(Pitch(points[:-1], tags["sport"], way_id == FOOTBALL_STADIUM))
            if highway in ROADS and not is_paved_area:
                roads.append((px, ROADS[highway], tags.get("surface") not in UNPAVED))
            if highway in PATHS:
                paint_trail.line(px, fill=255, width=1)
            if tags.get("railway") == "rail":
                paint_road.line(px, fill=4, width=2)
            if tags.get("waterway") == "river":
                paint_land.line(px, fill=7, width=20)
    for cover, points in sorted(parcels, key=lambda item: item[0]):
        paint_land.polygon(points, fill=cover)
    for pitch in pitches:
        if pitch.stadium:
            outline = [tuple(point) for point in ground.coords(pitch.stadium_perimeter())]
            paint_land.polygon(outline, fill=5)
            paint_occupied.polygon(outline, fill=255)
    for points, width, paved in sorted(roads, key=lambda item: item[1]):
        radius = max(1, round(width / ground.step))
        paint_road.line(points, fill=2 if paved else 3, width=radius+2, joint="curve")
        paint_occupied.line(points, fill=255, width=radius+3, joint="curve")
    for points, width, paved in sorted(roads, key=lambda item: item[1]):
        paint_road.line(points, fill=1 if paved else 3, width=max(1,round(width/ground.step)), joint="curve")
    return tuple(np.asarray(img) for img in (land,road,trail,occupied)), buildings, pitches


def civic_structures(scene, ground, nodes, pitches):
    pos, _ = nodes[TOWN_FOUNTAIN]
    center = np.asarray(pos)
    datum = float(ground.height(*center))
    angles = np.linspace(0, 2*math.pi, 12, endpoint=False)
    radii = 1 + .12*np.sin(angles*3+1.3)
    rim = np.column_stack([np.cos(angles)*3.3, np.sin(angles)*4.2]) * radii[:,None] + center
    inside = center + (rim - center)*.76
    for i, p in enumerate(rim):
        j = (i+1) % len(rim)
        q = rim[j]
        scene.face([(*p,datum-.7),(*q,datum-.7),(*q,datum+.95),(*p,datum+.95)], [(0,0)]*4, 60)
        scene.face([(*p,datum+.95),(*q,datum+.95),(*inside[j],datum+.95),(*inside[i],datum+.95)],
                   [(0,0)]*4, 60)
        scene.face([(*center,datum+.72),(*inside[i],datum+.72),(*inside[j],datum+.72)], [(0,0)]*3, 61)
    # Raised stone lip and a small cascade above the basin.
    west, east = center + [-1.65,1.1], center + [1.2,1.1]
    scene.face([(*west,datum+.65),(*east,datum+.65),(*east,datum+2.12),(*west,datum+2.40)],
               [(0,0)]*4, 62)
    rear = center + [0,2.85]
    scene.face([(*west,datum+2.40),(*east,datum+2.12),(*rear,datum+1.72)], [(0,0)]*3, 60)

    for pitch in pitches:
        if not pitch.stadium:
            continue
        south = pitch.short * (1 if pitch.short[1] < 0 else -1)
        y = np.hypot(pitch.length*.35, pitch.width) + 11.0
        step = pitch.center + south*y
        length = pitch.length*.79
        for i in range(3):
            p = step + south*i*1.4
            q = p + south*1.45
            strip = [p-pitch.long*length,p+pitch.long*length,q+pitch.long*length,q-pitch.long*length]
            z = float(ground.height(*p)) + 1.2 + .9*i
            scene.face([(*v,z) for v in strip], [(0,0)]*4, 60)
            scene.face([(*strip[0],z-.9),(*strip[1],z-.9),(*strip[1],z),(*strip[0],z)], [(0,0)]*4, 63)
