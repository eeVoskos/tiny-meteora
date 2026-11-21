"""Project streets, driving routes, and waterways for the live map layer."""

import json

import numpy as np
from PIL import Image, ImageFilter

WALKABLE = {"residential", "living_street", "pedestrian", "footway", "path", "steps", "service", "tertiary", "unclassified"}
DRIVE_SPEED = {"trunk": 11, "primary": 10, "secondary": 9, "tertiary": 8,
               "residential": 6, "unclassified": 6, "living_street": 3}
# Stagon Street, the eastern Kalampaka–Meteora road, and the branch to St. Stephen.
EAST_ACCESS_WAYS = {"1413837828", "47310211", "1174092380", "1174023045", "1174023044", "428790439"}


class SceneLife:
    def __init__(self, ways, ground, project, size):
        self.width, self.height = size
        self.edges, self.water = [], []
        self.windows = Image.new("L", size)
        vertices = {}
        walk_keys, drive_keys = set(), set()
        routes = []

        def line(a, b, spacing):
            length = np.linalg.norm(np.subtract(b, a))
            count = max(2, int(np.ceil(length / spacing)) + 1)
            points = np.linspace(a, b, count)
            heights = ground.height(*points.T) + .27
            return length, project(np.column_stack([points, heights]))

        for way_id, (tags, segments) in ways.items():
            highway = tags.get("highway")
            if tags.get("waterway") == "river":
                for points in segments:
                    for a, b in zip(points, points[1:]):
                        _, screen = line(a, b, 15)
                        self.water.extend([round(float(x), 1), round(float(y), 1)]
                                          for x,y,_ in screen if 0 < x < self.width and 0 < y < self.height)
            if (tags.get("area") == "yes"
                    or tags.get("tunnel") not in (None, "no")
                    or tags.get("access") in ("private", "no")):
                continue
            walking = highway in WALKABLE and tags.get("foot") not in ("private", "no")
            speed = (DRIVE_SPEED.get(highway)
                     if all(tags.get(mode) not in ("private", "no") for mode in ("vehicle", "motor_vehicle", "motorcar"))
                     else None)
            if not walking and speed is None:
                continue
            oneway = tags.get("oneway")
            direction = (2 if oneway == "-1" else
                         1 if oneway in ("yes", "1", "true") or (oneway is None and tags.get("junction") == "roundabout")
                         else 3)
            for points in segments:
                for a, b in zip(points, points[1:]):
                    e, n = (np.asarray(a) + b) / 2
                    kalampaka = (e/970)**2 + ((n+840)/680)**2 < 1
                    kastraki = ((e+635)/475)**2 + ((n-269)/450)**2 < 1
                    monastery_road = ((e+125)/1060)**2 + ((n-740)/1120)**2 < 1
                    walk = walking and (kalampaka or kastraki)
                    drive = speed is not None and (kalampaka or kastraki or monastery_road or way_id in EAST_ACCESS_WAYS)
                    if not (walk or drive):
                        continue
                    length, screen = line(a, b, 3.0)
                    if length < 1.2 or length > 250 or not np.isfinite(screen).all():
                        continue
                    first = vertices.setdefault(a, len(vertices))
                    last = vertices.setdefault(b, len(vertices))
                    key = (min(first, last), max(first, last))
                    walk = walk and key not in walk_keys and length <= 170
                    drive = drive and key not in drive_keys
                    if not (walk or drive):
                        continue
                    if walk:
                        walk_keys.add(key)
                    if drive:
                        drive_keys.add(key)
                    self.edges.append((first, last, float(length), len(screen), walk, direction if drive else 0, speed))
                    routes.append(screen)
        self.points = np.concatenate(routes) if routes else np.empty((0, 3))
        self.seen = np.zeros(len(self.points), bool)

    def record(self, depth, color, material, start, top, end, world):
        x, y, z = self.points.T
        xx, yy = np.floor(x).astype(int), np.floor(y).astype(int)
        active = (xx >= 0) & (xx < self.width) & (yy >= start) & (yy < end)
        self.seen[active] = z[active] >= depth[yy[active]-top, xx[active]] - 1.45

        # Only windows still visible after vegetation get lights.
        wall = (material >= 20) & (material < 30)
        e, n, h = world.transpose(2, 0, 1)
        room = (np.floor(e/3.1)*37 + np.floor(n/3.1)*17 + np.floor(h/3.1)*11) % 13
        lit = wall & np.all(color == [62, 61, 62], axis=2) & (room < 7)
        self.windows.paste(Image.fromarray(lit.astype("u1")*255), (0, start))

    def save(self, assets):
        paths, roads, cursor = [], [], 0
        for first, last, length, count, walk, direction, speed in self.edges:
            end = cursor + count
            screen = self.points[cursor:end,:2].round(1).tolist()
            visibility = "".join("1" if visible else "0" for visible in self.seen[cursor:end])
            cursor = end
            if "1" in visibility:
                segment = [first, last, round(length, 1), screen, visibility]
                if walk:
                    paths.append(segment)
                if direction:
                    roads.append([*segment, direction, speed])
        (assets / "life.json").write_text(json.dumps(
            {"paths": paths, "roads": roads, "water": self.water}, separators=(",", ":")) + "\n")
        glow = Image.new("RGBA", self.windows.size, (255, 173, 74, 0))
        halo = self.windows.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(.8))
        glow.putalpha(halo.point(lambda a: int(a*.43)))
        lights = Image.new("RGBA", self.windows.size, (255, 222, 140, 0))
        lights.putalpha(self.windows.point(lambda a: int(a*.94)))
        Image.alpha_composite(glow, lights).save(assets / "window-lights.png", optimize=True)
