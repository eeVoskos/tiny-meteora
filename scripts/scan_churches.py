"""Orthodox church geometry registered to the mapped building outlines."""

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from scan_geometry import half_plane, signed_area, triangulate

STONE, PLASTER, BRICKWORK, BELL, CROSS, TILE, DARK_TILE, DRUM, PORTICO, CLOCK = range(70, 80)
TERRACE, CLERESTORY = 80, 81
BASILICA_STONE, BELFRY_STONE, CATHEDRAL_TILE, CATHEDRAL_DRUM = range(82, 86)
APSE_STONE = 86
CHAPEL = "chapel"


@dataclass(frozen=True)
class Profile:
    eave: float
    rise: float
    wall: int = STONE
    roof: int = TILE
    form: str = CHAPEL
    hip: float = 0
    canopy: bool = False


# Observed roof forms and materials for named buildings. Other mapped chapels
# use a modest single nave. Separate footprints are kept for the Dormition tower
# and its adjoining structure.
PROFILES = {
    "609218810": Profile(5.65, 1.2, BASILICA_STONE, TILE, "dormition"),
    "1057465731": Profile(13.8, 1.2, BELFRY_STONE, DARK_TILE, "dormition-tower"),
    "1057465730": Profile(3.0, 1.1, form="annex"),
    "585513480": Profile(9.7, 2.2, BRICKWORK, CATHEDRAL_TILE, "bessarion"),
    "379736575": Profile(7.7, 3.0, PLASTER, TILE, "holy-fathers"),
    "415534160": Profile(7.0, 3.3, PLASTER, TILE, "constantine"),
    "415534162": Profile(7.6, 2.9, PLASTER, DARK_TILE, "peter-paul"),
    "1099702667": Profile(3.1, 1.45, STONE, DARK_TILE),
    "1331982115": Profile(3.0, 1.25, STONE, DARK_TILE, hip=2.7, canopy=True),
    "1331968962": Profile(3.0, 1.5),
    "379736576": Profile(3.8, 1.9, STONE, DARK_TILE, hip=2),
    "1067374522": Profile(3.4, 1.4),
    "1057465715": Profile(3.6, 1.55, PLASTER, TILE, hip=2.4, canopy=True),
    "1015988447": Profile(3.4, 1.15, STONE, DARK_TILE, hip=1.9),
    "813239171": Profile(3.8, 1.6, STONE, TILE, hip=1.9),
    "813239172": Profile(3.5, 1.5, STONE, DARK_TILE, hip=1.8),
    "586377055": Profile(3.2, 1.55, PLASTER),
    "704404787": Profile(3.0, 1.5),
    "399713406": Profile(2.9, 1.4),
}


def rectangle(west, east, south, north):
    return np.array([[west, south], [east, south], [east, north], [west, north]])


class Church:
    def __init__(self, scene, ground, outline, profile):
        self.scene = scene
        self.ground = ground
        self.profile = profile
        self.origin = outline.mean(axis=0)
        edges = np.roll(outline, -1, axis=0) - outline
        edge_lengths = np.linalg.norm(edges, axis=1)
        axis = edges[edge_lengths.argmax()] / edge_lengths.max()
        if profile.form == "holy-fathers":
            axis = np.array([cos(radians(20)), -sin(radians(20))])
        elif profile.form in ("bessarion", "dormition", "peter-paul"):
            # The long sides of these three plans run almost east–west.
            idx = (np.abs(edges[:, 0])).argmax()
            axis = edges[idx] / edge_lengths[idx]
        if axis[0] < 0:
            axis = -axis
        self.basis = np.array([axis, [-axis[1], axis[0]]])
        self.outline = (outline - self.origin) @ self.basis.T
        if signed_area(self.outline) < 0:
            self.outline = self.outline[::-1]
        self.west, self.south = self.outline.min(axis=0)
        self.east, self.north = self.outline.max(axis=0)
        self.mid = (self.north + self.south) / 2
        heights = ground.height(*outline.T)
        self.base = float(np.percentile(heights, 28))
        self.eave = max(profile.eave, float(np.percentile(heights, 82)) - self.base + .65)

    def face(self, plan, heights, material, uv=None):
        world = np.asarray(plan) @ self.basis + self.origin
        if uv is None:
            uv = plan
        self.scene.face(np.column_stack([world, np.asarray(heights) + self.base]), uv, material)

    def walls(self, poly, lower, upper, material, ground=False):
        count = len(poly)
        floor = np.full(count, lower, dtype=float)
        if ground:
            existing = self.ground.height(*(poly @ self.basis + self.origin).T) - self.base
            floor = np.minimum(lower, existing - .9)
        top = np.broadcast_to(upper, floor.shape)
        for i in range(count):
            j = (i + 1) % count
            length = np.linalg.norm(poly[j] - poly[i])
            if length < .01:
                continue
            midpoint = length / 2
            self.face(poly[[i, j, j, i]], [floor[i], floor[j], top[j], top[i]], material,
                      [(-midpoint, floor[i]-lower), (midpoint, floor[j]-lower),
                       (midpoint, top[j]-lower), (-midpoint, top[i]-lower)])

    def roof(self, poly, eave, rise, material, hip=0):
        lowest, highest = poly.min(axis=0), poly.max(axis=0)
        across = (highest[1] - lowest[1]) / 2
        if across < .05:
            return
        center = (highest[1] + lowest[1]) / 2
        slope = rise / across

        def planes(points):
            x, y = points.T
            sides = [rise - (y-center)*slope, rise + (y-center)*slope]
            if hip:
                sides += [(x-lowest[0])*rise/hip, (highest[0]-x)*rise/hip]
            return np.column_stack(sides)

        for tri in triangulate(poly):
            triangle = poly[list(tri)]
            for surface in range(4 if hip else 2):
                piece = triangle
                for other in range(4 if hip else 2):
                    if len(piece) < 3:
                        break
                    if surface != other:
                        heights = planes(piece)
                        piece = half_plane(piece, heights[:, other] - heights[:, surface])
                if len(piece) >= 3:
                    # Clip intersections can land exactly on a triangle vertex.
                    _, unique = np.unique(piece.round(6), axis=0, return_index=True)
                    piece = piece[np.sort(unique)]
                    if len(piece) >= 3 and signed_area(piece) > .00001:
                        heights = eave + planes(piece)[:, surface]
                        for i in range(1, len(piece) - 1):
                            indices = [0, i, i+1]
                            self.face(piece[indices], heights[indices], material)
        if hip:
            return
        # Close the gables under the roof across the ridge line.
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            if abs(b[1]-a[1]) < .03:
                continue
            end = [a, b]
            if (a[1]-center)*(b[1]-center) < 0:
                end.insert(1, a + (b-a)*(center-a[1])/(b[1]-a[1]))
            if len(end) == 2:
                segments = [(a, b)]
            else:
                segments = [(a, end[1]), (end[1], b)]
            for p, q in segments:
                pz, qz = eave + planes(np.asarray([p, q])).min(axis=1)
                length = np.linalg.norm(q-p)
                if max(pz, qz) < eave+.02:
                    continue
                plan = np.asarray([p, q, q, p])
                heights = [eave, eave, qz, pz]
                uv = [(-length/2,eave), (length/2,eave), (length/2,qz), (-length/2,pz)]
                # Omit the degenerate triangle where the wall meets an eave.
                ix = [0, 1, 3] if qz < eave+.001 else [0, 1, 2] if pz < eave+.001 else [0, 1, 2, 3]
                self.face(plan[ix], np.asarray(heights)[ix], self.profile.wall, np.asarray(uv)[ix])

    def box(self, x, y, width, depth, bottom, top, material):
        poly = rectangle(x-width/2, x+width/2, y-depth/2, y+depth/2)
        self.walls(poly, bottom, top, material)
        self.face(poly, [top]*4, material)

    def transverse_roof(self, poly, eave, rise):
        rotated = np.column_stack([poly[:, 1], -poly[:, 0]])
        basis = self.basis
        self.basis = np.array([basis[1], -basis[0]])
        self.roof(rotated, eave, rise, self.profile.roof)
        self.basis = basis

    def masonry_tower(self, x, y, width, eave, roof_rise):
        # Two tall arches per level and projecting stone courses between stages.
        levels = (-.75, eave*.37, eave*.71, eave)
        for lower, upper in zip(levels, levels[1:]):
            self.box(x, y, width, width, lower, upper, BELFRY_STONE)
        for level in (0, *levels[1:-1], eave-.28):
            self.box(x, y, width+.40, width+.40, level, level+.32, BASILICA_STONE)
        cap = rectangle(x-width*.56, x+width*.56, y-width*.56, y+width*.56)
        self.roof(cap, eave+.03, roof_rise, self.profile.roof, hip=width*.52)
        self.cross(x, y, eave+roof_rise-.07, 1.35)

    def cross(self, x, y, base, size):
        self.box(x, y, .45, .45, base, base+size, CROSS)
        start = base+size*.61
        self.box(x, y, size*.64, .45, start, start+.36, CROSS)

    def dome(self, x, y, radius, base, drum_height, rise, wall=DRUM):
        center = np.array([x, y])
        angles = np.linspace(0, 2*np.pi, 17)[:-1]
        unit = np.column_stack([np.cos(angles), np.sin(angles)])
        drum = center + radius*unit
        self.walls(drum, base, base+drum_height, wall)
        cornice = base + drum_height
        distances = radius*np.array([1.035, .94, .71, .39, .07])
        levels = cornice + rise*np.array([0, .36, .71, .94, 1])
        for inner, outer, low, high in zip(distances[1:], distances, levels, levels[1:]):
            foot, cap = center+outer*unit, center+inner*unit
            for i in range(len(unit)):
                j = (i+1) % len(unit)
                self.face([foot[i], foot[j], cap[j], cap[i]], [low, low, high, high], self.profile.roof)
        self.cross(x, y, levels[-1]-.1, 1.8 if radius >= 3 else 1.35)

    def belfry(self, x, y, size, deck, wall, bottom=0, rounded=False):
        self.box(x, y, size, size, bottom, deck-.35, wall)
        stone = size*.22
        open_height = size*.68
        shadow = size*.39
        for dx, dy in ((-1,-1), (-1,1), (1,-1), (1,1)):
            self.box(x+dx*shadow, y+dy*shadow, stone, stone, deck-.35, deck+open_height, wall)
        self.box(x, y, size*.30, size*.30, deck+open_height*.44, deck+open_height*.76, BELL)
        self.box(x, y, size*1.09, size*1.09, deck+open_height-.46, deck+open_height+.10, wall)
        top = deck+open_height+.10
        if rounded:
            self.dome(x, y, size*.56, top-.15, .35, size*.36, wall)
        else:
            cap = rectangle(x-size*.57, x+size*.57, y-size*.57, y+size*.57)
            self.roof(cap, top, size*.32, self.profile.roof, hip=size*.56)
            self.cross(x, y, top+size*.31, 1.55)

    def porch(self, west, east, south, north, height, bays=3, arches=True):
        width = north-south
        piers = np.linspace(south+.18, north-.18, bays+1)
        for y in piers:
            self.box(west+.25, y, .54, .50, 0, height, self.profile.wall)
        if arches:
            for a, b in zip(piers, piers[1:]):
                y = np.linspace(a, b, 7)
                radius = (b-a)/2
                fascia = height-.28
                intrados = fascia - radius*.64 + np.sqrt(np.maximum(0, radius**2-(y-(a+b)/2)**2))*.43
                for i in range(len(y)-1):
                    end, start = y[i+1], y[i]
                    self.face([[west,end],[west,start],[west,start],[west,end]],
                              [intrados[i+1],intrados[i],fascia,fascia], PORTICO)
        # A step and a deep, low roof leave the approach visibly open.
        self.box((east+west)/2, (south+north)/2, east-west+.75, width+.55, -.35, .23, STONE)
        poly = rectangle(west-.22, east+.18, south-.20, north+.20)
        self.walls(poly, height-.53, height, PORTICO)
        self.roof(poly, height, .88, self.profile.roof, hip=min(width/2, 1.3))

    def apse(self, x, y, depth, radius, height):
        angle = np.linspace(-np.pi/2, np.pi/2, 9)
        poly = np.array([x, y]) + np.column_stack([depth*np.cos(angle), radius*np.sin(angle)])
        self.walls(poly, -.55, height, APSE_STONE)
        apex = [x-.32, y]
        for a, b in zip(poly[:-1], poly[1:]):
            self.face([a, b, apex], [height, height, height+radius*.55], self.profile.roof)

    def dormition(self):
        # From the measured exterior survey. The raised nave has low pent roofs
        # to either side, two successive narthex roofs, and three conical apses.
        west, east, south, north = self.west, self.east, self.south, self.north
        nave_y = self.mid + .65
        w = north-south
        entrance = west + 11.5
        nave_end = east - 1.6
        nave_s, nave_n = nave_y-w*.24, nave_y+w*.24
        nave = rectangle(entrance-.35, nave_end, nave_s, nave_n)
        self.walls(nave, self.eave-.34, self.eave+4.13, CLERESTORY)
        self.roof(nave, self.eave+4.13, 2.55, self.profile.roof)

        aisle_eave = self.eave-.06
        aisle_upper = self.eave+1.45
        # The mapped southern wall jogs in below the east end of the nave.
        south_edge = south+2.6
        for lo, hi, south_z, north_z in (
            (south_edge, nave_s, aisle_eave, aisle_upper),
            (nave_n, north-.50, aisle_upper, aisle_eave),
        ):
            aisle = rectangle(entrance-.55, nave_end+.18, lo, hi)
            heights = np.array([south_z, south_z, north_z, north_z])
            self.walls(aisle, self.eave-.45, heights, BASILICA_STONE)
            self.face(aisle, heights, self.profile.roof)

        for x in (entrance+2.6, entrance+8.4, entrance+14.2):
            skylight_y = nave_s-1.45
            skylight = rectangle(x-.72, x+.72, skylight_y-.83, skylight_y+.83)
            self.walls(skylight, aisle_eave+.32, aisle_upper+.38, CLERESTORY)
            self.roof(skylight, aisle_upper+.38, .63, self.profile.roof)

        # The western cross roofs step up toward the nave.
        for start, end, height, rise in (
            (west-.31, west+5.25, self.eave+1.03, 1.35),
            (west+5.05, entrance+.49, self.eave+2.20, 1.77),
        ):
            narthex = rectangle(start, end, south_edge-.25, north+.19)
            self.walls(narthex, -.65, height, BASILICA_STONE, ground=True)
            self.transverse_roof(narthex, height, rise)

        for y, radius, projection, height in (
            (nave_y-w*.32, 1.8, 1.85, self.eave+.12),
            (nave_y, 3.75, 3.35, self.eave+1.14),
            (nave_y+w*.33, 1.86, 1.95, self.eave+.12),
        ):
            self.apse(east-1.65, y, projection, radius, height)
        self.cross(entrance+.22, nave_y, self.eave+6.56, 1.6)

        # The church stands on a stone terrace above the lower forecourt.
        front = south-5.2
        terrace = rectangle(west+.8, east+.95, front, south+1.35)
        self.walls(terrace, -4.8, -.30, TERRACE, ground=True)
        self.face(terrace, [-.30]*4, TERRACE)
        self.box((west+east)/2+.85, front+.18, east-west+.15, .44, -.45, .06, BASILICA_STONE)
        for i in range(6):
            self.box(east-1.7, front-(i+1)*.70, 2.35, .71, -5.25, -.78-i*.52, TERRACE)

    def bessarion(self):
        west, east, south, north = self.west, self.east, self.south, self.north
        y = self.mid
        width = north-south
        x = west + (east-west)*.65
        nave = rectangle(west+.35, east-.55, y-width*.25, y+width*.25)
        nave_eave = self.eave+3.2
        self.walls(nave, self.eave-.35, nave_eave, BRICKWORK)
        self.roof(nave, nave_eave, 3.7, self.profile.roof)
        south_arm = rectangle(x-5.65, x+5.28, south+1.75, y+.15)
        self.walls(south_arm, self.eave-.68, self.eave+2.55, BRICKWORK)
        self.transverse_roof(south_arm, self.eave+2.55, 2.95)
        self.dome(x, y, width*.208, self.eave+4.25, 4.35, 4.25, CATHEDRAL_DRUM)
        self.masonry_tower(west+5.0, y+width*.28, 5.85, self.eave+9.4, 1.35)
        self.porch(west-3.0, west+.47, y-width*.24, y+width*.24, 5.15)

    def parish(self):
        x = self.west + (self.east-self.west)*.63
        y = self.mid
        width = self.north-self.south
        span = self.east-self.west
        nave = rectangle(self.west+.4, self.east-.85, y-width*.26, y+width*.26)
        self.walls(nave, self.eave-.26, self.eave+1.9, self.profile.wall)
        self.roof(nave, self.eave+1.9, self.profile.rise, self.profile.roof)
        self.dome(x, y, width*.203, self.eave+2.25, 4.6, width*.126)
        if self.profile.form == "holy-fathers":
            for dy in (-width*.23, width*.20):
                self.belfry(self.west+span*.25, y+dy, 3.8, self.eave+4.25, PLASTER, rounded=True)
            self.porch(self.west-1.15, self.west+1.9, y-width*.21, y+width*.23, 3.35)
        else:
            for dy in (-width*.36, width*.35):
                self.belfry(self.west+span*.16, y+dy, 3.5, self.eave+5.2, PLASTER, rounded=True)
            # The long arcade faces the road along the southern wall.
            previous = self.basis
            self.basis = np.array([previous[1], -previous[0]])
            self.porch(self.south-2.0, self.south+.3, -(self.west+span*.84), -self.west, 3.25, bays=4)
            self.basis = previous

    def peter_paul(self):
        y = self.mid
        x = self.west+2.5
        width = self.north-self.south
        top = self.eave+6.6
        self.box(x, y+width*.32, 4.8, 4.65, 0, top-3.2, PLASTER)
        self.box(x, y+width*.32, 4.55, 4.40, top-3.2, top-.45, CLOCK)
        self.box(x, y+width*.32, 4.88, 4.73, top-.45, top, PLASTER)
        self.porch(self.west-2.65, self.west+.65, y-width*.42, y+width*.44, 4.4, bays=4)
        self.cross(self.west+1.2, y, self.eave+self.profile.rise-.13, 1.75)

    def draw(self):
        p = self.profile
        if p.form == "dormition-tower":
            self.masonry_tower((self.west+self.east)/2, self.mid, (self.north-self.south)*.97,
                               self.eave, p.rise)
            return
        self.walls(self.outline, -.65, self.eave, p.wall, ground=True)
        self.roof(self.outline, self.eave, p.rise, p.roof, p.hip)
        if p.form == "dormition":
            self.dormition()
        elif p.form == "bessarion":
            self.bessarion()
        elif p.form in ("holy-fathers", "constantine"):
            self.parish()
        elif p.form == "peter-paul":
            self.peter_paul()
        elif p.form == CHAPEL:
            self.cross(self.west+max(p.hip, .9), self.mid, self.eave+p.rise-.15, 1.35)
            if p.canopy:
                self.porch(self.west-1.9, self.west+.45, self.mid-1.6, self.mid+1.6, self.eave-.8, bays=1, arches=False)


def draw_church(way_id, tags, outline, scene, ground):
    profile = PROFILES.get(way_id)
    if profile is None:
        worship = tags.get("amenity") == "place_of_worship" and tags.get("denomination") != "jehovahs_witness"
        if tags.get("building") not in ("church", "chapel") and not worship:
            return False
        profile = Profile(3.25, 1.45, hip=1.2)
    _, unique = np.unique(outline.round(3), axis=0, return_index=True)
    outline = outline[np.sort(unique)]
    if len(outline) < 3 or abs(signed_area(outline)) < 7:
        return False
    if not np.isfinite(ground.height(*outline.T)).all():
        return False
    Church(scene, ground, outline, profile).draw()
    return True


def shade_churches(image, materials, u, v, direct, sun, roof_light, shades, palette):
    where = (materials >= STONE) & (materials <= APSE_STONE)
    if not where.any():
        return
    mat = materials[where]
    x, y = u[where], v[where]
    sun, direct = sun[where], direct[where]
    light = .30+.38*direct+.24*sun
    pixel = np.empty((len(mat), 3), dtype=np.uint8)
    course = np.floor(y/.84)
    bond = (x + course*.71) % 1.48
    masonry = .07*((y % .84 < .151) | (bond < .17))
    grain = ((np.floor(x*1.34)*7 + course*17) % 7 - 3)*.018
    walls = {
        STONE: "7a6970 918073 ab937e c0a58b d4b89d e5c9a9 f0d9bb",
        PLASTER: "817679 9e9086 b7a598 d0bfb0 e2d4c0 f0e3cc fff4df",
        BRICKWORK: "505052 676661 807e74 969387 b1ad9c c8c1ac e1d8be",
        DRUM: "877472 9e897b bca18c d1b59c e5c9ad f5dec0",
        PORTICO: "816d6c aa8f80 c7ad9d e2cbbb f1dcc1 fbf0d6",
        CLOCK: "817679 9e9086 b7a598 d0bfb0 e2d4c0 f0e3cc fff4df",
        TERRACE: "686560 837c70 a19988 b9b1a0 d1c8b5 e8dfcb",
        CLERESTORY: "6f6664 8e7f72 aa9986 c5b29a e1ccb0 f6dfc0",
        BASILICA_STONE: "736d67 918877 a99e89 c4b69d dcd0b5 e8dcc4 f7ecd4",
        BELFRY_STONE: "69655f 817970 9a907d b6a991 cbbda1 e0d3b6 eddec2",
        CATHEDRAL_DRUM: "64605c 807974 9d9389 bfb2a4 d4c2ac f0dec2",
        APSE_STONE: "6b625b 8a7e6e a2947f bdae96 d8c6ac e7d7be",
    }
    masonry_walls = (STONE, BRICKWORK, TERRACE, CLERESTORY, BASILICA_STONE, BELFRY_STONE, APSE_STONE)
    for material, colors in walls.items():
        use = mat == material
        texture = grain[use] - masonry[use] if material in masonry_walls else grain[use]*.28
        pixel[use] = shades(palette(colors), light[use]+texture)
    # The tall openings use rounded heads with terracotta surrounds and stone sills.
    drum = (mat == DRUM) | (mat == CATHEDRAL_DRUM)
    arched = np.isin(mat, (STONE, PLASTER, BRICKWORK, DRUM, CATHEDRAL_DRUM))
    tall = mat == BRICKWORK
    bay = np.abs(np.where(tall, (x+2.0) % 4.0-2.0, (x+1.92) % 3.84-1.92))
    bay = np.where(drum, np.abs(x), bay)
    # Small chapels have high, narrow slits. The cathedral gets a taller arcade.
    sill = np.where(tall, 2.25, 1.45)
    shoulder = np.where(tall, 6.35, np.where(drum, 2.65, 2.50))
    radius = np.where(tall, .77, .48)
    arch = np.hypot(bay, np.maximum(y-shoulder, 0))
    opening = arched & (y > sill) & (arch < radius)
    surround = arched & ~opening & (y > sill-.36) & (arch < radius+.34)
    pixel[surround] = shades(palette("694b49 8c5a4c a96c57 c48b6e d7a181"), light[surround])
    pixel[opening] = shades(palette("302e36 413a3b 544344 665251"), light[opening]-.05)
    clerestory = mat == CLERESTORY
    slit = clerestory & (np.hypot(np.abs((x+2.28) % 4.56-2.28), np.maximum(y-2.75, 0)) < .62) & (y > 1.50)
    pixel[slit] = [78, 59, 54]
    pixel[clerestory & (y > 4.03) & (y < 4.47)] = [147, 86, 62]
    basilica_arch = (mat == BASILICA_STONE) & (np.hypot((x+3.5) % 7-3.5, np.maximum(y-2.72,0)) < .62) & (y > 1.01)
    pixel[basilica_arch] = [110, 80, 66]
    belfry = mat == BELFRY_STONE
    arch_center = np.abs(x)-1.12
    inner = np.hypot(arch_center, np.maximum(y-3.28, 0))
    jamb = belfry & (inner < .90) & (y > .68)
    opening = belfry & (inner < .64) & (y > .94)
    pixel[jamb] = shades(palette("5e514a 88766a a08a73 d2b899"), light[jamb])
    pixel[opening] = shades(palette("302e31 473b35 625145"), light[opening]-.13)
    cornice = (mat == CATHEDRAL_DRUM) & (y > 3.62) & (y < 4.20)
    pixel[cornice] = [160, 90, 66]
    clock = mat == CLOCK
    dial = clock & (np.hypot(x, y-1.69) < 1.07)
    pixel[dial] = [70, 64, 60]
    pixel[clock & (np.hypot(x,y-1.69) < .78)] = [238, 226, 206]
    roof_light = roof_light[where]
    for material, colors in (
        (TILE, "5d3e43 77474a 934f46 aa5b46 c16c50 d58059 e69868 f2ac78"),
        (DARK_TILE, "594146 6a4a4c 81574e 996452 ae785e c18766 d39b75 e3ae85"),
        (CATHEDRAL_TILE, "392c30 523138 713b37 91483b ad5a45 c77455 db8c67 e5a27c"),
        (BELL, "4e4142 675446 8e714d af915e"),
        (CROSS, "4a4546 645750 95816a cbb08a"),
    ):
        use = mat == material
        brightness = roof_light[use] if material in (TILE, DARK_TILE, CATHEDRAL_TILE) else light[use]
        pixel[use] = shades(palette(colors), brightness)
    image[where] = pixel
