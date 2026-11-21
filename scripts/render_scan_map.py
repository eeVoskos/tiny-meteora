"""Render the aligned 2016 scan and geographic overlays as a static tile pyramid."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
import json
import math
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from roof_survey import SURVEY_ROOF_BASE, RoofSurvey, draw_observed_roof, shade_survey_roofs
from scan_churches import PROFILES, draw_church, shade_churches
from scan_features import MONASTERIES, civic_structures, masks_for, read_osm
from scan_geometry import (ElevationTiles, Ground, ROOT, Raster, local, signed_area,
                           texture_colors, triangulate, unpack_scan)
from scan_life import SceneLife

ORIGINAL_WIDTH = 4608
W, H = ORIGINAL_WIDTH + 512, 3584
METERS_PER_PIXEL = .86
CENTER = np.array([-235, -75, -268])
TURN, TILT = math.radians(13), math.radians(38)
BASIS = np.column_stack((
    [math.cos(TURN), math.sin(TURN), 0],
    [math.sin(TURN)*math.sin(TILT), -math.cos(TURN)*math.sin(TILT), -math.cos(TILT)],
    [math.sin(TURN)*math.cos(TILT), -math.cos(TURN)*math.cos(TILT), math.sin(TILT)],
))
SUN = np.array([-.62, -.39, .68])
SUN /= np.linalg.norm(SUN)
RNG = np.random.default_rng(350)
EAST_RNG = np.random.default_rng(351)
TILE_SIZE, STRIP_HEIGHT = 512, 384


def project(points):
    screen=(points-CENTER)@BASIS
    screen[:,:2]/=METERS_PER_PIXEL
    # Extend the right edge while keeping the original pixel coordinates in shared views.
    screen[:,:2]+=(ORIGINAL_WIDTH/2,H/2)
    return screen

class Solids:
    def __init__(self):
        self.positions=[]
        self.normals=[]
        self.uvs=[]
        self.faces=[]
        self.materials=[]

    def face(self, coords, uvs, material):
        coords=np.array(coords)
        normal=np.cross(coords[1]-coords[0],coords[2]-coords[0])
        length=np.linalg.norm(normal)
        if length<.00001:
            return
        start=len(self.positions)
        self.positions.extend(coords)
        self.normals.extend([normal/length]*len(coords))
        self.uvs.extend(uvs)
        for i in range(1,len(coords)-1):
            self.faces.append((start,start+i,start+i+1))
            self.materials.append(material)

    def arrays(self):
        return tuple(np.asarray(x) for x in (self.positions,self.normals,self.uvs,self.faces,self.materials))

def house(tags, poly, scene, ground, on_cliff, observed=None):
    # Remove repeated points while retaining the order of the mapped outline.
    unique = np.unique(np.round(poly, 2), axis=0, return_index=True)[1]
    poly = poly[np.sort(unique)]
    if len(poly) < 3:
        return
    area=signed_area(poly)
    if abs(area)<10 or abs(area)>6500:
        return
    if area < 0:
        poly=poly[::-1]

    center=np.mean(poly,axis=0)
    samples=np.concatenate([poly,center[None],(poly+center)/2])
    heights=ground.height(*samples.T)
    if not np.isfinite(heights).all() or (np.ptp(heights)>34 and not on_cliff):
        return
    datum=float(np.percentile(heights,25))
    guess=int(tags.get('building:levels','0')) if tags.get('building:levels','').isdigit() else 0
    church=(tags.get('building') in ('church','chapel','monastery') or
            tags.get('amenity')=='place_of_worship')
    if church:
        height=7.6
    elif guess:
        height=np.clip(3*guess,3,14)
    else:
        height=RNG.choice([3.3,4.2,4.9,5.7,6.3,7.2],p=[.11,.12,.24,.26,.19,.08])
    eave=max(datum+height,float(np.percentile(heights,85))+.55)
    if on_cliff:
        eave=max(eave,float(np.percentile(heights,80))+2.7)
    bounds=project(np.column_stack([poly,np.full(len(poly),eave)]))
    if (bounds[:,0].max()<-30 or bounds[:,0].min()>W+30 or
            bounds[:,1].max()<-30 or bounds[:,1].min()>H+55):
        return
    base=np.column_stack([poly,np.clip(heights[:len(poly)]-1.2,datum-11,eave-2.0)])
    plate=np.column_stack([poly,np.full(len(poly),eave)])
    colors_rng = RNG if bounds[:,0].min() <= ORIGINAL_WIDTH+30 else EAST_RNG
    family=colors_rng.choice([0,1,2,3,4,5,6],p=[.20,.18,.16,.16,.15,.09,.06])
    roof_sample=ground.color[ground.pixel(samples[:,0],samples[:,1])[::-1]].astype(float)
    tints=np.median(roof_sample,axis=0)
    has_scan=ground.scan_known[ground.pixel(np.asarray(center[0]),np.asarray(center[1]))[::-1]]
    if church or on_cliff:
        family=2
    elif tags.get('leisure')=='sports_centre' or tags.get('roof:material') in ('metal','concrete'):
        family=5
    elif has_scan and tints[0]-tints[2]<9 and tints.mean()>139:
        family=colors_rng.choice((4,5,6))
    elif has_scan and tints[0]-tints[2]>32:
        family=colors_rng.choice((0,1,2,3))
    for i in range(len(poly)):
        j=(i+1)%len(poly)
        length=np.linalg.norm(poly[j]-poly[i])
        scene.face([base[i],base[j],plate[j],plate[i]],
                   [(0,base[i,2]-eave),(length,base[j,2]-eave),(length,0),(0,0)],20+family)
    roof_material=30+family
    if observed is not None:
        roof_material = SURVEY_ROOF_BASE + observed.color
        if draw_observed_roof(scene, observed, poly, eave, 20+family, roof_material):
            return observed.shape
    edges=np.roll(poly,-1,axis=0)-poly
    long=edges[np.linalg.norm(edges,axis=1).argmax()]
    long=long/np.linalg.norm(long)
    wide=np.array([-long[1],long[0]])
    along=(poly-center)@long
    across=(poly-center)@wide
    roof_height=np.clip(np.ptp(across)*.24,1.3,4.4)
    if len(poly)==4:
        half_ridge=max(.0,(np.ptp(along)-np.ptp(across)*.73)/2)
        ridge=np.column_stack([
            center+np.array([[-half_ridge],[half_ridge]])*long,
            [eave+roof_height,eave+roof_height],
        ])
        groups=(along>=0).astype(int)
        if groups.sum()==2:
            for i in range(4):
                j=(i+1)%4
                roof=[plate[i],plate[j],ridge[groups[j]]]
                if groups[i]!=groups[j]:
                    roof.append(ridge[groups[i]])
                uv=[((p[:2]-center)@long,(p[:2]-center)@wide) for p in roof]
                scene.face(roof,uv,roof_material)
            return "color only" if observed is not None else None
    for tri in triangulate(poly):
        uv=[(along[i],across[i]) for i in tri]
        scene.face(plate[list(tri)],uv,roof_material)
    return "color only" if observed is not None else None

def cast_shadows(raster, positions, faces):
    right=np.array([-SUN[1], SUN[0], 0])
    right/=np.linalg.norm(right)
    axes=np.column_stack([right,np.cross(SUN,right),SUN])
    lo,hi=np.array([-4400,-4400]),np.array([4400,4400])
    step=2.3
    size=tuple(np.ceil((hi-lo)/step).astype(int))

    def space(points):
        light=points@axes
        light[..., :2]=(light[..., :2]-lo)/step
        return light

    buffer=raster.draw(size, space(positions), faces)

    def sunlight(world):
        projected=space(world)
        x=np.clip(projected[..., 0].astype(int),0,size[0]-1)
        y=np.clip(projected[..., 1].astype(int),0,size[1]-1)
        result=np.ones(world.shape[:2],np.float32)
        for dx,dy in ((0,0),(1,0),(0,1),(-1,0),(0,-1)):
            shadow=buffer[(y+dy).clip(0,size[1]-1),(x+dx).clip(0,size[0]-1)]
            result-=(shadow>projected[...,2]+2.8)*.20
        return result

    return sunlight


def palette(colors):
    return np.asarray([tuple(int(h[i:i+2],16) for i in (0,2,4)) for h in colors.split()],dtype=np.uint8)

ROCK=palette('292c36 3f414d 515362 626472 727480 83838b 939398 a3a3a6 b4b3b2 c5c3bf d1d0cb dedbd4 ebe7dd')
WATER=palette('203c4a 315361 3c6470 487683 589198 69a5a8 83b9b7 a1cec8 c3dfd1 d6e8d8')
VEG=palette('2c3936 39493c 48583f 546648 60754b 6f8253 7d8e55 8d9a5c 9ba65f adb271 bdc083')
GRASS=palette('464737 575741 68684a 77774e 878354 97935e a5a165 b6ae74 c7bb82 d2c497 e0cf9e')
YARD=palette('494742 615747 76634f 8d785e a18a6c ad9677 bba583 cab38d d8c199 e2caa3 edd3ae')
ROAD=palette('47484a 555455 666162 77726f 8b8178 9d8f80 aaa18c b7aa95 c6b9a4 d3c6b0 dfcfb5')
STREET=palette('8c8981 a5a197 bcb7ab cdc9bd ddd9cc e9e5d8 f3efe3 faf6ea')
COURTYARD=palette('827b72 9b9183 b1a795 c6bba8 d4c9b6 e1d7c4')
WALL=palette('53494d 685754 7d655b 987a68 aa8974 bf9e85 d1b39a e1c4ad ebd3b2 f1dcb8')
ROOFS=np.array([palette(s) for s in (
    '623b3e 7d4b44 965443 aa604b b96e50 cf7e54 dd8e5c eca56c f6b678',
    '6a473f 81513f 955c44 a86748 be774f d18a58 e39a62 ebae72 f8c584',
    '693a3c 804240 934843 ae5345 c36449 cf734d e38256 ea9460 f4a26a',
    '5d403a 765040 8b5c49 a56e56 bc7f5f cd8f68 dc9b71 e8aa7c f3bc8c',
    '624546 795249 8f5d4f a46c54 b7785f c48465 d48e6f de9c7a e9b38b',
    '49484d 5f5b61 706e72 828082 8c8c86 9d9b90 ada99a bbb5a4 cfc5b2',
    '51464a 675c5b 7c6b60 8c7e6d a29179 b29f88 bfae94 cfbd9f d8c8ac',
)])
def shades(colors,brightness):
    return colors[np.clip(np.rint(brightness*(len(colors)-1)),0,len(colors)-1).astype(int)]

def terrain_colors(src,norm,world,roads,land,sun,ground):
    e,n,z=world.transpose(2,0,1)
    up=np.clip(norm[:,:,2],0,1)
    near_sun=np.clip(norm@SUN,0,1)
    r,g,b=src.astype('f').transpose(2,0,1)/255
    lum=.22*r+.65*g+.13*b
    green=g-(r+b)/2
    x,y=ground.pixel(e,n)
    cover=land[y,x]
    rockmask=cover==1
    forest=cover==2
    steep=up<.62
    exposed=(steep&(green<.082)) | ((green<.038)&(lum>.29)&(up<.93)) | (rockmask&(up<.89)&(green<.11))
    urban=(n<-438+e*.16) & (e>-760)&(e<720) & (z<-342)
    exposed&=(~urban | (up<.39)) & ((ground.coverage[y,x]>.86) | rockmask)
    leaves=((green>.022) & ((ground.coverage[y,x]>.52) | (ground.woodland[y,x]>.49))) | forest
    leaves&=~exposed
    tint_grid=ground.variation[y,x]*.20
    # Bedding bands are tied to absolute elevation and projected on the real faces.
    strata=np.sin(z*.245+e*.023+n*.014)*.018+np.sin(z*.461+e*.071)*.011
    br=.23+near_sun*.43+sun*.16+(lum-.42)*.36+tint_grid+strata
    cast_baked=np.clip((b-g)*7,0,.25)*np.clip((.39-lum)*5,0,1)
    br+=np.clip(norm[:,:,0]*-.14,0,.13)-cast_baked
    rocky=shades(ROCK,br)
    shrub=shades(VEG,.16+near_sun*.23+.17*sun+(lum-.37)*.94+green*.45+tint_grid)
    meadow=shades(GRASS,.28+near_sun*.17+sun*.13+(lum-.32)*.75+green*.25+tint_grid)
    yard=shades(YARD,.28+.15*near_sun+.16*sun+(lum-.43)*.35+tint_grid)
    # Use inferred gardens in town while assigning paved yards a warm neutral.
    base=np.where(exposed[:,:,None],rocky,np.where(leaves[:,:,None],shrub,np.where(urban[:,:,None],yard,meadow)))
    farmland=cover==3
    stripe=(np.sin(e*.30+n*.16)+np.cos(n*.185))*0.054
    base[farmland]=shades(GRASS,.50+sun*.17+tint_grid+stripe)[farmland]
    stone=cover==4
    court=cover==5
    stone_tiles=(((e*.53+n*.36)%1)<.12)|(((n*.48-e*.25)%1)<.11)
    base[stone]=shades(YARD,.61+.20*sun-.066*stone_tiles)[stone]
    forecourt=cover==8
    base[forecourt]=shades(COURTYARD,.58+.14*sun-.093*stone_tiles)[forecourt]
    base[court]=shades(VEG,.55+.25*sun+tint_grid)[court]
    water=cover==7
    shimmer=((np.floor(e*1.4)+np.floor(n*2.1)*31)%17==0)
    base[water]=shades(WATER,.37+.26*sun+.17*shimmer)[water]
    street_mask=roads[y,x]
    street=(street_mask==1)&(up>.35)
    curb=(street_mask==2)&(up>.35)
    foot=(street_mask==3)&(up>.31)&~exposed
    paving=shades(STREET,.55+.12*near_sun+.23*sun+(lum-.40)*.09+tint_grid*.32)
    base[street]=paving[street]
    base[curb]=shades(STREET,.26+.15*sun)[curb]
    base[foot]=shades(YARD,.53+.18*sun)[foot]
    rail=street_mask==4
    ballast=(np.floor(e*.6+n*.8)%3)==0
    base[rail]=shades(ROAD,.22+.22*sun+.11*ballast)[rail]
    return base, exposed

def architecture(im,fr,mat,norm,sun):
    u,v=fr[:,:,6],fr[:,:,7]
    direct=np.clip(norm@SUN,0,1)
    tile_phase=(v+np.floor(u/.66)*.08)%1.27
    tile_grid=(tile_phase<.20)*.13+(((u+np.floor(v/1.27)*.32)%1.11)<.14)*.06
    rainbow=((np.floor(u*1.27)*11+np.floor(v*1.32)*17+mat)%11-5)*.009
    brightness=.31+direct*.38+sun*.18-tile_grid+rainbow
    for family in range(len(ROOFS)):
        walls=(mat==20+family)
        roofs=(mat==30+family)
        # Windows repeat along each face, with a one-pixel sill under taller openings.
        height=-v
        bay=(u+.8)%3.1
        story=(height-1.1)%3.05
        window=(bay>1.30)&(bay<2.34)&(story>.68)&(story<2.02)&(height>.65)
        variation=(np.floor(u/3.1)*31+np.floor(height/3.05)*17+family*7)%11
        window&=variation>1
        sill=(bay>1.17)&(bay<2.5)&(story>.54)&(story<=.68)&(height>.65)
        top_band=(height<.52)
        wall_color=shades(WALL,.48+.25*direct+.17*sun+((family%3)-1)*.041-0.18*top_band)
        wall_color[window]=[62,61,62]
        wall_color[sill]=[177,144,117]
        im[walls]=wall_color[walls]
        roof_color=shades(ROOFS[family],brightness)
        im[roofs]=roof_color[roofs]
    shade_churches(im, mat, u, v, direct, sun, brightness, shades, palette)
    shade_survey_roofs(im, mat, brightness)
    border=np.zeros(mat.shape,bool)
    building=((mat>=20)&(mat<40))|(mat>=70)
    for axis in (0,1):
        neighbor=np.roll(mat,1,axis=axis)
        border|=building&(neighbor<=2)
    im[border]=np.round(im[border]*.72).astype('u1')
    return im

def cliff_edges(color,depth,exposed,kind):
    # Mark the front lip only where the depth buffer reveals a real occlusion.
    edges=np.zeros_like(exposed)
    for axis in (0,1):
        adjacent = [slice(None), slice(None)]
        shifted = adjacent.copy()
        adjacent[axis], shifted[axis] = slice(1,None), slice(None,-1)
        a, b = tuple(adjacent), tuple(shifted)
        gap = depth[a] - depth[b]
        threshold = 7 if axis == 0 else 11
        edges[a] |= gap > threshold
        edges[b] |= gap < -threshold
    edge=edges&exposed&(kind==1)
    color[edge]=(color[edge]*.72).astype('u1')

TREE_PALETTES = [
    palette('344138 46533b 525f40 61704a 718052 819454 97a062'),
    palette('354338 49583b 586940 687a47 7d8b4d 8f9d59 aab263'),
    palette('35433c 495b45 596f4d 687e55 7d8d60 8f9e6a a4ab72'),
    palette('3b483c 4e6044 657648 788751 879656 97a567 aeb875'),
]

def canopy(width, height, seed, shrub=False, slender=False):
    margin=2
    y,x=np.indices((height+2*margin,width+2*margin))
    px=(x-width/2-margin)/max(1,width/2)
    mid=margin+(height-1)/2
    py=(y-mid)/max(1,height/2)
    rand=np.random.default_rng(seed)
    bumpy=(np.cos(px*17+py*7)*np.sin(py*12)*.11+np.cos(py*9-px*8)*.08)
    shape=px**2+py**2+bumpy
    if slender:
        shape=px**2*(1+.9*np.clip(-py,0,1))+py**2*.8+bumpy
    foliage=(shape<1)&(y<height+margin-1)
    bevel=0.76-shape*.28-px*.28-py*.22
    clusters=np.sin(px*10+np.sin(py*8)*1.3)*np.cos(py*12-px*6)*.14
    small=rand.uniform(-.075,.075,(height+2*margin,width+2*margin))
    shade=np.clip(np.rint((bevel+clusters+small)*6),1,6).astype('u1')
    shade[(shape>.80)|(py>.72)]=1
    shade[~foliage]=0
    trunk=(np.abs(x-(width//2+margin))<=0)&(y>=height+margin-2)&(y<height+2*margin)
    if shrub:
        trunk[:]=False
    return shade,trunk

def plants(ground, masks):
    rng=np.random.default_rng(713)
    land,roads,_,occupied=masks
    blocked=np.asarray(Image.fromarray(occupied).filter(ImageFilter.MaxFilter(5)))>0
    columns,rows=ground.size
    # Work from one geographic grid so that plants and their shadows agree across strips.
    xx,yy=np.meshgrid(np.arange(3,columns-3,2),np.arange(3,rows-3,2))
    xx=(xx+rng.integers(-1,2,xx.shape)).ravel()
    yy=(yy+rng.integers(-1,2,yy.shape)).ravel()
    east=ground.left+(xx+.5)*ground.step
    north=ground.top-(yy+.5)*ground.step
    p=project(np.column_stack([east,north,ground.heights[yy,xx]]))
    valid=((p[:,0]>-20)&(p[:,0]<W+20)&(p[:,1]>-20)&(p[:,1]<H+20)
           &(roads[yy,xx]==0)&~blocked[yy,xx])
    west = p[:,0] < ORIGINAL_WIDTH+20
    # Append the eastern trees without reseeding the established town and cliff scenery.
    entries = plant_inventory(ground, land, rng, xx, yy, east, north, p, valid & west)
    entries.extend(plant_inventory(ground, land, rng, xx, yy, east, north, p, valid & ~west))
    return entries


def plant_inventory(ground, land, rng, xx, yy, east, north, p, valid):
    xx,yy,east,north,p=(value[valid] for value in (xx,yy,east,north,p))
    cover=land[yy,xx]
    allowed=np.isin(cover,[0,1,2])
    r,g,b=ground.color[yy,xx].astype(float).T/255
    green=g-.52*r-.48*b
    slope=np.hypot(ground.heights[yy,xx+2]-ground.heights[yy,xx-2],
                   ground.heights[yy+2,xx]-ground.heights[yy-2,xx])/(4*ground.step)
    urban=(north<-437+east*.17)&(east>-800)
    density=np.clip(.40+green*6+(cover==2)*.25-(cover==1)*.54-slope*.27,0,.86)
    density[urban]*=.33
    apron_light=ground.woodland[yy,xx]
    apron_density=.012+.50*np.clip((apron_light-.32)/.65,0,1)**1.4
    apron_density[cover==2]=.48
    blend=Ground.blend(ground.coverage[yy,xx])
    density=apron_density+(density-apron_density)*blend
    scanned_canopy=(green>.007)&(g>r*.83)
    take=(allowed&((blend<.75)|scanned_canopy)&(slope<1.08)&(rng.random(len(p))<density))
    p,urban=p[take],urban[take]
    entries=[]
    for i in np.argsort(p[:,2]):
        shrub=rng.random()<(.32 if urban[i] else .26)
        slender=rng.random()<(.076 if urban[i] else .013) and not shrub
        if shrub:
            width,height=int(rng.integers(3,8)),int(rng.integers(3,7))
        elif slender:
            width,height=int(rng.integers(3,7)),int(rng.integers(11,18))
        else:
            width=int(rng.integers(5,12))
            height=width+int(rng.integers(-2,4))
        shape=(width,height,int(rng.integers(0,5)),shrub,slender)
        palette_id=int(rng.integers(0,len(TREE_PALETTES)))
        entries.append((*p[i],shape,palette_id))
    return entries


def vegetation(color,depth,sun,plants,start):
    sprites={}
    height=color.shape[0]
    for x,world_y,base_depth,shape,palette_id in plants:
        y=world_y-start
        if y < -1 or y > height+24:
            continue
        if shape not in sprites:
            sprites[shape]=canopy(*shape)
        pattern,bark=sprites[shape]
        rows,columns=pattern.shape
        left,top=round(x-columns/2),round(y-rows+1)
        x0,y0=max(0,left),max(0,top)
        x1,y1=min(W,left+columns),min(height,top+rows)
        if x0>=x1 or y0>=y1:
            continue
        roi=(slice(y0,y1),slice(x0,x1))
        crop=(slice(y0-top,y1-top),slice(x0-left,x1-left))
        shade,face=pattern[crop],bark[crop]
        top_depth=base_depth+np.maximum(y-np.arange(y0,y1),.3)[:,None]*METERS_PER_PIXEL*.74+1.
        dst=depth[roi]
        visible=((shade>0)|face)&(top_depth>dst)
        if not visible.any():
            continue
        shadow=sun[min(max(int(y),0),height-1),min(max(int(x),0),W-1)]<.45
        brightness=np.maximum(shade.astype(int)-(2 if shadow else 0),1)
        colors=TREE_PALETTES[palette_id][brightness]
        colors[face]=[100,84,59]
        dst[visible]=np.broadcast_to(top_depth,dst.shape)[visible]
        color[roi][visible]=colors[visible]


def render(raster, ground, texture, geometry, masks, buildings, pitches, nodes, roofs, life):
    pos, norm, faces, uv, terrain_ids = geometry
    scenery = Solids()
    count, churches = 0, 0
    applied = {}
    for way_id, tags, polygon, on_cliff in buildings:
        previous = len(scenery.faces)
        if not on_cliff and draw_church(way_id, tags, polygon, scenery, ground):
            churches += 1
        else:
            updated = house(tags, polygon, scenery, ground, on_cliff, roofs.get(way_id))
            if updated:
                applied[updated] = applied.get(updated, 0) + 1
        count += len(scenery.faces) > previous
    civic_structures(scenery, ground, nodes, pitches)
    structures, normals, surface_uv, indices, ids = scenery.arrays()
    positions = np.concatenate([pos, structures])
    vertices = np.column_stack([positions, np.concatenate([norm, normals]), np.concatenate([uv, surface_uv])])
    faces = np.concatenate([faces, indices + len(pos)])
    materials = np.concatenate([terrain_ids, ids])
    screen = project(positions)
    sunlight = cast_shadows(raster, positions, faces)
    inventory = plants(ground, masks)
    land, roads, trail, _ = masks
    base = Image.new("RGB", (W, H))
    paths = Image.new("L", (W, H))
    print(f"Rendering {count} building footprints ({churches} church structures) and {len(inventory)} trees and shrubs", flush=True)
    if applied:
        print("Survey roofs drawn: " + ", ".join(f"{total} {kind}" for kind,total in sorted(applied.items())), flush=True)

    for y in range(0, H, STRIP_HEIGHT):
        end = min(y + STRIP_HEIGHT, H)
        top, bottom = max(y - 24, 0), min(end + 24, H)
        strip = screen.copy()
        strip[:, 1] -= top
        depth, fragments, kind = raster.draw((W, bottom - top), strip, faces, vertices, materials)
        world = fragments[:, :, :3]
        normal = fragments[:, :, 3:6]
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), .0001)
        xx, yy = ground.pixel(world[:, :, 0], world[:, :, 1])
        blend = Ground.blend(ground.coverage[yy, xx]) * (kind == 1)
        aerial = texture_colors(texture, fragments[:, :, 6:])
        growth=ground.woodland[yy,xx]
        hue=ground.variation[yy,xx]
        grass=np.stack([140 + hue*30-growth*41, 145+hue*28-growth*24, 98+hue*19-growth*18],axis=-1)
        src=np.clip(np.round(grass+(aerial.astype("f")-grass)*blend[:,:,None]),0,255).astype("u1")
        sun = sunlight(world)
        color, exposed = terrain_colors(src, normal, world, roads, land, sun, ground)
        terrain = (kind == 1) | (kind == 2)
        for pitch in pitches:
            pitch.draw(color, world, terrain, sun, shades, palette)
        cliff_edges(color, depth, exposed, kind)
        architecture(color, fragments, kind, normal, sun)
        for key, shade in ((60, [192,170,142]), (61, [94,163,169]), (62, [161,211,205]), (63, [89,76,70])):
            where = kind == key
            color[where] = np.around(np.asarray(shade) * (.81 + sun[where,None] * .19)).astype("u1")
        color[kind == 0] = [116,132,85]
        vegetation(color, depth, sun, inventory, top)
        region = slice(y - top, end - top)
        life.record(depth, color[region], kind[region], y, top, end, world[region])
        base.paste(Image.fromarray(color[region]), (0, y))
        route = np.where(terrain & (normal[:, :, 2] > .22) & (roads[yy,xx] != 1), trail[yy,xx], 0)
        paths.paste(Image.fromarray(route[region].astype("u1")), (0, y))
        print(f"Rendered rows {y}–{end}", flush=True)
    return base, paths


def save_map(assets, artwork, paths, ground, places):
    assets.mkdir(parents=True, exist_ok=True)
    levels = [1, 2, 4]
    for factor in levels:
        level = artwork if factor == 1 else artwork.resize((W//factor, H//factor), Image.Resampling.BOX)
        folder = assets / "scan-tiles" / str(factor)
        folder.mkdir(parents=True, exist_ok=True)
        for y in range(0, level.height, TILE_SIZE):
            for x in range(0, level.width, TILE_SIZE):
                image = level.crop((x,y,min(x+TILE_SIZE,level.width),min(y+TILE_SIZE,level.height)))
                image.save(folder / f"{x//TILE_SIZE}-{y//TILE_SIZE}.webp", lossless=True)
        if factor == 4:
            level.save(assets / "meteora.webp", quality=91, method=6)
    artwork.resize((384, round(384 * H / W)), Image.Resampling.BOX).save(assets / "overview.webp", quality=86)
    artwork.resize((1440, round(1440 * H / W)), Image.Resampling.BOX).save(assets / "share.jpg", quality=89)
    edge = paths.filter(ImageFilter.MaxFilter(5)).point(lambda value: value * .78)
    outline = Image.new("RGBA", paths.size, (49, 45, 39, 0))
    outline.putalpha(edge)
    route = Image.new("RGBA", paths.size, (255, 215, 118, 0))
    route.putalpha(paths)
    Image.alpha_composite(outline, route).save(assets / "trails.png", optimize=True)

    for place in places:
        e,n = local(place["lat"],place["lon"])
        if place["id"] in MONASTERIES:
            offset, _ = MONASTERIES[place["id"]]
            e += offset[0]
            n += offset[1]
        height = float(ground.height(np.asarray(e),np.asarray(n)))
        if place["type"] == "monastery":
            height += 4
        else:
            height += place.get("marker_height", 0)
        x,y,_ = project(np.array([[e,n,height]]))[0]
        if not (0 < x < W and 0 < y < H):
            raise ValueError(f"Landmark outside the rendered map: {place['id']} ({x:.0f}, {y:.0f})")
        place["x"], place["y"] = round(x, 1), round(y, 1)
    config = {
        "version": "scan-2016",
        "width": W, "height": H, "metersPerPixel": METERS_PER_PIXEL,
        "northAngle": math.degrees(math.atan2(math.sin(TURN), math.cos(TURN)*math.sin(TILT))),
        "tileSize": TILE_SIZE, "levels": levels, "places": places,
    }
    (assets / "map.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def save_roof_preview(folder):
    template = ROOT / "dist"
    for name in ("atlas.js", "atlas.css", "i18n.js", "scene-life.js", "assets/favicon.png", "assets/atlas-cover.webp"):
        copy2(template / name, folder / name)
    index = (template / "index.html").read_text()
    index = index.replace('<title data-i18n="pageTitle">Tiny Meteora · Kalampaka, Greece</title>',
                          '<title data-i18n="previewTitle">Tiny Meteora · Roof survey preview</title>')
    index = index.replace('<span data-i18n="brandLocation">Kalampaka, Greece</span>',
                          '<span data-i18n="previewLocation">Local roof survey preview</span>')
    (folder / "index.html").write_text(index)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True, help="Original 2016 Meteora GLB from Zenodo")
    parser.add_argument("--roof-survey", type=Path, help="Local directory of aligned RGB, DSM and mask GeoTIFF triplets")
    parser.add_argument("--output", type=Path, help="Override the output asset directory")
    args = parser.parse_args()
    output = args.output or ROOT / ("prototype/solar-roof-preview/assets" if args.roof_survey else "dist/assets")
    with tempfile.TemporaryDirectory(prefix="meteora-raster-") as temporary:
        raster = Raster(Path(temporary))
        scan, normals, uv, faces, texture = unpack_scan(args.scan)
        filtered = texture.filter(ImageFilter.GaussianBlur(.68))
        ground = Ground(raster, scan, faces, uv, filtered, ElevationTiles(ROOT / "data/terrain"))
        # Remove the two roof forms baked into the scan at their measured site grade.
        # All coordinates and elevations are in the registered 2016 scan frame.
        ground.repair_grade(scan, normals, center=(-144, -618), extent=(20, 18),
                            elevation=-426.2, grade=(.055, .086), feather=8)
        ground.repair_grade(scan, normals, center=(2, -450), extent=(20, 8),
                            elevation=-391.1, grade=(-.06, .11), feather=5)
        geometry = ground.mesh(scan, normals, faces, uv)
        nodes, ways = read_osm(ROOT / "data/kalampaka.osm.gz")
        masks, buildings, pitches = masks_for(ground, ways)
        roofs = RoofSurvey(args.roof_survey).match(buildings, PROFILES) if args.roof_survey else {}
        life = SceneLife(ways, ground, project, (W, H))
        art, trails = render(raster, ground, np.asarray(filtered), geometry, masks, buildings, pitches, nodes, roofs, life)
        places = json.loads((ROOT / "data/places.json").read_text())
        save_map(output, art, trails, ground, places)
        life.save(output)
        if args.roof_survey and not args.output:
            save_roof_preview(output.parent)


if __name__ == "__main__":
    main()
