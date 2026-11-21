# Working on Tiny Meteora

Tiny Meteora is a static, bilingual pixel-art atlas. The browser runs the checked-in HTML, CSS, and JavaScript in `dist/`. The Python renderer is a separate tool that generates the map imagery and geographic overlays.

## Start here

- Read the [README](README.md) for local serving, input data, full renders, and deployment.
- For viewer changes, start `python3 -m http.server 4173 --bind 127.0.0.1 --directory dist` from the repository root. The bundled map is enough to test UI changes.
- Use `scripts/render_scan_map.py` for geographic changes. It needs the original 2016 GLB in addition to the locally fetched OSM and terrain data, plus `cc`, NumPy, and Pillow. Render to an ignored preview directory before replacing bundled assets.

## Find the right code

| Task | Start with |
| --- | --- |
| Layout, map controls, markers, search, share links | `dist/index.html`, `dist/atlas.css`, `dist/atlas.js` |
| English and Greek UI text | `dist/i18n.js` and the corresponding `data-i18n` binding in `dist/index.html` |
| Landmark names, coordinates, and descriptions | `data/places.json`, then regenerate the projected `dist/assets/map.json` |
| Animation sprites and playback | `dist/scene-life.js` |
| Walkable and drivable routes, water samples, lit windows | `scripts/scan_life.py`, then regenerate the overlay assets |
| Scan registration, elevation, materials, and rasterization | `scripts/scan_geometry.py`, `scripts/render_scan_map.py`, `scripts/scan_raster.c` |
| Road masks, monastery offsets, churches and civic geometry | `scripts/scan_features.py`, `scripts/scan_churches.py` |

## Preserve map behavior

- Keep map imagery, projected place markers, trails, and animation routes on the same coordinate system. Use the renderer’s projection for geographic features.
- Keep landmark IDs stable. Shared URLs store place IDs, pixel coordinates, map version, language, lighting, and layer flags. Keep the serialized layer order stable when rearranging UI controls. If you change the projection, update the map version and handle old shared views.
- Keep the previous pixel origin when extending a map edge, where practical. Regenerate the whole asset set together when changing coverage or projection.
- Give new labels and landmark details both English and Greek copy. Test search using both alphabets in either UI language.
- Keep imagery and interaction separate. The viewer uses small bundles and a tiled atlas. Preserve keyboard controls and the existing reduced-motion behavior.

## Verify changes locally

- Run `git diff --check`. Check modified JavaScript modules with commands such as `node --input-type=module --check < dist/atlas.js`.
- For a UI change, exercise the affected controls on desktop and a narrow viewport. Check both languages and a shared view whenever changing labels, search, or URL state.
- For a renderer change, build into a scratch directory. Check tile coverage at every level, the bounds of projected points, and the alignment of markers and overlays at street zoom.
- For a Cloudflare configuration change, use Wrangler’s `deploy --dry-run` and local development mode. Publishing and account changes need a separate user request.

## Data and release boundaries

- Keep original survey files, personal reference photos, API credentials, and preview outputs out of Git. Use `cache/` for local inputs and `prototype/` for scratch output.
- Keep downloaded OSM and terrain inputs and personal deployment metadata out of Git. See [map sources](README.md#license-and-map-sources) for attribution.
