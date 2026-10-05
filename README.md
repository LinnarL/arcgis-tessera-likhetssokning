# Tessera likhetssökning

ArcGIS Pro Python toolbox that compares every pixel in an embedding raster against a reference
point or polygon, using the dot product between the pixel's vector and the reference vector.
High values mean the pixel looks like the reference.

Built for Tessera embeddings (see "Tessera embeddings to GDB" in this project) but works on any
multi-band raster where each pixel is a vector in the same space, for example Google's Satellite
Embedding dataset. The method follows Google Earth Engine's similarity search tutorial: average
the embedding over a reference geometry, then take the dot product of that vector against every
pixel.

https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search

Google's Satellite Embedding vectors are unit length by construction, so the dot product is
already cosine similarity. Tessera's documentation does not guarantee the same, so this tool
normalizes both vectors before taking the dot product by default, which gives the same result
when the input is already unit length and a well-defined -1 to 1 similarity measure otherwise.
Normalization can be turned off for a raw, unscaled dot product.

Multiple reference features (several drawn points, or several polygons in one layer) each
produce their own reference vector. The output is the average of the similarity against each
one, the same approach the Earth Engine tutorial uses when several reference points are given.

## Requirements

- ArcGIS Pro 3.x. Developed and tested on 3.6 with Python 3.13.
- No dependencies beyond `arcpy`, `numpy` and the standard library. No Spatial Analyst or 3D
  Analyst extension is needed — polygon rasterization is done with a small vectorized
  point-in-polygon test instead of `PolygonToRaster`, which requires one of those extensions on
  a Basic or Standard license.

## Install

1. Clone or download this repo.
2. In ArcGIS Pro: Catalog, Toolboxes, Add Toolbox, select `TesseraLikhetssokning.pyt`.
3. Open Tessera similarity, Likhetssökning i embedding-raster.

## The tool dialog

The UI is in Swedish, matching a Swedish ArcGIS Pro install.

| Parameter | Default | Notes |
|---|---|---|
| Embedding-raster | - | Any multi-band raster, e.g. a Tessera mosaic |
| Band att använda | all bands | Accepts `1-16,64` |
| Ny referenspunkt | - | Sketch tool, draw one or more points directly on the map |
| Ny referenspolygon | - | Sketch tool, draw one or more polygons directly on the map |
| Eller befintligt punkt- eller polygonlager | - | Use an existing layer instead of sketching |
| Normalisera vektorer | on | Cosine similarity (-1 to 1) instead of a raw dot product |
| Utdata: likhetsraster | `<raster>_likhet` in the project default gdb | |
| Tröskelvärde | off | If set, also extracts polygons of pixels at or above this value |
| Utdata: polygoner över tröskelvärdet | `<likhetsraster>_omraden` | Required if a threshold is set |
| Skriv över befintlig utdata | on | |
| Lägg till resultatet i kartan | on | |

Give exactly one reference: a sketched point, a sketched polygon, or an existing point or
polygon layer.

## Output

A single-band float raster the same size and grid as the input, holding the similarity value at
every pixel. Pixels that are NoData in any used band of the input are NoData in the output.

With a threshold set, a second output holds the polygons where the similarity is at or above
that value — the same "extract candidate matches" step as the Earth Engine tutorial's threshold
mask, done here as a vector output instead of a binary raster.

## Notes on the method

- The reference vector is the mean of the embedding over the reference geometry: the pixel under
  a point, or the mean of all valid pixels inside a polygon (holes are respected).
- Reading is done row-block by row-block to bound memory use regardless of raster size; the
  output, being a single band, is built in memory and written in one pass. A raster much larger
  than about 150 million pixels will warn that it may need clipping first.
- Polygon rasterization does not use `arcpy.conversion.PolygonToRaster`: that tool requires a
  Spatial Analyst or 3D Analyst license on ArcGIS Pro Basic or Standard. Instead the reference
  polygon is rasterized with a vectorized even-odd point-in-polygon test against cell centers,
  which needs no extension.

## Source

- Method: https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search
- Data this is built for: https://geotessera.org/
