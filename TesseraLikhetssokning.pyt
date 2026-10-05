# -*- coding: utf-8 -*-
"""
TesseraLikhetssokning.pyt

Likhetssökning i ett embedding-raster: hur likt är varje pixel en referenspunkt
eller referenspolygon, uttryckt som skalärprodukten mellan pixelns vektor och
referensens vektor.

Verktyget är byggt för Tessera-embeddings (se "Tessera embeddings to GDB" i denna
mapp) men fungerar på vilket flerbandsraster som helst där varje pixel är en
vektor i samma rum, t.ex. Googles "Satellite Embedding"-dataset. Metoden är
densamma som i Google Earth Engines exempel på likhetssökning:

    https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search

Där bygger man en referensvektor genom att medelvärdesbilda embeddingen inom en
referensgeometri, och jämför den mot varje pixel med skalärprodukt. Googles
embeddings är enhetsvektorer per pixel, så skalärprodukten blir kosinuslikhet
(-1 till 1) utan vidare beräkning. Tesseras dokumentation garanterar inte att
dess 128-kanalers embeddings är enhetsnormerade på samma sätt, så det här
verktyget normerar båda vektorerna explicit innan skalärprodukten tas
("Normalisera vektorer"), vilket ger samma resultat när indata redan är
enhetsvektorer och ett väldefinierat -1-till-1-mått annars. Normeringen kan
stängas av för en ren, oskalad skalärprodukt.

Flera referensobjekt (flera ritade punkter, eller flera polygoner i samma
lager) ger var sin referensvektor. Likhetsrastret blir då medelvärdet av
likheten mot var och en, samma princip som i Earth Engine-exemplet när flera
referenspunkter används.

Beräkningen läser embedding-rastret radvis i block för att hålla minnesbehovet
nere — det är bara indata som strömmas, utdata (ett enda band) hålls i minnet
och skrivs i ett svep.

Krav : ArcGIS Pro 3.x (arcpy). Inga paket utöver Pythons standardbibliotek och
       numpy.
"""

import os
import re
import shutil
import tempfile

import numpy as np

import arcpy

# ── Konstanter ────────────────────────────────────────────────────────────────

SWEREF99TM_WKID = 3006

CAT_REF = "Referens"
CAT_CALC = "Beräkning"
CAT_MAP = "Karta"

_SCRATCH_DIRNAME = "TesseraSimilarity_arbetsmapp"

# Minnesbudget per inläst radblock (bytes). Utdata är ett enda band och hålls
# alltid helt i minnet — bara indataläsningen strömmas i block.
_BLOCK_BUDGET_BYTES = 256 * 1024 ** 2

# Varna om utdatarastret blir större än så här (bredd x höjd), eftersom hela
# likhetsrastret hålls i minnet som en float32-array.
_MAX_SANE_PIXELS = 150_000_000


def _sr(wkid):
    return arcpy.SpatialReference(wkid)


def _sr_is_valid(sr):
    """Samma test som i Tessera-hämtningsverktyget: factoryCode 0 är giltigt
    för ett eget definierat koordinatsystem så länge det har en WKT-definition."""
    if sr is None:
        return False
    try:
        if sr.factoryCode:
            return True
        return bool(sr.exportToString())
    except Exception:
        return False


def _project_geometry(geom, target_sr):
    """Projicera en verklig geometri (från en featureklass eller ett Feature
    Set) till target_sr. Till skillnad från ett GPExtent-objekt bär en sådan
    geometri alltid sitt eget koordinatsystem, så projectAs kan användas
    direkt."""
    try:
        return geom.projectAs(target_sr)
    except Exception as exc:
        raise ValueError(
            "Kunde inte omvandla referensgeometrin till rastrets koordinatsystem "
            "({}): {}".format(target_sr.name, exc)
        )


def _new_feature_set(geometry_type, sr):
    """Ett tomt Feature Set-schema av given geometrityp, för att styra vilket
    ritverktyg (punkt eller polygon) som visas i dialogen."""
    workspace = "memory"
    name = "tesserasim_ref_{}".format(geometry_type.lower())
    fc = "{}/{}".format(workspace, name)
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)
    arcpy.management.CreateFeatureclass(workspace, name, geometry_type, spatial_reference=sr)
    return fc


# =============================================================================
# Band
# =============================================================================

def _parse_bands(text, n_bands):
    """Tolka en bandangivelse som "1-16,64" till nollbaserade index. Tom
    sträng ger alla band. Banden numreras 1-n_bands i dialogen."""
    text = (text or "").strip()
    if not text:
        return list(range(n_bands))

    if not re.fullmatch(r"[0-9,\-\s]+", text):
        raise ValueError(
            "Ogiltig bandangivelse: '{}'. Ange band som t.ex. 1-16,64.".format(text)
        )

    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            if first > last:
                raise ValueError("Ogiltigt bandintervall: '{}'.".format(part))
            values = range(first, last + 1)
        else:
            values = [int(part)]
        for value in values:
            if not 1 <= value <= n_bands:
                raise ValueError(
                    "Band {} finns inte — rastret har band 1-{}.".format(value, n_bands)
                )
            if value - 1 not in indices:
                indices.append(value - 1)

    if not indices:
        raise ValueError("Ingen giltig bandangivelse.")
    return sorted(indices)


# =============================================================================
# Referensvektorer
# =============================================================================

def _feature_count(value):
    if value is None:
        return 0
    try:
        return int(arcpy.management.GetCount(value)[0])
    except Exception:
        return 0


def _reference_kind_and_source(points_value, polygon_value, layer_value):
    """
    Vilken av de tre referensparametrarna som är ifylld, och vilken sorts
    geometri den innehåller. Exakt en får vara ifylld.
    """
    candidates = []
    if _feature_count(points_value) > 0:
        candidates.append(("point", points_value))
    if _feature_count(polygon_value) > 0:
        candidates.append(("polygon", polygon_value))
    if _feature_count(layer_value) > 0:
        shape = arcpy.Describe(layer_value).shapeType
        if shape == "Point":
            candidates.append(("point", layer_value))
        elif shape == "Polygon":
            candidates.append(("polygon", layer_value))
        else:
            raise ValueError(
                "Referenslagret måste innehålla punkter eller polygoner (har {}).".format(shape)
            )

    if not candidates:
        raise ValueError(
            "Ange en referens: rita en punkt, rita en polygon, eller välj ett "
            "befintligt punkt- eller polygonlager."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Ange bara en referenskälla — punkt, polygon eller befintligt lager, inte flera."
        )
    return candidates[0]


def _point_vectors(raster, raster_sr, band_indices, source, messages):
    """Embedding-vektorn i pixeln under varje referenspunkt."""
    ext = raster.extent
    cell_w, cell_h = raster.meanCellWidth, raster.meanCellHeight
    n_bands = raster.bandCount

    vectors = []
    skipped = 0
    with arcpy.da.SearchCursor(source, ["SHAPE@"]) as cursor:
        for (shape,) in cursor:
            point = _project_geometry(shape, raster_sr)
            x, y = point.centroid.X, point.centroid.Y
            if not (ext.XMin <= x < ext.XMax and ext.YMin <= y < ext.YMax):
                skipped += 1
                continue
            col = int((x - ext.XMin) / cell_w)
            row = int((ext.YMax - y) / cell_h)
            origin = arcpy.Point(ext.XMin + col * cell_w, ext.YMax - (row + 1) * cell_h)
            block = arcpy.RasterToNumPyArray(raster, origin, 1, 1, nodata_to_value=np.nan)
            if n_bands == 1:
                block = block[np.newaxis, :, :]
            pixel = block[:, 0, 0]
            if np.isnan(pixel).any():
                skipped += 1
                continue
            vectors.append(pixel[band_indices].astype(np.float32))

    if skipped:
        messages.addWarningMessage(
            "{} referenspunkt(er) låg utanför rastret eller på NoData och "
            "hoppades över.".format(skipped)
        )
    if not vectors:
        raise ValueError("Ingen referenspunkt gav en giltig embedding-vektor.")
    return vectors


def _polygon_rings(geom):
    """
    Alla ringar (yttre och hål) i en polygongeometri, som listor av (x, y).

    Vid iteration över en arcpy Polygon separeras en rings punkter från nästa
    med värdet None inom samma part — det är så hål upptäcks utan att behöva
    tolka ringriktning (medurs/moturs).
    """
    rings = []
    for part in geom:
        ring = []
        for pnt in part:
            if pnt is None:
                if len(ring) >= 3:
                    rings.append(ring)
                ring = []
            else:
                ring.append((pnt.X, pnt.Y))
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _points_in_rings(rings, xs, ys):
    """
    Jämna-udda punkt-i-polygon-test över samtliga ringar i en polygon,
    vektoriserat över en punktmängd (xs, ys, samma form).

    Jämna-udda-regeln räknar med alla ringar tillsammans — en hålring vänder
    på pariteten inom sitt område, så hål hanteras korrekt utan särskild
    hål-logik.
    """
    inside = np.zeros(xs.shape, dtype=bool)
    for ring in rings:
        rx = np.array([p[0] for p in ring], dtype=np.float64)
        ry = np.array([p[1] for p in ring], dtype=np.float64)
        rx_next = np.roll(rx, -1)
        ry_next = np.roll(ry, -1)
        for xi, yi, xj, yj in zip(rx, ry, rx_next, ry_next):
            cond = (yi > ys) != (yj > ys)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_intersect = (xj - xi) * (ys - yi) / (yj - yi) + xi
            cond &= xs < x_intersect
            inside ^= cond
    return inside


def _polygon_vectors(raster, raster_sr, band_indices, source, messages):
    """
    Medelvärdesvektorn per referenspolygon.

    Rasteriseringen görs med en egen vektoriserad punkt-i-polygon-test i
    stället för PolygonToRaster: det verktyget kräver Spatial Analyst eller
    3D Analyst-licens på Basic/Standard-nivå (mätt — ger "ERROR 000824: The
    tool is not licensed" på en Pro Basic-licens), vilket bryter mot kravet
    att bara använda arcpy, numpy och standardbiblioteket.
    """
    ext = raster.extent
    cell_w, cell_h = raster.meanCellWidth, raster.meanCellHeight
    width, height = raster.width, raster.height
    n_bands = raster.bandCount

    vectors = []
    n_skipped = 0
    with arcpy.da.SearchCursor(source, ["SHAPE@"]) as cursor:
        for (shape,) in cursor:
            polygon = _project_geometry(shape, raster_sr)
            rings = _polygon_rings(polygon)
            if not rings:
                n_skipped += 1
                continue

            p_ext = polygon.extent
            c0 = max(0, int((p_ext.XMin - ext.XMin) / cell_w) - 1)
            c1 = min(width, int((p_ext.XMax - ext.XMin) / cell_w) + 2)
            r0 = max(0, int((ext.YMax - p_ext.YMax) / cell_h) - 1)
            r1 = min(height, int((ext.YMax - p_ext.YMin) / cell_h) + 2)
            if c1 <= c0 or r1 <= r0:
                n_skipped += 1
                continue

            cols = np.arange(c0, c1)
            rows = np.arange(r0, r1)
            xs = ext.XMin + (cols + 0.5) * cell_w
            ys = ext.YMax - (rows + 0.5) * cell_h
            xx, yy = np.meshgrid(xs, ys)
            inside = _points_in_rings(rings, xx.ravel(), yy.ravel()).reshape(xx.shape)
            if not inside.any():
                n_skipped += 1
                continue

            origin = arcpy.Point(ext.XMin + c0 * cell_w, ext.YMax - r1 * cell_h)
            block = arcpy.RasterToNumPyArray(raster, origin, c1 - c0, r1 - r0, nodata_to_value=np.nan)
            if n_bands == 1:
                block = block[np.newaxis, :, :]
            valid = inside & ~np.isnan(block).any(axis=0)
            if not valid.any():
                n_skipped += 1
                continue
            mean_vec = block[:, valid].mean(axis=1)
            vectors.append(mean_vec[band_indices].astype(np.float32))

    if n_skipped:
        messages.addWarningMessage(
            "{} referenspolygon(er) täckte inga giltiga pixlar och hoppades "
            "över.".format(n_skipped)
        )
    if not vectors:
        raise ValueError("Ingen referenspolygon gav en giltig embedding-vektor.")
    return vectors


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("En referensvektor är nollvektorn och kan inte normaliseras.")
    return (vector / norm).astype(np.float32)


# =============================================================================
# Likhetsraster
# =============================================================================

def _similarity_raster(raster, band_indices, ref_vectors, normalize, messages):
    """
    Beräkna likheten mot referensvektorn (eller medelvärdet av flera) för
    varje pixel i rastret. Läser indata radvis i block, håller hela utdata i
    minnet som en float32-array.
    """
    ext = raster.extent
    width, height = raster.width, raster.height
    cell_w, cell_h = raster.meanCellWidth, raster.meanCellHeight
    n_bands_total = raster.bandCount

    vectors = [_unit(v) for v in ref_vectors] if normalize else list(ref_vectors)

    out = np.full((height, width), np.nan, dtype=np.float32)

    bytes_per_row = max(n_bands_total * width * 4, 1)
    block_rows = max(1, min(height, int(_BLOCK_BUDGET_BYTES / bytes_per_row)))

    arcpy.SetProgressor("step", "Beräknar likhet...", 0, height, block_rows)
    try:
        row = 0
        while row < height:
            rows = min(block_rows, height - row)
            y_top = ext.YMax - row * cell_h
            y_bottom = y_top - rows * cell_h
            origin = arcpy.Point(ext.XMin, y_bottom)

            block = arcpy.RasterToNumPyArray(raster, origin, width, rows, nodata_to_value=np.nan)
            if n_bands_total == 1:
                block = block[np.newaxis, :, :]
            block = block[band_indices, :, :].astype(np.float32)

            invalid = np.isnan(block).any(axis=0)
            block_filled = np.where(np.isnan(block), 0.0, block)

            if normalize:
                norm = np.sqrt((block_filled ** 2).sum(axis=0))
                zero_norm = norm == 0
                norm_safe = np.where(zero_norm, 1.0, norm)
                invalid = invalid | zero_norm

            acc = np.zeros((rows, width), dtype=np.float64)
            for vector in vectors:
                dot = np.tensordot(vector, block_filled, axes=(0, 0))
                if normalize:
                    dot = dot / norm_safe
                acc += dot
            acc /= len(vectors)
            acc[invalid] = np.nan

            # Radblocket motsvarar rader [row, row + rows) uppifrån i rastret,
            # samma ordning som RasterToNumPyArray läser och NumPyArrayToRaster
            # förväntar sig vid skrivningen.
            out[row:row + rows, :] = acc.astype(np.float32)
            row += rows
            arcpy.SetProgressorPosition(row)
    finally:
        arcpy.ResetProgressor()

    return out


def _save_raster(array, raster, raster_sr, out_path, messages):
    previous_sr = arcpy.env.outputCoordinateSystem
    arcpy.env.outputCoordinateSystem = raster_sr
    try:
        result = arcpy.NumPyArrayToRaster(
            array, arcpy.Point(raster.extent.XMin, raster.extent.YMin),
            raster.meanCellWidth, raster.meanCellHeight,
            value_to_nodata=np.nan,
        )
        result.save(out_path)
    finally:
        arcpy.env.outputCoordinateSystem = previous_sr

    try:
        arcpy.management.CalculateStatistics(out_path)
    except Exception as exc:
        messages.addWarningMessage("Kunde inte beräkna statistik för rastret: {}".format(exc))
    return out_path


def _threshold_polygons(sim_array, raster, raster_sr, threshold, out_path, scratch_dir, messages):
    mask = np.where(np.isfinite(sim_array) & (sim_array >= threshold), 1, 0).astype(np.uint8)
    if not mask.any():
        messages.addWarningMessage(
            "Inga pixlar nådde tröskelvärdet {} — inga polygoner skapades.".format(threshold)
        )
        return None

    mask_path = os.path.join(scratch_dir, "sim_mask.tif")
    previous_sr = arcpy.env.outputCoordinateSystem
    arcpy.env.outputCoordinateSystem = raster_sr
    try:
        result = arcpy.NumPyArrayToRaster(
            mask, arcpy.Point(raster.extent.XMin, raster.extent.YMin),
            raster.meanCellWidth, raster.meanCellHeight,
            value_to_nodata=0,
        )
        result.save(mask_path)
    finally:
        arcpy.env.outputCoordinateSystem = previous_sr

    arcpy.conversion.RasterToPolygon(mask_path, out_path, "SIMPLIFY", "VALUE")
    return out_path


def _add_to_map(outputs, messages):
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        map_obj = aprx.activeMap
        if map_obj is None:
            maps = aprx.listMaps()
            map_obj = maps[0] if maps else None
    except Exception:
        map_obj = None
    if map_obj is None:
        messages.addWarningMessage("Ingen aktiv karta — resultatet lades inte till.")
        return
    for path in outputs:
        try:
            map_obj.addDataFromPath(path)
        except Exception as exc:
            messages.addWarningMessage("  Kunde inte lägga till {}: {}".format(path, exc))


def _default_gdb():
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        if aprx.defaultGeodatabase:
            return aprx.defaultGeodatabase
    except Exception:
        pass
    workspace = arcpy.env.workspace
    if workspace and str(workspace).lower().endswith(".gdb"):
        return workspace
    return None


# =============================================================================
# Toolbox
# =============================================================================

class Toolbox:
    def __init__(self):
        self.label = "Tessera: likhetssökning"
        self.alias = "tessera_likhet"
        self.tools = [EmbeddingSimilarity]


class EmbeddingSimilarity:
    def __init__(self):
        self.label = "Likhetssökning i embedding-raster"
        self.description = (
            "Jämför varje pixel i ett embedding-raster (t.ex. Tessera) mot en "
            "referenspunkt eller referenspolygon med skalärprodukt. Resultatet är "
            "ett enbandsraster där höga värden betyder att pixeln liknar "
            "referensen.\n\n"
            "Bygger på samma metod som Google Earth Engines exempel på "
            "likhetssökning i satellit-embeddings: medelvärdesbilda embeddingen "
            "inom referensen, och ta skalärprodukten mot varje pixel. Med "
            "normering (standard) blir måttet kosinuslikhet, -1 till 1."
        )
        self.canRunInBackground = False
        self._out_name_memo = ""
        self._poly_name_memo = ""

    def getParameterInfo(self):
        p_raster = arcpy.Parameter(
            displayName="Embedding-raster (t.ex. Tessera-mosaik)",
            name="in_raster", datatype="DERasterDataset",
            parameterType="Required", direction="Input",
        )

        p_bands = arcpy.Parameter(
            displayName="Band att använda, t.ex. 1-16,64 (tomt = alla band)",
            name="bands", datatype="GPString",
            parameterType="Optional", direction="Input",
        )

        p_ref_points = arcpy.Parameter(
            displayName="Ny referenspunkt (rita i kartan)",
            name="ref_points", datatype="GPFeatureRecordSetLayer",
            parameterType="Optional", direction="Input", category=CAT_REF,
        )
        p_ref_points.value = _new_feature_set("POINT", _sr(SWEREF99TM_WKID))

        p_ref_polygon = arcpy.Parameter(
            displayName="Ny referenspolygon (rita i kartan)",
            name="ref_polygon", datatype="GPFeatureRecordSetLayer",
            parameterType="Optional", direction="Input", category=CAT_REF,
        )
        p_ref_polygon.value = _new_feature_set("POLYGON", _sr(SWEREF99TM_WKID))

        p_ref_layer = arcpy.Parameter(
            displayName="Eller befintligt punkt- eller polygonlager",
            name="ref_layer", datatype="GPFeatureLayer",
            parameterType="Optional", direction="Input", category=CAT_REF,
        )
        p_ref_layer.filter.list = ["Point", "Polygon"]

        p_normalize = arcpy.Parameter(
            displayName="Normalisera vektorer (kosinuslikhet, -1 till 1)",
            name="normalize", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category=CAT_CALC,
        )
        p_normalize.value = True

        p_out_raster = arcpy.Parameter(
            displayName="Utdata: likhetsraster",
            name="out_raster", datatype="DERasterDataset",
            parameterType="Required", direction="Output",
        )
        gdb = _default_gdb()
        if gdb:
            p_out_raster.value = os.path.join(gdb, "likhet")

        p_threshold = arcpy.Parameter(
            displayName="Tröskelvärde för att extrahera de mest lika områdena (tomt = hoppa över)",
            name="threshold", datatype="GPDouble",
            parameterType="Optional", direction="Input", category=CAT_CALC,
        )

        p_out_polygons = arcpy.Parameter(
            displayName="Utdata: polygoner över tröskelvärdet",
            name="out_polygons", datatype="DEFeatureClass",
            parameterType="Optional", direction="Output", category=CAT_CALC,
        )

        p_overwrite = arcpy.Parameter(
            displayName="Skriv över befintlig utdata",
            name="overwrite", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category=CAT_CALC,
        )
        p_overwrite.value = True

        p_add = arcpy.Parameter(
            displayName="Lägg till resultatet i kartan",
            name="add_to_map", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category=CAT_MAP,
        )
        p_add.value = True

        return [p_raster, p_bands, p_ref_points, p_ref_polygon, p_ref_layer,
                p_normalize, p_out_raster, p_threshold, p_out_polygons,
                p_overwrite, p_add]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        (p_raster, _p_bands, _p_points, _p_polygon, _p_layer, _p_normalize,
         p_out_raster, p_threshold, p_out_polygons, _p_overwrite, _p_add) = parameters

        # Rasternamnet följer indatarastret tills användaren skrivit ett eget,
        # samma memo-mönster som i Tessera-hämtningsverktyget.
        if p_raster.value is not None:
            in_name = os.path.basename(str(p_raster.valueAsText)).rsplit(".", 1)[0]
            gdb = _default_gdb()
            suggestion = os.path.join(gdb, "{}_likhet".format(in_name)) if gdb else ""
            current = (p_out_raster.valueAsText or "").strip()
            if suggestion and current in ("", self._out_name_memo):
                p_out_raster.value = suggestion
            self._out_name_memo = suggestion

        p_out_polygons.enabled = p_threshold.value is not None
        if p_threshold.value is not None:
            out_text = (p_out_raster.valueAsText or "").strip()
            if out_text:
                suggestion = out_text + "_omraden"
                current = (p_out_polygons.valueAsText or "").strip()
                if current in ("", self._poly_name_memo):
                    p_out_polygons.value = suggestion
                self._poly_name_memo = suggestion

    def updateMessages(self, parameters):
        (p_raster, p_bands, p_points, p_polygon, p_layer, p_normalize,
         _p_out_raster, p_threshold, p_out_polygons, _p_overwrite, _p_add) = parameters

        if p_raster.value is not None:
            try:
                n_bands = arcpy.Describe(p_raster.value).bandCount
                _parse_bands(p_bands.valueAsText, n_bands)
            except ValueError as exc:
                p_bands.setErrorMessage(str(exc))
            except Exception:
                pass

        try:
            _reference_kind_and_source(p_points.value, p_polygon.value, p_layer.value)
        except ValueError as exc:
            p_layer.setErrorMessage(str(exc))

        if p_threshold.value is not None and bool(p_normalize.value if p_normalize.value is not None else True):
            if not (-1.0 <= p_threshold.value <= 1.0):
                p_threshold.setWarningMessage(
                    "Med normaliserade vektorer ligger likheten mellan -1 och 1."
                )

        if p_threshold.value is not None and not (p_out_polygons.valueAsText or "").strip():
            p_out_polygons.setErrorMessage(
                "Ange var polygonerna över tröskelvärdet ska sparas."
            )

    def execute(self, parameters, messages):
        in_raster = parameters[0].valueAsText
        bands_text = parameters[1].valueAsText
        points_value = parameters[2].value
        polygon_value = parameters[3].value
        layer_value = parameters[4].value
        normalize = bool(parameters[5].value) if parameters[5].value is not None else True
        out_raster = parameters[6].valueAsText
        threshold = parameters[7].value
        out_polygons = parameters[8].valueAsText
        overwrite = bool(parameters[9].value) if parameters[9].value is not None else True
        add_to_map = bool(parameters[10].value) if parameters[10].value is not None else True

        try:
            _run(in_raster, bands_text, points_value, polygon_value, layer_value,
                 normalize, out_raster, threshold, out_polygons, overwrite,
                 add_to_map, messages)
        except ValueError as exc:
            messages.addErrorMessage(str(exc))
            raise arcpy.ExecuteError

    def postExecute(self, parameters):
        return


# =============================================================================
# Körningens innehåll (separat funktion — går att testa utanför Pro)
# =============================================================================

def _run(raster_path, bands_text, points_value, polygon_value, layer_value,
         normalize, out_raster_path, threshold, out_polygons_path, overwrite,
         add_to_map, messages):
    """Utför hela likhetsberäkningen. Returnerar listan med skapade dataset."""

    if not arcpy.Exists(raster_path):
        raise ValueError("Rastret {} finns inte.".format(raster_path))

    kind, source = _reference_kind_and_source(points_value, polygon_value, layer_value)

    if not out_raster_path:
        raise ValueError("Ange var likhetsrastret ska sparas.")
    if arcpy.Exists(out_raster_path):
        if not overwrite:
            raise ValueError(
                "{} finns redan. Kryssa i 'Skriv över befintlig utdata' eller "
                "välj ett annat namn.".format(out_raster_path)
            )
        arcpy.management.Delete(out_raster_path)

    if threshold is not None:
        if not out_polygons_path:
            raise ValueError("Ange var polygonerna över tröskelvärdet ska sparas.")
        if arcpy.Exists(out_polygons_path):
            if not overwrite:
                raise ValueError(
                    "{} finns redan. Kryssa i 'Skriv över befintlig utdata' eller "
                    "välj ett annat namn.".format(out_polygons_path)
                )
            arcpy.management.Delete(out_polygons_path)

    raster = arcpy.Raster(raster_path)
    raster_sr = raster.spatialReference
    if not _sr_is_valid(raster_sr):
        raise ValueError("Embedding-rastret saknar koordinatsystem.")

    band_indices = _parse_bands(bands_text, raster.bandCount)
    messages.addMessage(
        "{} av {} band används.".format(len(band_indices), raster.bandCount)
    )

    if raster.width * raster.height > _MAX_SANE_PIXELS:
        messages.addWarningMessage(
            "Rastret har {} x {} pixlar. Likhetsrastret hålls helt i minnet — "
            "klipp indatarastret till ett mindre område om du får minnesfel.".format(
                raster.width, raster.height)
        )

    scratch_dir = tempfile.mkdtemp(prefix=_SCRATCH_DIRNAME + "_")
    try:
        if kind == "point":
            ref_vectors = _point_vectors(raster, raster_sr, band_indices, source, messages)
        else:
            ref_vectors = _polygon_vectors(raster, raster_sr, band_indices, source, messages)
        messages.addMessage(
            "{} referensvektor(er) extraherade ur {}.".format(
                len(ref_vectors), "punkt(er)" if kind == "point" else "polygon(er)")
        )

        messages.addMessage("Beräknar likhet mot hela rastret...")
        sim_array = _similarity_raster(raster, band_indices, ref_vectors, normalize, messages)

        valid = np.isfinite(sim_array)
        if not valid.any():
            raise ValueError("Alla pixlar blev NoData — inget resultat att skriva.")
        messages.addMessage(
            "Likhet ({}): min {:.3f}, max {:.3f}.".format(
                "kosinus" if normalize else "oskalad skalärprodukt",
                float(sim_array[valid].min()), float(sim_array[valid].max()))
        )

        _save_raster(sim_array, raster, raster_sr, out_raster_path, messages)
        messages.addMessage("Skrev {}".format(out_raster_path))
        outputs = [out_raster_path]

        if threshold is not None:
            poly_path = _threshold_polygons(
                sim_array, raster, raster_sr, threshold, out_polygons_path,
                scratch_dir, messages
            )
            if poly_path:
                messages.addMessage("Skrev {}".format(poly_path))
                outputs.append(poly_path)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    if add_to_map:
        _add_to_map(outputs, messages)

    messages.addMessage("Klar!")
    return outputs
