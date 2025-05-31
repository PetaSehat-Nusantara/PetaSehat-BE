import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import googlemaps
from geopy.geocoders import Nominatim
from google import genai

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Konfigurasi Logging & Env
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ----------------------------
# Inisialisasi FastAPI + CORS
# ----------------------------
app = FastAPI(title="Rekomendasi Lokasi Rumah Sakit", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # ["*"] untuk pengembangan lokal (opsional)
    ],
    allow_methods=["GET", "POST", "OPTIONS"],  # Harus mencakup "OPTIONS"
    allow_headers=["*"],
    allow_credentials=True,
)


# ----------------------------
# Pydantic Models
# ----------------------------
class InformasiUmum(BaseModel):
    facilityClass: str
    facilityName: str
    facilityType: str
    maxCapacity: int
    minCapacity: int
    services: List[str]


class KriteriaDemografi(BaseModel):
    targetDemografiUtama: List[str]
    targetPendapatanPasien: List[str]


class KriteriaKeuangan(BaseModel):
    estimasiAnggaranMaksimum: int
    estimasiAnggaranMinimum: int
    targetROI: int
    targetWaktuPembangunan: str


class LokasiLahan(BaseModel):
    accessibilityPreferences: List[str]
    areaPreferences: List[str]
    city: str
    district: str
    environmentalPreferences: List[str]
    landPreferences: List[str]
    province: str


class HospitalRequest(BaseModel):
    informasiUmum: InformasiUmum
    kriteriaDemografi: KriteriaDemografi
    kriteriaKeuangan: KriteriaKeuangan
    lokasiLahan: LokasiLahan


class LocationRecommendation(BaseModel):
    kecamatan: str
    gmaps_coordinates: Dict[str, float]
    gmaps_place_id: Optional[str]
    composite_score: float
    rank: int
    population_served: int
    healthcare_gap_filled: float
    justification: str
    impact_metrics: Dict[str, Any]
    normalized_scores: Dict[str, float]
    # Tambahan field sesuai permintaan:
    kecocokan_percentage: float
    poin_unggul: List[str]
    poin_perhatian: List[str]
    estimasi_biaya_konstruksi: str
    estimasi_waktu_pengembangan: str
    luasan_bangunan: str


class RecommendationResponse(BaseModel):
    recommendations: List[LocationRecommendation]
    methodology: str
    analysis_timestamp: str
    processing_steps: List[str]


# ----------------------------
# JSONOutputParser (tidak berubah)
# ----------------------------
class JSONOutputParser:
    def parse(self, text: str) -> dict:
        try:
            t = text.strip()
            if t.startswith("```json"):
                t = t[len("```json") :].strip()
            if t.endswith("```"):
                t = t[:-3].strip()
            return json.loads(t)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}\nTeks:\n{text}")
            return {"error": f"Gagal mengurai JSON: {e}"}


json_parser = JSONOutputParser()

# ----------------------------
# Placeholder Layanan Global
# ----------------------------
db = None
gmaps = None
geolocator = None
genai_client = None


# ----------------------------
# Fungsi Pembantu
# ----------------------------
async def get_firestore_schema_func() -> Dict[str, Any]:
    try:
        collections = db.collections()
        schema: Dict[str, Any] = {}
        for collection in collections:
            ref = db.collection(collection.id)
            docs = ref.limit(3).stream()
            sample_docs = [doc.to_dict() for doc in docs]
            if sample_docs:
                schema[collection.id] = {
                    "fields": list(sample_docs[0].keys()),
                    "sample_data": sample_docs[0],
                    "estimated_count": len(sample_docs),
                }
        return schema
    except Exception as e:
        logger.error(f"Error mendapatkan skema Firestore: {e}")
        return {}


async def collect_firestore_data_func(
    retrieval_plan: Dict[str, Any], target_province: str
) -> Dict[str, Any]:
    try:
        plan = (
            retrieval_plan
            if isinstance(retrieval_plan, dict)
            else json.loads(retrieval_plan)
        )
        collected_data: Dict[str, Any] = {}
        for query_config in plan.get("collectionsToQuery", []):
            name = query_config["collection"]
            fields = query_config.get("fields", [])
            filters = query_config.get("filters", {})

            ref = db.collection(name)
            query = ref
            for f_key, f_val in filters.items():
                if f_key == "province":
                    query = query.where(f_key, "==", target_province)
                else:
                    query = query.where(f_key, "==", f_val)

            docs = query.limit(50).stream()
            docs_list = []
            for doc in docs:
                data = doc.to_dict()
                filtered = {fld: data[fld] for fld in fields if fld in data}
                if filtered:
                    filtered["doc_id"] = doc.id
                    docs_list.append(filtered)
            collected_data[name] = docs_list
        return collected_data
    except Exception as e:
        logger.error(f"Error mengumpulkan data Firestore: {e}")
        return {"error": str(e)}


def normalize_and_scale_func(
    locations_data: List[Dict[str, Any]], priority_weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    if not locations_data:
        return []

    metrics = [
        "populationDensity",
        "healthcareGap",
        "economicViability",
        "demographicMatch",
        "accessibility",
    ]
    ranges: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = [float(loc.get(m, 0)) for loc in locations_data]
        if vals:
            ranges[m] = {"min": min(vals), "max": max(vals)}
        else:
            ranges[m] = {"min": 0.0, "max": 1.0}

    scored = []
    for loc in locations_data:
        composite = 0.0
        normalized_scores: Dict[str, float] = {}
        for m in metrics:
            raw = float(loc.get(m, 0))
            rmin = ranges[m]["min"]
            rmax = ranges[m]["max"]
            if rmax == rmin:
                norm = 1.0
            else:
                norm = (raw - rmin) / (rmax - rmin)
            scaled = norm * 100.0
            normalized_scores[m] = round(scaled, 2)
            w = float(priority_weights.get(m.lower(), 0.2))
            composite += scaled * w

        new_loc = loc.copy()
        new_loc["normalized_scores"] = normalized_scores
        new_loc["composite_score"] = round(composite, 2)
        scored.append(new_loc)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, loc in enumerate(scored):
        loc["rank"] = i + 1
    return scored


async def get_gmaps_coordinates_func(
    kecamatan: str, city: str, province: str
) -> Dict[str, Any]:
    try:
        q = f"{kecamatan}, {city}, {province}, Indonesia"
        res = gmaps.geocode(q)
        if res:
            first = res[0]
            loc = first["geometry"]["location"]
            return {
                "lat": loc["lat"],
                "lng": loc["lng"],
                "place_id": first.get("place_id"),
                "formatted_address": first.get("formatted_address"),
                "gmaps_url": f"https://maps.google.com/?q={loc['lat']},{loc['lng']}",
            }
        else:
            return {
                "lat": -6.2088,
                "lng": 106.8456,
                "place_id": None,
                "formatted_address": f"Lokasi perkiraan: {q}",
                "gmaps_url": None,
            }
    except Exception as e:
        logger.error(f"Error geocoding '{kecamatan}': {e}")
        return {
            "lat": -6.2088,
            "lng": 106.8456,
            "place_id": None,
            "formatted_address": f"Gagal geocode: {q}",
            "gmaps_url": None,
        }


# ----------------------------
# Startup Event
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global db, gmaps, geolocator, genai_client

    # 1) Inisialisasi Firebase
    FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")
    if not FIREBASE_CRED_PATH or not os.path.exists(FIREBASE_CRED_PATH):
        raise RuntimeError(
            f"FIREBASE_CRED_PATH tidak ditemukan atau tidak valid: {FIREBASE_CRED_PATH!r}"
        )
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        _ = list(db.collections())
        logger.info("✅ Firebase berhasil dihubungkan.")
    except Exception as e:
        logger.error(f"❌ Inisialisasi Firebase gagal: {e}")
        raise RuntimeError(f"Gagal inisialisasi Firebase: {e}")

    # 2) Inisialisasi Google Maps
    GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
    if not GMAPS_API_KEY:
        raise RuntimeError("GMAPS_API_KEY tidak ditemukan")
    try:
        gmaps = googlemaps.Client(key=GMAPS_API_KEY)
        test = gmaps.geocode("Jakarta, Indonesia")
        if not test:
            raise Exception("Hasil geocode kosong")
        logger.info("✅ Google Maps API berhasil divalidasi.")
    except Exception as e:
        logger.error(f"❌ Inisialisasi Google Maps gagal: {e}")
        raise RuntimeError(f"Gagal inisialisasi Google Maps API: {e}")

    # 3) (Opsional) Inisialisasi Nominatim
    geolocator = Nominatim(user_agent="hospital_recommender")

    # 4) Inisialisasi GenAI Client
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY untuk GenAI tidak ditemukan")
    try:
        genai_client = genai.Client(api_key=GOOGLE_API_KEY)
        smoke = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
            },
            contents="Halo, GenAI! Jika kamu aktif, balas dengan 'siap'.",
        )
        logger.info(f"✅ Uji coba GenAI: {smoke.text[:20]!r}…")
    except Exception as e:
        logger.error(f"❌ Inisialisasi GenAI gagal: {e}")
        raise RuntimeError(f"Gagal inisialisasi GenAI: {e}")


# ----------------------------
# Endpoint Biasa (tanpa streaming)
# ----------------------------
@app.post("/recommend-hospital-location", response_model=RecommendationResponse)
async def recommend_hospital_location(request: HospitalRequest):
    """
    Menghasilkan rekomendasi lokasi rumah sakit berdasarkan kriteria pengguna,
    menyimpan hasil ke Firestore pada koleksi 'NusaCari', dan mengembalikan respons.
    Semua instruksi dan output dalam Bahasa Indonesia.
    """
    processing_steps: List[str] = []

    # Pastikan semua layanan telah diinisialisasi
    if not all([db, gmaps, genai_client]):
        raise HTTPException(
            status_code=500,
            detail="Layanan rekomendasi belum sepenuhnya siap. Periksa log startup.",
        )

    try:
        # -------------------
        # Langkah 1: Analisis Kebutuhan Pengguna
        # -------------------
        processing_steps.append("Menganalisis kebutuhan pengguna…")
        logger.info("Menganalisis kebutuhan pengguna…")

        user_input_json = request.json()
        requirements_prompt = f"""
Kamu adalah asisten perencanaan lokasi rumah sakit. Hanya berbicara dalam Bahasa Indonesia dan kembalikan jawaban murni dalam JSON.

Berikan saya analisis kebutuhan berdasarkan data berikut (format JSON):
{user_input_json}

– Identifikasi dan urutkan faktor prioritas (populationDensity, healthcareGap, economicViability, demographicMatch, accessibility).
– Kembalikan sebuah objek JSON persis dengan struktur ini (tanpa komentar tambahan):

{{
  "priorityFactors": {{
    "populationDensity": <angka bobot>,
    "healthcareGap": <angka bobot>,
    "economicViability": <angka bobot>,
    "demographicMatch": <angka bobot>,
    "accessibility": <angka bobot>
  }},
  "filters": {{
    "province": "<nama provinsi>",
    "facilityType": "<jenis fasilitas>"
  }}
}}
"""
        req_response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
            },
            contents=requirements_prompt,
        )
        requirements_analysis = json_parser.parse(req_response.text)

        # -------------------
        # Langkah 2: Ambil Skema Firestore dan Rencanakan Retrieval
        # -------------------
        processing_steps.append("Menganalisis skema Firestore…")
        logger.info("Menganalisis skema Firestore…")
        schema_dict = await get_firestore_schema_func()
        schema_json = json.dumps(schema_dict)

        retrieval_plan_prompt = f"""
Kamu memiliki skema Firestore (JSON) berikut:
{schema_json}

Dan kamu memiliki analisis prioritas/filter pengguna (JSON):
{json.dumps(requirements_analysis)}

Buat rencana query Firestore dalam bentuk JSON persis dengan struktur ini:

{{
  "collectionsToQuery": [
    {{
      "collection": "<nama_koleksi>",
      "fields": ["<field1>", "<field2>", …],
      "filters": {{ "province": "{request.lokasiLahan.province}", "facilityType": "{request.informasiUmum.facilityType}" }}
    }},
    …
  ]
}}
"""
        retrieval_plan_resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
            },
            contents=retrieval_plan_prompt,
        )
        retrieval_plan = json_parser.parse(retrieval_plan_resp.text)

        # -------------------
        # Langkah 3: Himpun Data dari Firestore
        # -------------------
        processing_steps.append("Mengumpulkan data dari Firestore…")
        logger.info("Mengumpulkan data dari Firestore…")
        collected_data = await collect_firestore_data_func(
            retrieval_plan, request.lokasiLahan.province
        )

        # -------------------
        # Langkah 4: Ekstrak Data Numerik
        # -------------------
        processing_steps.append("Mengekstrak data numerik…")
        logger.info("Mengekstrak data numerik…")
        extracted_prompt = f"""
Kamu memiliki data Firestore mentah (JSON):
{json.dumps(collected_data)}

Dan kamu memiliki analisis prioritas pengguna (JSON):
{json.dumps(requirements_analysis)}

Untuk provinsi: {request.lokasiLahan.province}

Ekstrak metrik numerik untuk setiap kecamatan ke dalam format JSON ini secara tepat:

{{
  "locations": [
    {{
      "kecamatan": "<nama_kecamatan>",
      "populationDensity": <angka>,
      "healthcareGap": <angka>,
      "economicViability": <angka>,
      "demographicMatch": <angka>,
      "accessibility": <angka>
    }},
    …
  ]
}}
"""
        extraction_resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
            },
            contents=extracted_prompt,
        )
        extracted_data = json_parser.parse(extraction_resp.text)

        # -------------------
        # Langkah 5: Normalisasi dan Skor
        # -------------------
        processing_steps.append("Menormalisasi dan memberi skor…")
        logger.info("Menormalisasi dan memberi skor…")
        priority_weights = requirements_analysis.get("priorityFactors", {})
        scored_locations = normalize_and_scale_func(
            extracted_data.get("locations", []), priority_weights
        )

        # -------------------
        # Langkah 6: Analisis Dampak + Tambahan Field Kustom
        # -------------------
        processing_steps.append(
            "Melakukan analisis dampak dan menambahkan field kustom…"
        )
        logger.info("Melakukan analisis dampak dan menambahkan field kustom…")
        top_3 = scored_locations[:3]
        top_3_json = json.dumps(top_3)
        hospital_specs_json = request.informasiUmum.json()

        impact_prompt = f"""
Kamu adalah asisten perencanaan. Semua output harus dalam Bahasa Indonesia, tepat format JSON.

Berikut daftar 3 kecamatan terbaik (JSON), dengan skor dan metrik numerik:
{top_3_json}

Berikut spesifikasi rumah sakit yang diusulkan (JSON):
{hospital_specs_json}

Untuk setiap kecamatan di atas, buatlah objek JSON yang berisi:
- "kecamatan"
- "population_served" (perkiraan jumlah penduduk yang dilayani)
- "healthcare_gap_filled" (perkiraan selisih kebutuhan layanan kesehatan yang terpenuhi)
- "justification" (penjelasan singkat)
- "impact_metrics" (objek metrik tambahan apa pun)

Tambahan:
- "kecocokan_percentage" (persentase kecocokan lokasi, skala 0–100)
- "poin_unggul" (daftar minimal 3 poin kekuatan/kelebihan lokasi)
- "poin_perhatian" (daftar poin-poin yang perlu menjadi perhatian)
- "estimasi_biaya_konstruksi" (rentang kisaran biaya pembangunan)
- "estimasi_waktu_pengembangan" (perkiraan durasi pengembangan, mis. "12–18 bulan")
- "luasan_bangunan" (misal "500m2" atau rentang)

Kembalikan struktur persis seperti ini:

{{
  "impact_analysis": [
    {{
      "kecamatan": "<nama>",
      "population_served": <angka>,
      "healthcare_gap_filled": <angka>,
      "justification": "<teks>",
      "impact_metrics": {{ … }},
      "kecocokan_percentage": <angka>,
      "poin_unggul": ["<poin1>", "<poin2>", "<poin3>"],
      "poin_perhatian": ["<perhatianA>", "<perhatianB>"],
      "estimasi_biaya_konstruksi": "<kisaran biaya>",
      "estimasi_waktu_pengembangan": "<durasi>",
      "luasan_bangunan": "<luasan>"
    }},
    …
  ]
}}
"""
        impact_resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
            },
            contents=impact_prompt,
        )
        impact_analysis = json_parser.parse(impact_resp.text)

        # -------------------
        # Langkah 7: Gabungkan Koordinat & Simpan Hasil ke Firestore
        # -------------------
        processing_steps.append(
            "Menggabungkan koordinat, menyiapkan output akhir, dan menyimpan ke Firestore…"
        )
        logger.info(
            "Menggabungkan koordinat, menyiapkan output akhir, dan menyimpan ke Firestore…"
        )

        recommendations: List[LocationRecommendation] = []
        firestore_record = {
            "analysis_timestamp": datetime.now().isoformat(),
            "methodology": (
                "Analisis multi-faktor menggunakan kecocokan demografi, kebutuhan layanan kesehatan, "
                "kelayakan ekonomi, serta aksesibilitas, dengan tambahan analisis AI untuk detail biaya, "
                "waktu, dan luasan."
            ),
            "processing_steps": processing_steps,
            "recommendations": [],
        }

        for item in impact_analysis.get("impact_analysis", []):
            kecamatan = item.get("kecamatan", "Unknown")
            lokasi_match = next(
                (l for l in scored_locations if l["kecamatan"] == kecamatan), {}
            )
            coords = await get_gmaps_coordinates_func(
                kecamatan, request.lokasiLahan.city, request.lokasiLahan.province
            )

            rec = LocationRecommendation(
                kecamatan=kecamatan,
                gmaps_coordinates={"lat": coords["lat"], "lng": coords["lng"]},
                gmaps_place_id=coords.get("place_id"),
                composite_score=lokasi_match.get("composite_score", 0.0),
                rank=lokasi_match.get("rank", 0),
                population_served=item.get("population_served", 0),
                healthcare_gap_filled=item.get("healthcare_gap_filled", 0.0),
                justification=item.get("justification", ""),
                impact_metrics=item.get("impact_metrics", {}),
                normalized_scores=lokasi_match.get("normalized_scores", {}),
                kecocokan_percentage=item.get("kecocokan_percentage", 0.0),
                poin_unggul=item.get("poin_unggul", []),
                poin_perhatian=item.get("poin_perhatian", []),
                estimasi_biaya_konstruksi=item.get("estimasi_biaya_konstruksi", ""),
                estimasi_waktu_pengembangan=item.get("estimasi_waktu_pengembangan", ""),
                luasan_bangunan=item.get("luasan_bangunan", ""),
            )

            recommendations.append(rec)
            firestore_record["recommendations"].append(rec.dict())

        # Simpan agregasi hasil ke koleksi "NusaCari"
        try:
            db.collection("NusaCari").add(firestore_record)
            logger.info("✅ Hasil rekomendasi berhasil disimpan ke koleksi 'NusaCari'.")
        except Exception as e:
            logger.error(f"❌ Gagal menyimpan ke Firestore: {e}")

        processing_steps.append("Hasil rekomendasi disimpan ke Firestore.")

        return RecommendationResponse(
            recommendations=recommendations,
            methodology=firestore_record["methodology"],
            analysis_timestamp=firestore_record["analysis_timestamp"],
            processing_steps=processing_steps,
        )

    except Exception as e:
        logger.error(f"Kesalahan dalam proses rekomendasi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Kesalahan proses: {str(e)}")


class GraphData(BaseModel):
    # x‐axis (tahun)
    x: List[int]
    # y‐axis (nilai metrik)
    y: List[float]


class FinancialDataResponse(BaseModel):
    # Tiap metrik sekarang berupa GraphData
    pendapatan: GraphData
    biaya: GraphData
    gross_profit: GraphData


@app.get("/financial-data", response_model=FinancialDataResponse)
async def get_financial_data():
    """
    Mengambil data rekomendasi dari Firestore ('NusaCari'),
    kemudian memanggil GenAI untuk memproyeksikan:
      - pendapatan (total per tahun)
      - biaya (asumsi 60% dari pendapatan)
      - gross_profit = pendapatan - biaya
    untuk tahun 2026–2031.
    Output JSON tetap “murni” (tanpa penjelasan tambahan).
    """
    if not db or not genai_client:
        raise HTTPException(status_code=500, detail="Layanan belum siap")

    try:
        # 1) Ambil dokumen terbaru dari Firestore
        docs = (
            db.collection("NusaCari")
            .order_by("analysis_timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        latest = None
        for d in docs:
            latest = d.to_dict()
            break

        if not latest:
            raise HTTPException(status_code=404, detail="Tidak ada data di 'NusaCari'")

        recs = latest.get("recommendations", [])
        if not recs:
            raise HTTPException(status_code=404, detail="Dokumen 'NusaCari' kosong")

        # 2) Susun input sederhana untuk LLM
        simple_recs = []
        for r in recs:
            simple_recs.append(
                {
                    "kecamatan": r.get("kecamatan"),
                    "population_served": r.get("population_served"),
                    "healthcare_gap_filled": r.get("healthcare_gap_filled"),
                    "impact_metrics": r.get("impact_metrics", {}),
                }
            )

        # 3) Buat prompt
        prompt = f"""
Kamu adalah asisten perencanaan finansial rumah sakit. Semua output harus dalam Bahasa Indonesia,
murni JSON tanpa penjelasan tambahan.

Berikut data rekomendasi lokasi rumah sakit (array JSON) dengan metrik:
{json.dumps(simple_recs)}

Berdasarkan data di atas, proyeksikan untuk tahun 2026 hingga 2031:
1. “pendapatan”: total pendapatan tahunan dari kategori rawat_inap, rawat_jalan, layanan_unggulan.
   - Untuk setiap tahun, jumlahkan nilai pada impact_metrics.rawat_inap[tahun],
     impact_metrics.rawat_jalan[tahun], dan impact_metrics.layanan_unggulan[tahun]
     di semua kecamatan.
2. “biaya”: estimasi total biaya tahunan (asumsikan 60% dari pendapatan setiap tahun sebagai biaya operasional).
3. “gross_profit”: pendapatan dikurangi biaya untuk tiap tahun.

Keluaran persis dalam format:
{{
  "years": [2026,2027,2028,2029,2030,2031],
  "pendapatan": [<angka2026>, <angka2027>, ..., <angka2031>],
  "biaya": [<angka2026>, <angka2027>, ..., <angka2031>],
  "gross_profit": [<angka2026>, <angka2027>, ..., <angka2031>]
}}
"""

        # 4) Panggil GenAI
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
            },
            contents=prompt,
        )

        # **Logging: tampilkan isi mentah LLM supaya kita bisa inspect jika error**
        raw_text = response.text or ""
        logger.info(f"Hasil GenAI (mentah):\n{raw_text}")

        # 5) Jika response.text kosong, kembalikan error yang lebih jelas
        if not raw_text.strip():
            raise HTTPException(
                status_code=500, detail="LLM mengembalikan teks kosong."
            )

        # 6) Parse JSON menggunakan JSONOutputParser, bukan langsung json.loads
        parsed = json_parser.parse(raw_text)
        if "error" in parsed:
            # Jika JSONOutputParser mem‐report kesalahan, tangani di sini
            raise HTTPException(
                status_code=500, detail=f"Error parsing JSON LLM: {parsed['error']}"
            )

        # 7) Extract hasil
        years = parsed.get("years", [])
        pendapatan_list = parsed.get("pendapatan", [])
        biaya_list = parsed.get("biaya", [])
        gross_list = parsed.get("gross_profit", [])

        # Validasi jumlah elemen: harus 6 (2026–2031)
        if not (
            isinstance(years, list)
            and len(years) == 6
            and isinstance(pendapatan_list, list)
            and len(pendapatan_list) == 6
            and isinstance(biaya_list, list)
            and len(biaya_list) == 6
            and isinstance(gross_list, list)
            and len(gross_list) == 6
        ):
            raise HTTPException(
                status_code=500,
                detail=f"Output LLM tidak sesuai format/tahun. Diterima: years={years}, "
                f"pendapatan={pendapatan_list}, biaya={biaya_list}, gross_profit={gross_list}",
            )

        # 8) Bungkus masing‐masing metrik ke GraphData
        pendapatan_graph = {"x": years, "y": pendapatan_list}
        biaya_graph = {"x": years, "y": biaya_list}
        gross_graph = {"x": years, "y": gross_list}

        return FinancialDataResponse(
            pendapatan=pendapatan_graph, biaya=biaya_graph, gross_profit=gross_graph
        )

    except HTTPException:
        # biarkan HTTPException dilempar‐ulang
        raise
    except Exception as e:
        # Jika terjadi exception lain, log stack trace dan return 500
        logger.error(f"Error di /financial-data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Kesalahan server: {str(e)}")
