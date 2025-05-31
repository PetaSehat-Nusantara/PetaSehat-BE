import os
import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import googlemaps
from geopy.geocoders import Nominatim
from google import genai

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  
# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="Hospital Location Recommender (Streaming)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Either explicitly list your front‐end’s origin(s), e.g.:
        "http://localhost:3000",   # if your React/Vue front‐end is running on this port
        "http://127.0.0.1:3000",
        # Or use ["*"] if you want to allow every origin (not recommended for production,
        # but okay for local dev).
        # "*"
    ],
    allow_methods=["GET", "POST", "OPTIONS"],         # <-- MUST include "OPTIONS"
    allow_headers=["*"],                              # <-- allow any custom headers
    allow_credentials=True,                           # <-- if you send cookies/auth tokens
)

# ----------------------------
# Pydantic Models (for reference)
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

class RecommendationResponse(BaseModel):
    recommendations: List[LocationRecommendation]
    methodology: str
    analysis_timestamp: str
    processing_steps: List[str]
# (Other models like LocationRecommendation / RecommendationResponse
# are no longer used directly for the streaming endpoint.)

# ----------------------------
# JSONOutputParser (unchanged)
# ----------------------------
class JSONOutputParser:
    def parse(self, text: str) -> dict:
        try:
            t = text.strip()
            if t.startswith("```json"):
                t = t[len("```json"):].strip()
            if t.endswith("```"):
                t = t[:-3].strip()
            return json.loads(t)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}\nText was:\n{text}")
            return {"error": f"Failed to parse JSON: {e}"}

json_parser = JSONOutputParser()

# ----------------------------
# Global Service Placeholders
# ----------------------------
db = None
gmaps = None
geolocator = None
genai_client = None

# ----------------------------
# Helper Functions (unchanged from before)
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
        logger.error(f"Error getting Firestore schema: {e}")
        return {}

async def collect_firestore_data_func(retrieval_plan: Dict[str, Any], target_province: str) -> Dict[str, Any]:
    try:
        plan = retrieval_plan if isinstance(retrieval_plan, dict) else json.loads(retrieval_plan)
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
        logger.error(f"Error collecting Firestore data: {e}")
        return {"error": str(e)}

def normalize_and_scale_func(locations_data: List[Dict[str, Any]], priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
    if not locations_data:
        return []

    metrics = ["populationDensity", "healthcareGap", "economicViability", "demographicMatch", "accessibility"]
    # Compute min/max for each metric
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

async def get_gmaps_coordinates_func(kecamatan: str, city: str, province: str) -> Dict[str, Any]:
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
                "formatted_address": f"Approximate location: {q}",
                "gmaps_url": None,
            }
    except Exception as e:
        logger.error(f"Error getting coordinates for {kecamatan}: {e}")
        return {
            "lat": -6.2088,
            "lng": 106.8456,
            "place_id": None,
            "formatted_address": f"Error fetching location: {q}",
            "gmaps_url": None,
        }

# ----------------------------
# Startup Event (initialize Firebase, GMaps, GenAI)
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global db, gmaps, geolocator, genai_client

    # 1) Initialize Firebase
    FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")
    if not FIREBASE_CRED_PATH or not os.path.exists(FIREBASE_CRED_PATH):
        raise RuntimeError(f"Missing or invalid FIREBASE_CRED_PATH: {FIREBASE_CRED_PATH!r}")
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        _ = list(db.collections())
        logger.info("✅ Firebase connected.")
    except Exception as e:
        logger.error(f"❌ Firebase init failed: {e}")
        raise RuntimeError(f"Failed to initialize Firebase: {e}")

    # 2) Initialize Google Maps
    GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
    if not GMAPS_API_KEY:
        raise RuntimeError("Missing GMAPS_API_KEY")
    try:
        gmaps = googlemaps.Client(key=GMAPS_API_KEY)
        test = gmaps.geocode("Jakarta, Indonesia")
        if not test:
            raise Exception("Empty geocode result")
        logger.info("✅ Google Maps API validated.")
    except Exception as e:
        logger.error(f"❌ GMaps init failed: {e}")
        raise RuntimeError(f"Failed to initialize Google Maps API: {e}")

    # 3) (Optional) Initialize Nominatim
    geolocator = Nominatim(user_agent="hospital_recommender")

    # 4) Initialize GenAI Client
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY for GenAI")
    try:
        genai_client = genai.Client(api_key=GOOGLE_API_KEY)
        smoke = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello, GenAI! If you can respond, please say 'ready'."
        )
        logger.info(f"✅ GenAI smoke test: {smoke.text[:20]!r}…")
    except Exception as e:
        logger.error(f"❌ GenAI init failed: {e}")
        raise RuntimeError(f"Failed to initialize GenAI: {e}")

@app.post("/recommend-hospital-location", response_model=RecommendationResponse)
async def recommend_hospital_location(request: HospitalRequest):
    """
    Generate hospital location recommendations based on user criteria,
    directly using genai.Client for each prompt step (no LangChain).
    """
    processing_steps: List[str] = []

    # Ensure all services are initialized
    if not all([db, gmaps, genai_client]):
        raise HTTPException(
            status_code=500,
            detail="Recommender services not fully initialized. Check startup logs."
        )

    try:
        # -------------------
        # Step 1: Analyze Requirements
        # -------------------
        processing_steps.append("Analyzing user requirements…")
        logger.info("Analyzing user requirements…")

        # Create prompt for requirements
        user_input_json = request.json()
        requirements_prompt = f"""
You are a hospital-site planning assistant.

Given the user's JSON requirements:
{user_input_json}

Identify and rank the priority factors (populationDensity, healthcareGap, economicViability, demographicMatch, accessibility).
Return a JSON object EXACTLY in this form (no extra keys, no commentary):

{{
  "priorityFactors": {{
    "populationDensity": <weight>,
    "healthcareGap": <weight>,
    "economicViability": <weight>,
    "demographicMatch": <weight>,
    "accessibility": <weight>
  }},
  "filters": {{
    "province": <string>,
    "facilityType": <string>
  }}
}}
"""

        # Call GenAI
        req_response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=requirements_prompt
        )
        requirements_analysis = json_parser.parse(req_response.text)

        # -------------------
        # Step 2: Get Firestore Schema and Plan Retrieval
        # -------------------
        processing_steps.append("Analyzing Firestore schema…")
        logger.info("Analyzing Firestore schema…")
        schema_dict = await get_firestore_schema_func()
        schema_json = json.dumps(schema_dict)

        # Create prompt for retrieval plan
        retrieval_plan_prompt = f"""
You have Firestore schema as JSON:
{schema_json}

And you have extracted user priorities/filters as:
{json.dumps(requirements_analysis)}

Generate a Firestore retrieval plan in JSON with this exact structure:

{{
  "collectionsToQuery": [
    {{
      "collection": "<collection_name>",
      "fields": [ "<field1>", "<field2>", … ],
      "filters": {{ "province": <string>, "facilityType": <string> }}
    }},
    …
  ]
}}
"""
        # Call GenAI
        retrieval_plan_resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=retrieval_plan_prompt
        )
        retrieval_plan = json_parser.parse(retrieval_plan_resp.text)

        # -------------------
        # Step 3: Collect Data from Firestore
        # -------------------
        processing_steps.append("Collecting data from Firestore…")
        logger.info("Collecting data from Firestore…")
        collected_data = await collect_firestore_data_func(
            retrieval_plan,
            request.lokasiLahan.province
        )

        # -------------------
        # Step 4: Extract Numeric Datapoints
        # -------------------
        processing_steps.append("Extracting numerical datapoints…")
        logger.info("Extracting numerical datapoints…")

        extracted_prompt = f"""
You have raw Firestore data (JSON):
{json.dumps(collected_data)}

And you have the user's priority analysis (JSON):
{json.dumps(requirements_analysis)}

for province: {request.lokasiLahan.province}

Extract numeric metrics for each kecamatan into this JSON format EXACTLY:

{{
  "locations": [
    {{
      "kecamatan": "<kecamatan_name>",
      "populationDensity": <number>,
      "healthcareGap": <number>,
      "economicViability": <number>,
      "demographicMatch": <number>,
      "accessibility": <number>
    }},
    …
  ]
}}
"""
        extraction_resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=extracted_prompt
        )
        extracted_data = json_parser.parse(extraction_resp.text)

        # -------------------
        # Step 5: Normalize and Scale Scores
        # -------------------
        processing_steps.append("Normalizing and scaling scores…")
        logger.info("Normalizing and scaling scores…")

        priority_weights = requirements_analysis.get("priorityFactors", {})
        scored_locations = normalize_and_scale_func(
            extracted_data.get("locations", []),
            priority_weights
        )

        # -------------------
        # Step 6: Impact Analysis (Top 5)
        # -------------------
        processing_steps.append("Analyzing potential impact…")
        logger.info("Analyzing potential impact…")

        top_5 = scored_locations[:5]
        top_5_json = json.dumps(top_5)
        hospital_specs_json = request.informasiUmum.json()

        impact_prompt = f"""
Given the top-5 scored locations as JSON:
{top_5_json}

And the proposed hospital specifications (JSON):
{hospital_specs_json}

For each location, estimate:
  - population_served
  - healthcare_gap_filled
  - justification (short explanation)
  - any additional impact_metrics

Return EXACTLY a JSON under key "impact_analysis" with this structure:

{{
  "impact_analysis": [
    {{
      "kecamatan": "<name>",
      "population_served": <number>,
      "healthcare_gap_filled": <number>,
      "justification": "<text>",
      "impact_metrics": {{ … }}
    }},
    …
  ]
}}
"""
        impact_resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=impact_prompt
        )
        impact_analysis = json_parser.parse(impact_resp.text)

        # -------------------
        # Step 7: Build Final Recommendations with GMaps Coordinates
        # -------------------
        processing_steps.append("Getting location coordinates…")
        logger.info("Getting location coordinates…")

        recommendations: List[LocationRecommendation] = []
        impact_data_map = {
            item.get("kecamatan"): item
            for item in impact_analysis.get("impact_analysis", [])
        }

        top_3 = scored_locations[:3]
        for loc in top_3:
            kecamatan = loc.get("kecamatan", "Unknown")
            coords = await get_gmaps_coordinates_func(
                kecamatan,
                request.lokasiLahan.city,
                request.lokasiLahan.province
            )
            impact_info = impact_data_map.get(kecamatan, {})

            recommendation = LocationRecommendation(
                kecamatan=kecamatan,
                gmaps_coordinates={"lat": coords["lat"], "lng": coords["lng"]},
                gmaps_place_id=coords.get("place_id"),
                composite_score=loc["composite_score"],
                rank=loc["rank"],
                population_served=impact_info.get(
                    "population_served",
                    loc.get("estimatedPopulationServed", 0)
                ),
                healthcare_gap_filled=impact_info.get(
                    "healthcare_gap_filled",
                    loc.get("currentHealthcareGap", 0)
                ),
                justification=impact_info.get(
                    "justification",
                    f"High composite score of {loc['composite_score']} based on demographic and healthcare factors."
                ),
                impact_metrics=impact_info.get("impact_metrics", {}),
                normalized_scores=loc.get("normalized_scores", {})
            )
            recommendations.append(recommendation)

        processing_steps.append("Recommendations generated successfully!")

        return RecommendationResponse(
            recommendations=recommendations,
            methodology=(
                "Multi-factor analysis using demographic match, healthcare gap, "
                "economic viability, and accessibility scores with LLM-driven insights."
            ),
            analysis_timestamp=datetime.now().isoformat(),
            processing_steps=processing_steps
        )

    except Exception as e:
        logger.error(f"Error in recommendation process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# ----------------------------
# Streaming Endpoint
# ----------------------------
@app.post("/recommend-hospital-location-stream")
async def recommend_hospital_location_stream(request: Request) -> StreamingResponse:
    """
    Stream incremental JSON updates back to the client (newline-delimited JSON).
    Each yielded line is one JSON object that includes:
      { "step": "<description>", "data": <any> }
    Or, in the final message, returns the “recommendations” array and metadata.
    """
    # Validate that services are ready
    if not all([db, gmaps, genai_client]):
        raise HTTPException(
            status_code=500,
            detail="Recommender services not fully initialized. Check startup logs."
        )

    # Load and validate the incoming JSON body (HospitalRequest)
    try:
        body = await request.json()
        req_obj = HospitalRequest.parse_obj(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        """Yields newline-delimited JSON for each stage."""

        # Stage 1: Analyze Requirements
        step1_desc = "Analyzing user requirements"
        yield (json.dumps({"step": step1_desc}) + "\n").encode("utf-8")
        user_input_json = json.dumps(body)  # using the raw JSON body as a string
        requirements_prompt = f"""
You are a hospital‐site planning assistant.

Given the user's JSON requirements:
{user_input_json}

Identify and rank the priority factors (populationDensity, healthcareGap, economicViability, demographicMatch, accessibility).
Return a JSON object EXACTLY in this form (no extra keys):

{{
  "priorityFactors": {{
    "populationDensity": <weight>,
    "healthcareGap": <weight>,
    "economicViability": <weight>,
    "demographicMatch": <weight>,
    "accessibility": <weight>
  }},
  "filters": {{
    "province": "<string>",
    "facilityType": "<string>"
  }}
}}
"""
        try:
            resp1 = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=requirements_prompt
            )
            requirements_analysis = json_parser.parse(resp1.text)
            yield (
                json.dumps({
                    "step": "Requirements analysis completed",
                    "data": requirements_analysis
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error in requirements analysis",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return  # Stop streaming on error

        # Stage 2: Plan Firestore Retrieval
        step2_desc = "Planning Firestore retrieval"
        yield (json.dumps({"step": step2_desc}) + "\n").encode("utf-8")
        try:
            schema_dict = await get_firestore_schema_func()
            schema_json = json.dumps(schema_dict)
            retrieval_plan_prompt = f"""
You have Firestore schema as JSON:
{schema_json}

And you have extracted user priorities/filters as:
{json.dumps(requirements_analysis)}

Generate a Firestore retrieval plan in JSON with EXACT structure:

{{
  "collectionsToQuery": [
    {{
      "collection": "<collection_name>",
      "fields": [ "<field1>", "<field2>", … ],
      "filters": {{ "province": "{req_obj.lokasiLahan.province}", "facilityType": "{req_obj.informasiUmum.facilityType}" }}
    }},
    …
  ]
}}
"""
            resp2 = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=retrieval_plan_prompt
            )
            retrieval_plan = json_parser.parse(resp2.text)
            yield (
                json.dumps({
                    "step": "Retrieval plan generated",
                    "data": retrieval_plan
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error in retrieval planning",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return

        # Stage 3: Collect Data from Firestore
        step3_desc = "Collecting data from Firestore"
        yield (json.dumps({"step": step3_desc}) + "\n").encode("utf-8")
        try:
            collected_data = await collect_firestore_data_func(
                retrieval_plan,
                req_obj.lokasiLahan.province
            )
            yield (
                json.dumps({
                    "step": "Data collected",
                    "sample": {  # send only a small “preview” of the data to avoid huge payloads
                        col: docs[:2] for col, docs in collected_data.items()
                    }
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error collecting data",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return

        # Stage 4: Extract Numeric Datapoints
        step4_desc = "Extracting numeric datapoints"
        yield (json.dumps({"step": step4_desc}) + "\n").encode("utf-8")
        try:
            extracted_prompt = f"""
You have raw Firestore data (JSON):
{json.dumps(collected_data)}

And you have the user's priority analysis (JSON):
{json.dumps(requirements_analysis)}

for province: {req_obj.lokasiLahan.province}

Extract numeric metrics for each kecamatan into this JSON format EXACTLY:

{{
  "locations": [
    {{
      "kecamatan": "<kecamatan_name>",
      "populationDensity": <number>,
      "healthcareGap": <number>,
      "economicViability": <number>,
      "demographicMatch": <number>,
      "accessibility": <number>
    }},
    …
  ]
}}
"""
            resp4 = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=extracted_prompt
            )
            extracted_data = json_parser.parse(resp4.text)
            yield (
                json.dumps({
                    "step": "Extraction complete",
                    "data": {"locations_count": len(extracted_data.get("locations", []))}
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error extracting numeric datapoints",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return

        # Stage 5: Normalize & Score
        step5_desc = "Normalizing and scoring locations"
        yield (json.dumps({"step": step5_desc}) + "\n").encode("utf-8")
        try:
            priority_weights = requirements_analysis.get("priorityFactors", {})
            scored_locations = normalize_and_scale_func(
                extracted_data.get("locations", []),
                priority_weights
            )
            yield (
                json.dumps({
                    "step": "Normalization & scoring done",
                    "data": {
                        "top_scores": [
                            { "kecamatan": loc["kecamatan"], "score": loc["composite_score"] }
                            for loc in scored_locations[:3]
                        ]
                    }
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error normalizing/scoring",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return

        # Stage 6: Impact Analysis (Top 5)
        step6_desc = "Performing impact analysis"
        yield (json.dumps({"step": step6_desc}) + "\n").encode("utf-8")
        try:
            top_5 = scored_locations[:5]
            top_5_json = json.dumps(top_5)
            hospital_specs_json = req_obj.informasiUmum.json()

            impact_prompt = f"""
Given the top-5 scored locations as JSON:
{top_5_json}

And the proposed hospital specifications (JSON):
{hospital_specs_json}

For each location, estimate:
  - population_served
  - healthcare_gap_filled
  - justification (short explanation)
  - any additional impact_metrics

Return EXACTLY a JSON under key "impact_analysis" with this structure:

{{
  "impact_analysis": [
    {{
      "kecamatan": "<name>",
      "population_served": <number>,
      "healthcare_gap_filled": <number>,
      "justification": "<text>",
      "impact_metrics": {{ … }}
    }},
    …
  ]
}}
"""
            resp6 = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=impact_prompt
            )
            impact_analysis = json_parser.parse(resp6.text)
            yield (
                json.dumps({
                    "step": "Impact analysis done",
                    "data": {"impact_count": len(impact_analysis.get("impact_analysis", []))}
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error in impact analysis",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return

        # Stage 7: Build Final 3 Recommendations
        step7_desc = "Fetching coordinates and building recommendations"
        yield (json.dumps({"step": step7_desc}) + "\n").encode("utf-8")
        try:
            recommendations: List[Dict[str, Any]] = []
            impact_map = { item["kecamatan"]: item for item in impact_analysis.get("impact_analysis", []) }
            top_3 = scored_locations[:3]

            for loc in top_3:
                kec = loc["kecamatan"]
                coords = await get_gmaps_coordinates_func(
                    kec, req_obj.lokasiLahan.city, req_obj.lokasiLahan.province
                )
                imp = impact_map.get(kec, {})
                rec = {
                    "kecamatan": kec,
                    "gmaps_coordinates": {"lat": coords["lat"], "lng": coords["lng"]},
                    "gmaps_place_id": coords.get("place_id"),
                    "composite_score": loc["composite_score"],
                    "rank": loc["rank"],
                    "population_served": imp.get("population_served", 0),
                    "healthcare_gap_filled": imp.get("healthcare_gap_filled", 0),
                    "justification": imp.get(
                        "justification",
                        f"High composite score of {loc['composite_score']}"
                    ),
                    "impact_metrics": imp.get("impact_metrics", {}),
                    "normalized_scores": loc.get("normalized_scores", {}),
                }
                recommendations.append(rec)

            yield (
                json.dumps({
                    "step": "Final recommendations ready",
                    "recommendations": recommendations,
                    "methodology": (
                        "Multi-factor analysis using demographic match, healthcare gap, "
                        "economic viability, and accessibility scores with LLM-driven insights."
                    ),
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
            ).encode("utf-8")
        except Exception as e:
            yield (
                json.dumps({
                    "step": "Error building final recommendations",
                    "error": str(e)
                }) + "\n"
            ).encode("utf-8")
            return

        # Done: close the generator
        return

    # Return a StreamingResponse with “application/x-ndjson” (newline-delimited JSON)
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
