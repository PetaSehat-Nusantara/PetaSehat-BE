from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class InformasiUmum(BaseModel):
    facilityClass: Literal["Tipe A", "Tipe B", "Tipe C", "Tipe D", "Klinik Utama", "Klinik Pratama", "Puskesmas", "Laboratorium", "Apotek"]
    facilityName: str
    facilityType: str
    maxCapacity: int = Field(..., ge=1)
    minCapacity: int = Field(..., ge=1)
    services: List[str]

class KriteriaDemografi(BaseModel):
    targetDemografiUtama: List[
        str 
    ]
    targetPendapatanPasien: List[
        str
    ]

class KriteriaKeuangan(BaseModel):
    estimasiAnggaranMaksimum: int = Field(..., ge=0)
    estimasiAnggaranMinimum: int = Field(..., ge=0)
    targetROI: int = Field(..., ge=0)
    targetWaktuPembangunan: str # Consider using a more specific type if possible, e.g., Pydantic's timedelta or a custom validator

class LokasiLahan(BaseModel):
    accessibilityPreferences: List[
       str
    ]
    areaPreferences: List[str]
    city: str
    district: str
    environmentalPreferences: List[
        str
    ]
    landPreferences: List[str]
    province: str

class FaskesData(BaseModel):
    informasiUmum: InformasiUmum
    kriteriaDemografi: KriteriaDemografi
    kriteriaKeuangan: KriteriaKeuangan
    lokasiLahan: LokasiLahan

