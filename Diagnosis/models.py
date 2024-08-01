import os

from django.db import models
from django.utils import timezone
from Accounts.models import Doctor

def upload_imgae_path(instance, filename) :
    make_path = f"{timezone.now().strftime('%Y/%m/%d')}/{Patient.objects.get(idx = instance.patient.idx).patient_name}/{instance.patient.idx}"

    if instance.is_axial :
        return f"{make_path}/axial/{filename}"
    else :
        return f"{make_path}/sagittal/{filename}"

def classifi_result_default_dict() :
    return {"K-TIRADS"      : [],
            "echogenicity"  : [],
            "margin"        : [],
            "calcification" : [],
            "orientation"   : [],
            "shape"         : [],
            "internal"      : [],
            "spongiform"    : []
            }

class Patient(models.Model) :
    idx             = models.AutoField(primary_key = True) 
    doctor          = models.ForeignKey(Doctor, on_delete = models.CASCADE, db_column = "doctor", null = False)
    patient_name    = models.CharField(max_length = 128)
    visit_date      = models.DateTimeField(verbose_name = "visit_date", auto_now = True)
    is_summary      = models.BooleanField(default = True)   # 요약 모달창을 아직 안띄워줘서 tagging 하기 전이면 True, 요약 모달창에서 선택을 끝내면 False

# @admin.register(Patient)
# class MyModelAdmin(admin.ModelAdmin):
#     list_display = ('idx', 'doctor', 'patient_name', 'visit_date', 'is_summary')

class ImagePath(models.Model) :
    idx             = models.AutoField(primary_key = True)
    patient         = models.ForeignKey(Patient, on_delete = models.CASCADE, db_column = "patient", null = False)
    img_path        = models.ImageField(upload_to = upload_imgae_path, max_length=1024, blank=True)  # 업로드 한 이미지
    is_axial        = models.BooleanField(default = True)

class Crop(models.Model) :
    idx             = models.AutoField(primary_key = True)
    img_path        = models.ForeignKey(ImagePath, on_delete = models.CASCADE, db_column="img_path", null = False)
    crop_img_path   = models.ImageField(max_length = 1024, blank = True)         # crop 이미지 한장
    is_nodule       = models.BooleanField(default = False)  # AI이 결절을 찾지 못하면 False # 정상데이터와 결절데이터 구분
    classifi_result = models.JSONField(default = classifi_result_default_dict)



# models.py

from django.db import models

class DiagnosisResult(models.Model):
    patient_age = models.IntegerField(null=True, blank=True)
    max_loss = models.FloatField(null=True, blank=True)
    avg_loss = models.FloatField(null=True, blank=True)
    total_teeth = models.IntegerField(null=True, blank=True)
    grade_a_count = models.IntegerField(null=True, blank=True)
    grade_b_count = models.IntegerField(null=True, blank=True)
    grade_c_count = models.IntegerField(null=True, blank=True)
    visualization_image = models.BinaryField()
    title = models.CharField(max_length=255, null=True, blank=True)
    saved_at = models.DateTimeField(auto_now_add=True)
    consent_status = models.CharField(max_length=10, null=True, blank=True)  # 'agree' or 'disagree'
