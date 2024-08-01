
from django.urls import path
from .views import *

urlpatterns = [
    path("", diagnosis_single, name ='diagnosis_single'),
    path("time_series/", diagnosis_time_series, name ='diagnosis_time_series'),
    path("clinic/", clinic, name ='clinic'),
    path('save_diagnosis_result/', save_diagnosis_result, name='save_diagnosis_result'),
    path('results/', result_list, name='result_list'),
    path('results/<int:pk>/', result_detail, name='result_detail'),  # 변경: <int:result_id> -> <int:pk>
    path('results/<int:pk>/delete/', delete_result, name='delete_result'),  # 변경: <int:result_id> -> <int:pk>
    path('results/<int:result_id>/update_consent/', update_consent_status, name='update_consent_status'),
    path('results/<int:result_id>/reset_consent/', reset_consent_status, name='reset_consent_status'),
    path('chart/', chart, name='chart'),  # 새로운 URL 패턴 추가
]
