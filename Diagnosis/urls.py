from django.urls import path
from .views import *

urlpatterns = [
    path("", diagnosis_single, name ='diagnosis_single'),
    path("time_series/", diagnosis_time_series, name ='diagnosis_time_series'),
     path("clinic/", clinic, name ='clinic'),
]