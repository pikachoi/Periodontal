from django.urls import path
from .views import *
from . import views

urlpatterns = [
    path("", diagnosis_home, name ='diagnosis_home'),
    
]