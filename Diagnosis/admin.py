from django.contrib import admin
from .models import Patient, ImagePath, Crop

admin.site.register(Patient)
admin.site.register(ImagePath)
admin.site.register(Crop)
