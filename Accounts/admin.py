from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import Group
from .models import Doctor



class doctorAdmin(UserAdmin) :
    list_display = ("username", )
    search_fields = ("username", )
    
    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()

admin.site.register(Doctor, doctorAdmin)
admin.site.unregister(Group)