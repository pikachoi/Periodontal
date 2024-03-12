from django.urls import path
from django.conf import settings
from django.contrib import admin
from django.urls.conf import include
from django.conf.urls.static import static


from django.views.static import serve
from django.urls import re_path

urlpatterns = [
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),

    path("admin/", admin.site.urls),
    path("", include("Diagnosis.urls")),
    path("Accounts/",include("Accounts.urls")),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)