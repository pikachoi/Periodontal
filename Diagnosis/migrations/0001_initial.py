# Generated by Django 4.2.1 on 2024-03-05 13:28

import Diagnosis.models
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Patient",
            fields=[
                ("idx", models.AutoField(primary_key=True, serialize=False)),
                ("patient_name", models.CharField(max_length=128)),
                (
                    "visit_date",
                    models.DateTimeField(auto_now=True, verbose_name="visit_date"),
                ),
                ("is_summary", models.BooleanField(default=True)),
                (
                    "doctor",
                    models.ForeignKey(
                        db_column="doctor",
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="ImagePath",
            fields=[
                ("idx", models.AutoField(primary_key=True, serialize=False)),
                (
                    "img_path",
                    models.ImageField(
                        blank=True,
                        max_length=1024,
                        upload_to=Diagnosis.models.upload_imgae_path,
                    ),
                ),
                ("is_axial", models.BooleanField(default=True)),
                (
                    "patient",
                    models.ForeignKey(
                        db_column="patient",
                        on_delete=django.db.models.deletion.CASCADE,
                        to="Diagnosis.patient",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Crop",
            fields=[
                ("idx", models.AutoField(primary_key=True, serialize=False)),
                (
                    "crop_img_path",
                    models.ImageField(blank=True, max_length=1024, upload_to=""),
                ),
                ("is_nodule", models.BooleanField(default=False)),
                (
                    "classifi_result",
                    models.JSONField(
                        default=Diagnosis.models.classifi_result_default_dict
                    ),
                ),
                (
                    "img_path",
                    models.ForeignKey(
                        db_column="img_path",
                        on_delete=django.db.models.deletion.CASCADE,
                        to="Diagnosis.imagepath",
                    ),
                ),
            ],
        ),
    ]
