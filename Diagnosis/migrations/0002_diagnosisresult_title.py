# Generated by Django 4.2.1 on 2024-07-15 01:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Diagnosis", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="diagnosisresult",
            name="title",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
