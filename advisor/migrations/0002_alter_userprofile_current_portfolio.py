# Generated by Django 4.2.5 on 2025-01-25 08:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('advisor', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofile',
            name='current_portfolio',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
