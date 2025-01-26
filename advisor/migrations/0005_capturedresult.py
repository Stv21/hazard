# Generated by Django 4.2.5 on 2025-01-25 22:06

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('advisor', '0004_financialdata_delete_incomedata'),
    ]

    operations = [
        migrations.CreateModel(
            name='CapturedResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ticker', models.CharField(max_length=10)),
                ('last_close_price', models.FloatField()),
                ('predicted_price', models.FloatField()),
                ('predicted_value', models.FloatField()),
                ('goal', models.CharField(max_length=255)),
                ('investment', models.FloatField()),
                ('risk_level', models.CharField(max_length=50)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
