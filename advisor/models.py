# advisor/models.py
from django.db import models
from django.contrib.auth.models import User  # Import the User model

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    financial_goals = models.TextField()
    risk_tolerance = models.CharField(max_length=50)
    current_portfolio = models.JSONField(null=True, blank=True)  # Allow null and blank values
