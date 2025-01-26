from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    financial_goals = models.TextField()
    risk_tolerance = models.CharField(max_length=50)
    current_portfolio = models.JSONField(null=True, blank=True)

class FinancialData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ticker = models.CharField(max_length=10)
    date = models.DateField()
    close_price = models.FloatField()
    volume = models.IntegerField()
    sma_50 = models.FloatField()
    sma_200 = models.FloatField()
    percent_change = models.FloatField()

class CapturedResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ticker = models.CharField(max_length=10)
    last_close_price = models.FloatField()
    predicted_price = models.FloatField()
    predicted_value = models.FloatField()
    goal = models.CharField(max_length=255)
    investment = models.FloatField()
    risk_level = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.ticker}"
