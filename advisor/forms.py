# advisor/forms.py
from django import forms
from django.contrib.auth.models import User
from .models import UserProfile

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match")

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['financial_goals', 'risk_tolerance', 'current_portfolio']

# advisor/forms.py
from django import forms

class FinancialGoalForm(forms.Form):
    TICKER_CHOICES = [
        ('AAPL', 'Apple Inc.'),
        ('MSFT', 'Microsoft Corporation'),
        ('GOOGL', 'Alphabet Inc.'),
        ('AMZN', 'Amazon.com Inc.'),
        ('TSLA', 'Tesla Inc.'),
        ('FB', 'Meta Platforms Inc.'),
        ('NFLX', 'Netflix Inc.'),
        ('NVDA', 'NVIDIA Corporation'),
        ('BABA', 'Alibaba Group Holding Limited'),
        ('V', 'Visa Inc.'),
    ]
    
    ticker = forms.ChoiceField(label='Ticker Symbol', choices=TICKER_CHOICES)
    investment = forms.FloatField(label='Investment Amount (₹)')
    goal = forms.CharField(label='Goal', max_length=100)

class IncomeStatusForm(forms.Form):
    income = forms.IntegerField(label='Income', widget=forms.NumberInput(attrs={'type': 'range', 'min': '0', 'max': '1000000', 'step': '1000'}))
    goal = forms.CharField(label='Goal', max_length=100)
