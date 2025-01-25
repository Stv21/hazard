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
