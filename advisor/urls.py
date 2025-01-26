# advisor/urls.py
from django.urls import path
from . import views
from django.views.generic.base import RedirectView

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('', RedirectView.as_view(url='login', permanent=False), name='home'),
    path('login/', views.login_view, name='login'),
    path('resources/', views.educational_resources, name='educational_resources'),
    path('logout/', views.logout_view, name='logout'),  # Add a logout view if needed
    path('register/', views.register_view, name='register'),
    path("datasetanlysis/", views.datasetanlysis, name="datasetanlysis"),
    path('financial_goal/', views.financial_goal_view, name='financial_goal'),
    path('ai_in_finance/', views.ai_in_finance_view, name='ai_in_finance'),
    
]
