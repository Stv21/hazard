# advisor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('login/', views.login_view, name='login'),
    path('resources/', views.educational_resources, name='educational_resources'),
    path('logout/', views.logout_view, name='logout'),  # Add a logout view if needed
    path('register/', views.register_view, name='register'),
    path("datasetanlysis/", views.datasetanlysis, name="datasetanlysis"),
    path('financial_goal/', views.financial_goal_view, name='financial_goal'),
    
]
