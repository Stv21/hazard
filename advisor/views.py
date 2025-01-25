# advisor/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import RegisterForm, UserProfileForm
from .models import UserProfile

def dashboard(request):
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        return redirect('create_profile')
    return render(request, 'advisor/dashboard.html', {'profile': user_profile})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
    return render(request, 'advisor/login.html')

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            UserProfile.objects.create(user=user)
            login(request, user)
            return redirect('dashboard')
    else:
        form = RegisterForm()
    return render(request, 'advisor/register.html', {'form': form})

def educational_resources(request):
    resources = [
        {'title': 'Investment Basics', 'content': '...'},
        {'title': 'Understanding Risk', 'content': '...'},
    ]
    return render(request, 'advisor/resources.html', {'resources': resources})

def logout_view(request):
    logout(request)
    return redirect('login')

def create_profile(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST)
        if form.is_valid():
            user_profile = form.save(commit=False)
            user_profile.user = request.user
            user_profile.save()
            return redirect('dashboard')
    else:
        form = UserProfileForm()
    return render(request, 'advisor/create_profile.html', {'form': form})

def datasetanlysis(request):
    return render(request, 'advisor/datasetanalysis.html')
