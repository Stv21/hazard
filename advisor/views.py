# advisor/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import RegisterForm, UserProfileForm, FinancialGoalForm
from .models import UserProfile
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import urllib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

API_KEY = 'NBWOL7M2GDDH723E'

def fetch_stock_data(ticker):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data = data.sort_index(ascending=True)
    data['Return'] = data['4. close'].pct_change()
    data = data.dropna()
    return data

def predict_best_investment(goal, investment, risk_level):
    tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    best_ticker = None
    best_predicted_value = -np.inf
    best_predicted_price = None
    last_close_price = None

    for ticker in tickers:
        try:
            data = fetch_stock_data(ticker)
            data['Previous Close'] = data['4. close'].shift(1)
            data.dropna(inplace=True)
            X = data[['Previous Close']]
            y = data['4. close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            last_close = data['4. close'].iloc[-1]
            predicted_price = model.predict([[last_close]])[0]
            shares = investment / last_close
            predicted_value = shares * predicted_price

            if risk_level == 'high':
                predicted_value *= 1.1  # Increase predicted value by 10% for high risk
            elif risk_level == 'low':
                predicted_value *= 0.9  # Decrease predicted value by 10% for low risk

            if predicted_value > best_predicted_value:
                best_ticker = ticker
                best_predicted_value = predicted_value
                best_predicted_price = predicted_price
                last_close_price = last_close

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return best_ticker, last_close_price, best_predicted_price, best_predicted_value

def ai_in_finance_view(request):
    if request.method == 'POST':
        goal = request.POST.get('goal')
        investment = float(request.POST.get('investment'))
        risk_level = request.POST.get('risk_level', 'medium')
        best_ticker, last_close_price, best_predicted_price, best_predicted_value = predict_best_investment(goal, investment, risk_level)

        suggestion = f"Based on your goal of {goal} and your investment of ₹{investment}, we suggest investing in {best_ticker}. Last closing price: ₹{last_close_price:.2f}. Predicted price for next day: ₹{best_predicted_price:.2f}. Predicted value of your investment: ₹{best_predicted_value:.2f}."

        data = fetch_stock_data(best_ticker)
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['4. close'], label='Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.title(f'{best_ticker} Stock Price')
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'advisor/ai_in_finance.html', {'ai_suggestion': suggestion, 'ai_graph': uri})
    return render(request, 'advisor/ai_in_finance.html')

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

def financial_goal_view(request):
    if request.method == 'POST':
        form = FinancialGoalForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            investment = form.cleaned_data['investment']
            goal = form.cleaned_data['goal']
            
            data = fetch_stock_data(ticker)
            data['Previous Close'] = data['4. close'].shift(1)
            data.dropna(inplace=True)
            X = data[['Previous Close']]
            y = data['4. close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            last_close = data['4. close'].iloc[-1]
            predicted_price = model.predict([[last_close]])[0]
            shares = investment / last_close
            predicted_value = shares * predicted_price
            
            plt.figure(figsize=(10, 5))
            plt.plot(data.index, data['4. close'], label='Close Price')
            plt.plot(data.index, data['Previous Close'], label='Previous Close')
            plt.xlabel('Date')
            plt.ylabel('Price (₹)')
            plt.title(f'{ticker} Stock Price')
            plt.legend()
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)
            
            suggestion = f"Based on your goal of {goal} and your investment of ₹{investment}, we suggest investing in {ticker}. Predicted stock price: ₹{predicted_price:.2f}. Predicted value of your investment: ₹{predicted_value:.2f}."
            
            return render(request, 'advisor/financial_goal.html', {'form': form, 'suggestion': suggestion, 'graph': uri})
    else:
        form = FinancialGoalForm()
    return render(request, 'advisor/financial_goal.html', {'form': form})
