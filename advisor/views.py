from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from .forms import RegisterForm, UserProfileForm, FinancialGoalForm
from .models import UserProfile, CapturedResult
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import urllib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries
import os

API_KEY = 'NBWOL7M2GDDH723E'

def fetch_stock_data(ticker):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data = data.sort_index(ascending=True)
    data = data.dropna()
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['4. close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_and_train_lstm(X_train, y_train, model_filename):
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        model.save(model_filename)
    return model

def predict_best_investment(goal, investment, risk_level):
    tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    best_ticker = None
    best_predicted_value = -np.inf
    best_predicted_price = None
    last_close_price = None

    for ticker in tickers:
        try:
            data = fetch_stock_data(ticker)
            X, y, scaler = preprocess_data(data)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_filename = f'models/{ticker}_lstm_model.h5'
            model = build_and_train_lstm(X_train, y_train, model_filename)

            inputs = data['4. close'].values[-60:].reshape(-1, 1)
            inputs = scaler.transform(inputs)

            X_future = np.array([inputs])
            X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

            predicted_price_scaled = model.predict(X_future)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0, 0]

            last_close = data['4. close'].iloc[-1]
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
        investment = request.POST.get('investment')
        risk_level = request.POST.get('risk_level', 'medium')

        if not investment:
            return render(request, 'advisor/ai_in_finance.html', {'ai_suggestion': 'Please provide a valid investment amount.'})

        try:
            investment = float(investment)
        except ValueError:
            return render(request, 'advisor/ai_in_finance.html', {'ai_suggestion': 'Please provide a valid investment amount.'})

        best_ticker, last_close_price, best_predicted_price, best_predicted_value = predict_best_investment(goal, investment, risk_level)

        if best_ticker and last_close_price and best_predicted_price and best_predicted_value:
            suggestion = f"Based on your goal of {goal} and your investment of ₹{investment}, we suggest investing in {best_ticker}. Last closing price: ₹{last_close_price:.2f}. Predicted price for next day: ₹{best_predicted_price:.2f}. Predicted value of your investment: ₹{best_predicted_value:.2f}."
        else:
            suggestion = "Unable to provide a suggestion at this time. Please try again later."
            return render(request, 'advisor/ai_in_finance.html', {'ai_suggestion': suggestion})

        if best_ticker:
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
        else:
            uri = None

        # Only create a CapturedResult if best_ticker is valid
        if best_ticker and 'capture' in request.POST:
            CapturedResult.objects.create(
                user=request.user,
                ticker=best_ticker,
                last_close_price=last_close_price,
                predicted_price=best_predicted_price,
                predicted_value=best_predicted_value,
                goal=goal,
                investment=investment,
                risk_level=risk_level
            )

        return render(request, 'advisor/ai_in_finance.html', {'ai_suggestion': suggestion, 'ai_graph': uri})
    return render(request, 'advisor/ai_in_finance.html')


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
            predicted_price = model.predict(pd.DataFrame([[last_close]], columns=['Previous Close']))[0]
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

            if 'capture' in request.POST:
                CapturedResult.objects.create(
                    user=request.user,
                    ticker=ticker,
                    last_close_price=last_close,
                    predicted_price=predicted_price,
                    predicted_value=predicted_value,
                    goal=goal,
                    investment=investment,
                    risk_level='N/A'  # Assuming risk level is not applicable here
                )
            
            return render(request, 'advisor/financial_goal.html', {'form': form, 'suggestion': suggestion, 'graph': uri})
        else:
            return render(request, 'advisor/financial_goal.html', {'form': form})
    else:
        form = FinancialGoalForm()
    return render(request, 'advisor/financial_goal.html', {'form': form})
CapturedResult

def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login')

    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        return redirect('create_profile')

    # Fetch captured results for the logged-in user
    captured_results = CapturedResult.objects.filter(user=request.user)

    # Handle delete request
    if request.method == 'POST' and 'delete_id' in request.POST:
        delete_id = request.POST.get('delete_id')
        result_to_delete = get_object_or_404(CapturedResult, id=delete_id, user=request.user)
        result_to_delete.delete()
        messages.success(request, "Result deleted successfully!")
        return redirect('dashboard')

    # Prepare data to display
    results_data = []
    for result in captured_results:
        results_data.append({
            'id': result.id,
            'ticker': result.ticker,
            'last_close_price': result.last_close_price,
            'predicted_price': result.predicted_price,
            'predicted_value': result.predicted_value,
            'goal': result.goal,
            'investment': result.investment,
            'risk_level': result.risk_level,
        })

    return render(
        request,
        'advisor/dashboard.html',
        {
            'profile': user_profile,
            'results': results_data,
        }
    )

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
            predicted_price = model.predict(pd.DataFrame([[last_close]], columns=['Previous Close']))[0]
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

            if 'capture' in request.POST:
                CapturedResult.objects.create(
                    user=request.user,
                    ticker=ticker,
                    last_close_price=last_close,
                    predicted_price=predicted_price,
                    predicted_value=predicted_value,
                    goal=goal,
                    investment=investment,
                    risk_level='N/A'  # Assuming risk level is not applicable here
                )
            
            return render(request, 'advisor/financial_goal.html', {'form': form, 'suggestion': suggestion, 'graph': uri})
    else:
        form = FinancialGoalForm()
    return render(request, 'advisor/financial_goal.html', {'form': form})
