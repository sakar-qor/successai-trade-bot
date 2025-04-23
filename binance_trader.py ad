import os
import ccxt
import pandas as pd
import numpy as np
import logging
import talib as ta
import time
from sklearn.ensemble import GradientBoostingClassifier
from dotenv import load_dotenv
from telegram import Bot
from datetime import datetime

# تحميل المتغيرات السرية
load_dotenv()

class BinanceSmartTrader:
    def __init__(self):
        self.exchange = self._init_exchange()
        self.symbols = ['BTC/USDT', 'ETH/USDT']  # الحد الأقصى الموصى به 5 أزواج
        self.telegram_bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self._setup_logging()
        self.model = self._train_ai_model()
        self.last_request = time.time()
        
    def _init_exchange(self):
        """تهيئة الاتصال مع مراعاة حدود Binance API"""
        return ccxt.binance({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('API_SECRET'),
            'options': {'adjustForTimeDifference': True},
            'enableRateLimit': True,
            'rateLimit': 1000  # 1200 request/min للوضع الآمن
        })
    
    def _setup_logging(self):
        """نظام تسجيل متطور"""
        logging.basicConfig(
            filename='binance_smart.log',
            level=logging.INFO,
            format='%(asctime)s|%(levelname)s|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("=== System Initialized ===")
    
    def _safe_api_call(self, func, *args, **kwargs):
        """إدارة طلبات API بشكل آمن"""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < 0.1:  # 10 requests/second كحد أقصى
            time.sleep(0.1 - elapsed)
        self.last_request = time.time()
        return func(*args, **kwargs)
    
    def _train_ai_model(self):
        """تدريب نموذج متوافق مع البيانات الحديثة"""
        data = self._safe_api_call(self.exchange.fetch_ohlcv, 'BTC/USDT', '5m', limit=500)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = self._apply_advanced_indicators(df)
        features = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'adx']]
        target = (df['close'].pct_change(6).shift(-6) > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=150, max_depth=4)
        model.fit(features.dropna(), target.dropna())
        return model
    
    def _apply_advanced_indicators(self, df):
        """مؤشرات Binance الرسمية"""
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['macd'], df['signal'], _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        return df.dropna()
    
    def _calculate_risk(self, symbol):
        """حساب المخاطر وفق شروط Binance"""
        balance = self._safe_api_call(self.exchange.fetch_balance)['USDT']['free']
        ticker = self._safe_api_call(self.exchange.fetch_ticker, symbol)
        min_notional = 10  # الحد الأدنى للصفقة في Binance
        return max((balance * 0.02) / ticker['last'], min_notional / ticker['last'])
    
    def _generate_signal(self, df):
        """إشارات متوافقة مع سياسة Binance"""
        latest = df.iloc[-1]
        prediction = self.model.predict([[latest['open'], latest['high'], latest['low'],
                                       latest['close'], latest['volume'], latest['rsi'],
                                       latest['macd'], latest['adx']]])
        
        buy_cond = (prediction[0] == 1) & (latest['rsi'] < 35) & (latest['macd'] > latest['signal'])
        sell_cond = (prediction[0] == 0) & (latest['rsi'] > 65) & (latest['macd'] < latest['signal'])
        
        return 'buy' if buy_cond else 'sell' if sell_cond else None
    
    def _execute_order(self, symbol, signal):
        """تنفيذ أوامر آمنة مع مراعاة الشروط"""
        try:
            amount = self._calculate_risk(symbol)
            order = self._safe_api_call(
                self.exchange.create_market_order,
                symbol=symbol,
                side=signal,
                amount=amount
            )
            self._send_alert(f"✅ {signal.upper()} {symbol}\nAmount: {amount:.5f}")
            return order
        except Exception as e:
            logging.error(f"Order Failed: {str(e)}")
            self._send_alert(f"❌ Error: {str(e)}")
            time.sleep(60)
    
    def _send_alert(self, message):
        """إرسال تنبيهات مع التعامل مع الأخطاء"""
        try:
            self.telegram_bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logging.error(f"Telegram Error: {str(e)}")
    
    def run(self):
        """الدورة الرئيسية مع الالتزام التام بالقيود"""
        while True:
            try:
                # تحديث النموذج كل ساعة
                if datetime.now().minute == 0:
                    self.model = self._train_ai_model()
                    logging.info("Model Updated")
                
                for symbol in self.symbols:
                    try:
                        data = self._safe_api_call(self.exchange.fetch_ohlcv, symbol, '5m', limit=100)
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df = self._apply_advanced_indicators(df)
                        
                        if len(df) < 20:
                            continue
                            
                        signal = self._generate_signal(df)
                        
                        if signal:
                            self._execute_order(symbol, signal)
                            
                        time.sleep(0.5)  # فاصل بين الطلبات
                    
                    except ccxt.NetworkError as e:
                        logging.warning(f"Network Issue: {str(e)}")
                        time.sleep(30)
                    except ccxt.ExchangeError as e:
                        logging.error(f"Exchange Error: {str(e)}")
                        time.sleep(300)
                
                time.sleep(300 - (time.time() % 300))  # دقة توقيت مطلقة
                
            except KeyboardInterrupt:
                self._send_alert("🚨 Emergency Stop!")
                break
            except Exception as e:
                logging.critical(f"Critical Error: {str(e)}")
                time.sleep(600)

if __name__ == "__main__":
    trader = BinanceSmartTrader()
    trader.run()
