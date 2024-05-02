import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import sys
import urllib.error
import urllib.request
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib
#matplotlib.use('Agg')

class ConstructionAndEvaluationSarima():
    def __init__(self, flg_web):
        #self.ticker = "7203.T"
        self.start_predict = "2016-01-01"
        self.end_predict = "2024-12-31"
        self.interval = "1mo"

        self.stock_score = None
        self.df_stock_score = None

        self.ticker = None
        self.ticker_T = None
        self.company_name = None

        self.df_stock = None
        self.df_stock_train = None
        self.df_stock_test = None

        self.train_st = None
        self.train_end = None
        self.test_st = None
        self.test_end = None

        self.stock_train_pred = None
        self.stock_test_pred = None

        self.best_param = None

        self.url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        self.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]

        #self.flg_web = True
        self.flg_web = flg_web

        if self.flg_web == True:
            matplotlib.use('Agg')

    # 証券コードから株価を取得し、詳細を表示
    # train, testに分割
    def get_stockdata(self, ticker=None):
        if self.flg_web:
            self.ticker = ticker #input('予測銘柄の証券コードを入力してください: ')
        else:
            self.ticker = input('予測銘柄の証券コードを入力してください: ')

        self.ticker_T = self.ticker + ".T"
        print(f'your input: {self.ticker_T}')

        opener = urllib.request.build_opener()
        opener.addheaders = self.addheaders
        urllib.request.install_opener(opener)
        save_path, _ = urllib.request.urlretrieve(self.url)
        stocklist = pd.read_excel(save_path)

        df_stocklist = stocklist.set_index('コード')
        print("指定した証券コードの銘柄詳細：")
        print(df_stocklist.loc[df_stocklist.index == int(self.ticker)])
        self.company_name = df_stocklist.loc[df_stocklist.index == int(self.ticker)]["銘柄名"].iloc[0]

        if self.flg_web == False:
            input("再開する場合はenterを押下")

        stock_Sorce = yf.download(self.ticker_T, start=self.start_predict, end=self.end_predict, interval=self.interval)       
        stock=stock_Sorce['Close']  
        self.df_stock = pd.DataFrame(stock)
        length = len(self.df_stock)
        train_length = int(length * 0.7)
        
        self.df_stock_train = self.df_stock[:train_length]
        self.train_st = self.df_stock_train.index[0]
        self.train_end = self.df_stock_train.index[-1]
        
        self.df_stock_test = self.df_stock[train_length:]
        self.test_st = self.df_stock_test.index[0]
        self.test_end = self.df_stock_test.index[-1]

    # 自己相関・偏自己相関係数を表示
    # ADF検定を実施
    def analize_trend(self):        
        #自己相関係数の可視化
        fig=plt.figure(figsize=(9,7))        
        ax1=fig.add_subplot(211)
        fig=sm.graphics.tsa.plot_acf(self.df_stock_train,ax=ax1)
        plt.show()

        #偏自己相関係数の可視化
        ax2=fig.add_subplot(212)
        fig=sm.graphics.tsa.plot_pacf(self.df_stock_train,ax=ax2)
        plt.show()
        
        #データパターンの確認
        #plt.rcParams['figure.figsize'] = [12, 9] # グラフサイズ設定
        sm.tsa.seasonal_decompose(self.df_stock_train, period=12).plot()

        # ADF検定（単位根テスト）を実施
        result = adfuller(self.df_stock_train)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

    def select_parameter(self, DATA, s):
    #非季節性パラメータ（p,d,q）
    #季節性パラメータ (P, D, Q, m)
    #ベイズ情報量基準(BIC)
        p=d=q=range(0,2)
        pdq=list(itertools.product(p,d,q))
        seasonal_pdq=[(x[0],x[1],x[2],s) for x in list(itertools.product(p,d,q))]
        parameters=[]
        BICs=np.array([])
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod=sm.tsa.statespace.SARIMAX(DATA,order=param,seasonal_order=param_seasonal)
                    results=mod.fit()
                    parameters.append([param,param_seasonal,results.bic])
                    BICs=np.append(BICs,results.bic)
                except:
                    print(sys.exc_info())
                    continue

        self.best_param = parameters[np.argmin(BICs)]
        return print(parameters[np.argmin(BICs)])

    def build_model(self):
        print("BICが最も良いモデル:", self.best_param) 
        if self.flg_web == False:
            input("再開する場合はenterを押下")
        prm_1 = self.best_param[0]
        prm_2 = self.best_param[1]

        #モデルの構築
        SARIMA_stock=sm.tsa.statespace.SARIMAX(self.df_stock_train,order=(prm_1[0],prm_1[1],prm_1[2]),seasonal_order=(prm_2[0],prm_2[1],prm_2[2],prm_2[3])).fit()
        print(SARIMA_stock.summary())
        SARIMA_stock.plot_diagnostics(lags=20, figsize=(16,16))
        self.stock_train_pred=SARIMA_stock.predict(self.train_st, self.train_end)
        self.stock_test_pred=SARIMA_stock.predict(self.test_st, self.test_end)

        plt.figure()
        plt.plot(self.stock_train_pred)
        plt.plot(self.stock_test_pred)
        plt.show()

    def evaluate_predict(self):
        #元の時系列データと予測データの比較
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"Results({self.company_name},{self.ticker})", fontname="MS Gothic")
        plt.plot(self.df_stock,color="blue",label="original")
        plt.plot(self.stock_train_pred,color="r", label="train predict")
        plt.plot(self.stock_test_pred,color="green", label="test predict")
        plt.legend()       
        plt.show()
              
        # 精度指標（テストデータ）
        list_stock_test = self.df_stock_test.to_numpy().tolist()
        list_stock_test = list(itertools.chain.from_iterable(list_stock_test))
        
        list_stock_test_pred = self.stock_test_pred.to_numpy().tolist()

        rmse = mae = mape = None
        if self.flg_web == True:
            rmse = np.sqrt(mean_squared_error(list_stock_test, list_stock_test_pred))
            mae = mean_absolute_error(list_stock_test, list_stock_test_pred)
            mape = mean_absolute_percentage_error(list_stock_test, list_stock_test_pred)
        else:
            print('RMSE:')
            print(np.sqrt(mean_squared_error(list_stock_test, list_stock_test_pred)))
            print('MAE:')
            print(mean_absolute_error(list_stock_test, list_stock_test_pred))
            print('MAPE:')
            print(mean_absolute_percentage_error(list_stock_test, list_stock_test_pred))

            print("end")

        return fig, rmse, mae, mape

    def main(self, ticker=None):
        self.get_stockdata(ticker)
        if self.flg_web == False:
            self.analize_trend()
        self.select_parameter(self.df_stock_train, 12)
        self.build_model()
        fig, rmse, mae, mape = self.evaluate_predict()

        return fig, rmse, mae, mape

if __name__ == "__main__":
    test = ConstructionAndEvaluationSarima(False)
    test.get_stockdata()
    if test.flg_web == False:
        test.analize_trend()
    test.select_parameter(test.df_stock_train, 12)
    test.build_model()
    test.evaluate_predict()