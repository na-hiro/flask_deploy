import urllib.error
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib
#matplotlib.use('Agg')


class AnalysisClusterAndTrend():
    def __init__(self, flg_web):

        # 業種別の株価指数
        self.url = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_500_stock_average_daily_jp.csv"
        # エラー回避
        self.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        # 業種を日本語に置き換え
        self.table_clums = {'終値':'日経500','業種別（水産）終値':'水産', '業種別（鉱業）終値':'鉱業', '業種別（建設）終値':'建設', '業種別（食品）終値':'食品',\
            '業種別（繊維）終値':'繊維','業種別（パルプ・紙）終値':'パルプ・紙', '業種別（化学）終値':'化学', '業種別（医薬品）終値':'医薬品',\
            '業種別（石油）終値':'石油','業種別（ゴム）終値':'ゴム', '業種別（窯業）終値':'窯業','業種別（鉄鋼）終値':'鉄鋼','業種別（非鉄・金属）終値':'非鉄・金属',\
            '業種別（機械）終値':'機械','業種別（電気機器）終値':'電気機器','業種別（造船）終値':'造船','業種別（自動車）終値':'自動車',\
            '業種別（輸送用機器）終値':'輸送用機器','業種別（精密機器）終値':'精密機器', '業種別（その他製造）終値':'その他製造','業種別（商社）終値':'商社',\
            '業種別（小売業）終値':'小売業','業種別（銀行）終値':'銀行','業種別（その他金融）終値':'その他金融','業種別（証券）終値':'証券',\
            '業種別（保険）終値':'保険','業種別（不動産）終値':'不動産','業種別（鉄道・バス）終値':'鉄道・バス','業種別（陸運）終値':'陸運',\
            '業種別（海運）終値':'海運', '業種別（空運）終値':'空運', '業種別（倉庫）終値':'倉庫', '業種別（通信）終値':'通信',\
            '業種別（電力）終値':'電力', '業種別（ガス）終値':'ガス', '業種別（サービス）終値':'サービス'}
        # グラフの色
        self.COLOR_LIST=["blue","orange","green","red","brown"]
        # クラスタの中心ベクトル
        self.centers = None
        # クラスタリング前のデータ(DataFrame)
        self.df_clustering = None
        # クラスタリング後のデータ(NumPyの二次元配列)
        self.df_clustering_sc = None
        # クラスタリング前のデータ(DataFrame) set index
        self.df_index = None

        #self.flg_web = True
        self.flg_web = flg_web

        if self.flg_web == True:
            matplotlib.use('Agg')

    # 業種別指数取得
    def get_stock_data(self):
        opener = urllib.request.build_opener()
        opener.addheaders = self.addheaders
        urllib.request.install_opener(opener)
        self.save_path, _ = urllib.request.urlretrieve(self.url)

    # データ整形
    def data_corr(self):
        # タブ区切りの文字を読み込む
        df = pd.read_csv(self.save_path, encoding='shift_jis')
        df = df.drop(df.shape[0]-1)
        #print(df)
        
        df["データ日付"] = pd.to_datetime(df["データ日付"], format='%Y/%m/%d')
        df = df.set_index('データ日付')
        df = df[df.index > dt.datetime(2022,2,28)]
        
        #不必要なカラムを削除。今回の指数の値は全て終値で示します。
        df = df.drop(['始値', '高値', '安値'], axis=1)
        df = df.sort_index(ascending=True)
        
        #名前の変更
        #'業種別（電力）終値':'Electric power', '業種別（ガス）終値':'Gas', '業種別（サービス）終値':'Services'}, inplace=True)
        df.rename(columns=self.table_clums, inplace=True)

        for x,y in df.items():
            m=df[x].mean()
            df[x]=df[x].div(m)

        # 転置
        df = df.transpose()
        self.df_index = df
        df = df.reset_index(drop=True)
        self.df_clustering = df

    # クラスタリング
    def analysis_cluster(self):
        #表示オプション調整
        #NumPyの浮動少数点の表示精度
        np.set_printoptions(suppress=True, precision=4)
        #pandasでの浮動少数点の表示精度
        pd.options.display.float_format = '{:.4f}'.format
        #データフレームですべての項目を表示
        pd.set_option("display.max_columns",None)
        #グラフのデフォルトフォント設定
        plt.rcParams["font.size"] = 14
        
        self.df_clustering_sc = self.df_clustering.to_numpy()
        
        #クラスタ解析
        N_CLUSTERS=5
        cls = KMeans(n_clusters=N_CLUSTERS, init='k-means++',
            n_init=10, max_iter=1000, tol=0.0001, verbose=0,
            random_state=0, copy_x=True, algorithm='lloyd')
        
        pred = cls.fit_predict(self.df_clustering_sc)
        
        self.centers = cls.cluster_centers_
        labels = cls.labels_
        self.df_clustering["cluster"] = pred

    #トレンド解析
    def analysis_trend(self):
        ratio_list = []
        fig = plt.figure(figsize=(12, 6))   #新しいウィンドウを描画
        fig.suptitle("Trend")
        for i, lists in enumerate(self.centers):
            #print("cluster: ", i, lists)
            #データパターンの確認
            stl = sm.tsa.seasonal_decompose(lists, period=12)
            stl_list = stl.trend.tolist()
            newlist = [x for x in stl_list if np.isnan(x) == False]
            st = newlist[0]
            fst_half = newlist[int(len(newlist)/3)]
            second_half = newlist[int(len(newlist)*2/3)]
            end = newlist[-1]
            ratio_end = end / st
            ratio_fst = fst_half/ st
            ratio_2nd = second_half/ st
            ratio = [ratio_fst, ratio_2nd, ratio_end]
            ratio_list.append(ratio)
            cluster_label = "cluster_" + str(i)
            plt.plot(newlist,color=self.COLOR_LIST[i],label=cluster_label)
            plt.legend()

        cluster_list = list(range(5))
        if self.flg_web:
            for i in self.df_clustering["cluster"].unique():
                tmp = self.df_clustering.loc[self.df_clustering["cluster"]==i]
                index_list = list(self.df_index.index[tmp.index])
                cluster_list[i] =index_list
        else:
            plt.show()
            print("Nikkei500が所属するクラスタ: ", self.df_clustering["cluster"][0])
            
            while True:
                n = input('input cluster number: ')
                print(f'your input: {n}')
                if n.isdecimal() == True:
                    for i in self.df_clustering["cluster"].unique():
                        tmp = self.df_clustering.loc[self.df_clustering["cluster"]==i]
                        if i == int(n):
                            index_list = list(self.df_index.index[tmp.index])
                            print("指定したクラスタの業種は、以下の通りです。")
                            print(index_list)
                            break
                else:
                    break

        
        #トレンド解析表示
        plt.figure()   #新しいウィンドウを描画
        for cluster_id, pos in enumerate(ratio_list):
            cluster_label = "cluster_" + str(cluster_id)
            plt.plot(pos, label=cluster_label)
            plt.legend()
        plt.show()

        return fig, self.df_clustering["cluster"][0], cluster_list

    # 次元圧縮（主成分分析）後、プロット
    def result(self):
        #PCAによるプロット
        from sklearn.decomposition import PCA
        X = self.df_clustering_sc
        pca = PCA(n_components=2)
        pca.fit(X)
        x_pca = pca.transform(X)
        pca_df = pd.DataFrame(x_pca)
        pca_df["cluster"] = self.df_clustering["cluster"]
        
        fig = plt.figure(figsize=(12, 6))   #新しいウィンドウを描画
        fig.suptitle("PCA plot")
        for i in self.df_clustering["cluster"].unique():
            tmp = pca_df.loc[pca_df["cluster"]==i]
            cluster_label = "cluster_" + str(i)
            plt.scatter(tmp[0], tmp[1], label=cluster_label, c=self.COLOR_LIST[i])
            plt.legend()

        plt.show()

        print("end")

        return fig

    def main(self):
        self.get_stock_data()
        self.data_corr()
        self.analysis_cluster()
        fig, cluster, cluster_list = self.analysis_trend()
        pca_fig = self.result()

        return fig, cluster, pca_fig, cluster_list
        
        
if __name__ == "__main__":
    test = AnalysisClusterAndTrend(False)
    test.get_stock_data()
    test.data_corr()
    test.analysis_cluster()
    test.analysis_trend()
    test.result()
