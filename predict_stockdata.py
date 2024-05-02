import AnalysisClusterAndTrend as analysis_trend
import ConstructionAndEvaluationSarima as consteval_sarima


def main():
    # 日経平均と業種別指数を解析して、解析したい業種の指数を抽出
    n = input("クラスタ解析を実施する場合(y)/しない場合(それ以外)：")
    print(f'your input: {n}')
    if n == "y":
        analysis_handler = analysis_trend.AnalysisClusterAndTrend(False)
        analysis_handler.main()

    # 証券コードを指定し、SARIMAモデルのパラメータを算出して、株価を予測
    n = input("株価予測を実施する場合(y)/しない場合(それ以外)：")
    print(f'your input: {n}')
    if n == "y":
        predict_handler = consteval_sarima.ConstructionAndEvaluationSarima(False)
        predict_handler.main()

if __name__ == "__main__":
    main()