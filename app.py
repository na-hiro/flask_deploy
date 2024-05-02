from flask import Flask, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64
import AnalysisClusterAndTrend as analysis_trend
import ConstructionAndEvaluationSarima as consteval_sarima

app = Flask(__name__)

def draw(fig):
    canvas = FigureCanvasAgg(fig)
    s = io.BytesIO()
    canvas.print_png(s)
    s = "data:image/png;base64," + base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return s
	
@app.route('/cluster', methods=['GET','POST'])
def cluster():
    if request.method == 'GET':
        msg = '業種別指数の解析を行います。実行する場合は、送信ボタンを押下してください。'
        return render_template('cluster_input.html', title='クラスタ数=5で解析を行います。', message=msg)

    if request.method == 'POST':
        analysis_handler = analysis_trend.AnalysisClusterAndTrend(True)
        fig, _, pca_fig, cluster_list = analysis_handler.main()
    
        return render_template(
            "cluster_ouput.html",
            plot=draw(fig), 
            plot_pca=draw(pca_fig),             
            title='業種別指数を5クラスタに分類', 
            message='クラスタ解析結果の表示', 
            cluster_0 = cluster_list[0],
            cluster_1 = cluster_list[1],
            cluster_2 = cluster_list[2],
            cluster_3 = cluster_list[3],
            cluster_4 = cluster_list[4]

            )

@app.route('/', methods=['GET','POST'])
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        msg = '証券コードを入力してください'
        return render_template('predict_input.html', title='指定された証券コードの株価を予測します。', message=msg)

    if request.method == 'POST':
        ticker = request.form['ticker_code']   

        predict_handler = consteval_sarima.ConstructionAndEvaluationSarima(True)
        fig, rmse, mae, mape = predict_handler.main(ticker) 
        
    
        return render_template(
            "predict_output.html",
            plot=draw(fig), 
            title='指定された証券コードの株価を予測', 
            message='結果の表示',
            ticker_code=ticker,
            rmse = rmse,
            mae = mae,
            mape = mape,
            company_name = predict_handler.company_name
            )

if __name__ == "__main__":
    app.run(debug=True)