from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	cvFile = open('vector_convert.vec','rb')
	cv = joblib.load(cvFile)

	modelFile = open('final_model.pkl','rb')
	clf = joblib.load(modelFile)
	if request.method == 'POST':
		message = request.form['message']
		data = [message]

		vect = cv.transform(data).toarray()
		#df = pd.DataFrame(vect, columns=cv.get_feature_names())
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)