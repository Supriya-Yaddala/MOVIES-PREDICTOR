from flask import Flask ,render_template,request
import pandas as pd 
from math import sqrt 

app=Flask(__name__)
@app.route('/')
def home():
	return render_template('userinput.html')
@app.route('/predict',methods=["POST"])
def predict():
	movies_df=pd.read_csv('data\movies.csv')
	movies_df['year']=movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
	movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)',expand=False)
	movies_df['title']=movies_df.title.str.replace('(\(\d\d\d\d\))','')
	movies_df['title']=movies_df.title.apply(lambda x:x.strip())
	movies_df=movies_df.drop('genres',1)
	#Reading ratings data
	ratings_df=pd.read_csv('data\\ratings.csv')
	#preprocessing rating data
	ratings_df=ratings_df.drop('timestamp',1)
	#Getting user input
	userinput=[]
	userinput.append({'title':request.form['m1'],'rating':float(request.form['r1'])})
	userinput.append({'title':request.form['m2'],'rating':float(request.form['r2'])})
	userinput.append({'title':request.form['m3'],'rating':float(request.form['r3'])})
	userinput.append({'title':request.form['m4'],'rating':float(request.form['r4'])})
	userinput.append({'title':request.form['m5'],'rating':float(request.form['r5'])})


	#Converting userdata into a dataframe 
	user_df=pd.DataFrame(userinput) 
	#finding movie ids for userinput movies
	usermovieid=movies_df[movies_df['title'].isin(user_df['title'].tolist())]
	#Add movieId to user dataframe 
	user_df=pd.merge(user_df,usermovieid)
	user_df=user_df.drop('year',1)
	#finding similar users  as our active users
	similarusers=ratings_df[ratings_df['movieId'].isin(user_df['movieId'].tolist())]
	similarusers=similarusers.groupby(['userId'])
	#sort users based on no of common movies watched
	similarusers=sorted(similarusers,key=lambda x:len(x[1]),reverse=True)  
	#finding similarity for users using pearson correlation coefficient 
	pearsoncoefficient={}
	for name,group in similarusers:
		user_df= user_df.sort_values(by='movieId')
		group=group.sort_values(by='movieId')
		nratings=len(group)
		tusers_df=user_df[user_df['movieId'].isin(group['movieId'].tolist())]
		group_df=group[group['movieId'].isin(user_df['movieId'].tolist())]
		sdx=sum([i**2 for i in tusers_df['rating']])-pow(sum(tusers_df['rating']),2)/float(nratings)
		sdy=sum([i**2 for i in group['rating']])-pow(sum(group['rating']),2)/float(nratings)
		sxy=sum([i*j for i,j in zip(tusers_df['rating'],group['rating'])])-sum(tusers_df['rating'])*sum(group['rating'])/float(nratings)
		if sdx!=0 and sdy!=0:
			pearsoncoefficient[name]=sxy/sqrt(sdx*sdy)
		else:
			pearsoncoefficient[name]=0
	#convert pearson coefficients for each user into a dataframe
	pearsondf=pd.DataFrame.from_dict(pearsoncoefficient,orient='index')
	pearsondf.columns=['similarity Index']
	pearsondf['userId']=pearsondf.index
	pearsondf.index=range(len(pearsondf))
	#find top similar users to our active user
	topusers=pearsondf.sort_values(by='similarity Index',ascending=False)[0:50]
	topusers=topusers.merge(ratings_df,left_on='userId',right_on='userId',how='inner')
	topusers['weigted ratings']=topusers['similarity Index']*topusers['rating']
	temptopusers=topusers.groupby('movieId').sum()[['weigted ratings','similarity Index']]
	temptopusers['norrating']=temptopusers['weigted ratings']/temptopusers['similarity Index']
	temptopusers=temptopusers.sort_values(by='norrating',ascending=False)[0:10]
	temptopusers['movieId']=temptopusers.index
	temptopusers.index=range(len(temptopusers))
	rec_mdf=movies_df[movies_df['movieId'].isin(temptopusers['movieId'].tolist())]
	temptopusers=temptopusers.merge(rec_mdf,left_on='movieId',right_on='movieId',how='inner')

	return render_template('result.html',movies=temptopusers['title'])
if __name__=='__main__':
	app.run(debug=True)