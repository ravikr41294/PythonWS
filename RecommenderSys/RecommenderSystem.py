import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data= fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
model=LightFM(loss='warp') #loss measures difference between model prediction and desired output
                            #WARP helps cretae recommendations for each user by looking at existing user rating pairs
                            #and predicting ranking for each, and uses Gradient Descent algo to find weights iteratively
model.fit(data['train'], epochs=30,num_threads=2) #epochs = run

def sample_recommend(model, data, user_ids):
    n_users, n_items= data['train'].shape

    #generate recoomendations for each user
    for user_id in user_ids:
        #movies they already like
        known_positives= data['item_labels'][data['train'].tocsr()[user_id].indices]
        #movies our model will predict they like
        scores= model.predict(user_id, np.arange(n_items))
        #rank in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known Positives:::")
        for x in known_positives[:3]:
            print("     %s" % x)
        print("     Recommended::::")
        for x in top_items:
            print("     %s" % x)


sample_recommend(model, data,[3,25,450])


# ####Assssignnment
# 1.New Method to fetch and format new dataset
# 2.Train on 3 models
# 3.Print results of best one