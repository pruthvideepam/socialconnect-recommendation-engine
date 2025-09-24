import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class ContentBasedFiltering:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.activity_features = None
        self.activities_df = None
        self.le_category = LabelEncoder()
        self.le_price = LabelEncoder()
        self.le_location = LabelEncoder()
        
    def fit(self, activities_df, interactions_df):
        """Train content-based filtering model"""
        print("🔄 Training Content-Based Filtering Model...")
        
        self.activities_df = activities_df.copy()
        
        # Create content features
        self.activities_df['content'] = (
            self.activities_df['category'] + ' ' + 
            self.activities_df['price_tier'] + ' ' + 
            self.activities_df['location'] + ' ' +
            self.activities_df['duration_hours'].astype(str) + 'h'
        )
        
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.activity_features = self.tfidf_vectorizer.fit_transform(self.activities_df['content'])
        
        # Calculate activity similarity matrix
        self.similarity_matrix = cosine_similarity(self.activity_features)
        
        print(f"✅ Content model trained with {len(activities_df)} activities")
        print(f"🔢 Feature dimensions: {self.activity_features.shape[1]}")
        
    def get_user_profile(self, user_id, interactions_df, top_n=10):
        """Create user profile based on highly rated activities"""
        user_interactions = interactions_df[
            (interactions_df['user_id'] == user_id) & 
            (interactions_df['rating'] >= 4)
        ]
        
        if user_interactions.empty:
            return np.zeros(self.activity_features.shape[1])
        
        # Get activities user liked
        liked_activities = user_interactions['activity_id'].tolist()
        activity_indices = [self.activities_df[self.activities_df['activity_id'] == aid].index[0] 
                          for aid in liked_activities 
                          if aid in self.activities_df['activity_id'].values]
        
        if not activity_indices:
            return np.zeros(self.activity_features.shape[1])
        
        # Average feature vectors of liked activities
        user_profile = np.mean(self.activity_features[activity_indices].toarray(), axis=0)
        return user_profile
        
    def recommend(self, user_id, interactions_df, n_recommendations=10):
        """Generate content-based recommendations"""
        
        # Get user profile
        user_profile = self.get_user_profile(user_id, interactions_df)
        
        if np.sum(user_profile) == 0:
            # Cold start: recommend popular activities
            popular_activities = interactions_df.groupby('activity_id')['rating'].agg(['count', 'mean'])
            popular_activities['score'] = popular_activities['count'] * popular_activities['mean']
            top_activities = popular_activities.nlargest(n_recommendations, 'score')
            
            recommendations = []
            for activity_id in top_activities.index:
                if activity_id in self.activities_df['activity_id'].values:
                    recommendations.append({
                        'activity_id': activity_id,
                        'predicted_rating': top_activities.loc[activity_id, 'mean']
                    })
            return recommendations
        
        # Calculate similarity with all activities
        user_profile = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile, self.activity_features)[0]
        
        # Get user's already rated activities
        user_rated = set(interactions_df[interactions_df['user_id'] == user_id]['activity_id'])
        
        # Create recommendations
        recommendations = []
        activity_scores = list(enumerate(similarities))
        activity_scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in activity_scores:
            activity_id = self.activities_df.iloc[idx]['activity_id']
            
            if activity_id not in user_rated and len(recommendations) < n_recommendations:
                recommendations.append({
                    'activity_id': activity_id,
                    'predicted_rating': min(5.0, 3.0 + (score * 2))  # Scale similarity to rating
                })
        
        return recommendations
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'activity_features': self.activity_features,
            'activities_df': self.activities_df,
            'similarity_matrix': self.similarity_matrix
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Content-based model saved to {filepath}")

if __name__ == "__main__":
    # Test content-based model
    activities_df = pd.read_csv('data/processed/activities_improved.csv')
    interactions_df = pd.read_csv('data/processed/interactions_improved.csv')
    
    cb_model = ContentBasedFiltering()
    cb_model.fit(activities_df, interactions_df)
    
    # Test recommendations
    recommendations = cb_model.recommend(1, interactions_df, n_recommendations=5)
    print(f"\n🎯 Content-based recommendations for User 1:")
    for i, rec in enumerate(recommendations, 1):
        activity = activities_df[activities_df['activity_id'] == rec['activity_id']].iloc[0]
        print(f"  {i}. {activity['title']} ({activity['category']}) - Score: {rec['predicted_rating']:.2f}")
    
    cb_model.save_model('saved_models/content_based.pkl')
