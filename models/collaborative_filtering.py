import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class CollaborativeFiltering:
    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        self.user_item_matrix = None
        self.user_means = None
        self.user_id_map = None
        self.item_id_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        
    def fit(self, interactions_df):
        """Train the collaborative filtering model"""
        print("🔄 Training Collaborative Filtering Model...")
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='activity_id', 
            values='rating', 
            fill_value=0
        )
        
        # Create mappings for matrix indices
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(self.user_item_matrix.index)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(self.user_item_matrix.columns)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        # Calculate user means for normalization
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Normalize the matrix by subtracting user means
        normalized_matrix = self.user_item_matrix.sub(self.user_means, axis=0)
        normalized_matrix = normalized_matrix.fillna(0)
        
        # Apply SVD for matrix factorization
        self.model = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.user_factors = self.model.fit_transform(normalized_matrix)
        self.item_factors = self.model.components_
        
        print(f"✅ Model trained with {len(self.user_id_map)} users and {len(self.item_id_map)} activities")
        print(f"📊 Matrix shape: {self.user_item_matrix.shape}")
        print(f"🔢 SVD factors: {self.n_factors}")
        
    def predict_rating(self, user_id, activity_id):
        """Predict rating for a specific user-activity pair"""
        if user_id not in self.user_id_map or activity_id not in self.item_id_map:
            return self.user_item_matrix.values.mean()  # Global average as fallback
            
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[activity_id]
        
        # Reconstruct rating using user and item factors
        predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[:, item_idx])
        predicted_rating += self.user_means.iloc[user_idx]
        
        # Clip rating to valid range (1-5)
        return np.clip(predicted_rating, 1, 5)
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """Generate recommendations for a user"""
        if user_id not in self.user_id_map:
            print(f"⚠️ User {user_id} not found in training data")
            return []
            
        user_idx = self.user_id_map[user_id]
        
        # Get all activity predictions for this user
        user_ratings = np.dot(self.user_factors[user_idx], self.item_factors)
        user_ratings += self.user_means.iloc[user_idx]
        
        # Get user's already rated activities
        if exclude_rated:
            rated_activities = self.user_item_matrix.iloc[user_idx]
            rated_indices = np.where(rated_activities > 0)[0]
            user_ratings[rated_indices] = -np.inf  # Exclude already rated
        
        # Get top recommendations
        top_indices = np.argsort(user_ratings)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            activity_id = self.reverse_item_map[idx]
            predicted_rating = np.clip(user_ratings[idx], 1, 5)
            recommendations.append({
                'activity_id': activity_id,
                'predicted_rating': round(predicted_rating, 2)
            })
            
        return recommendations
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'user_item_matrix': self.user_item_matrix,
            'user_means': self.user_means,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'n_factors': self.n_factors
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_means = model_data['user_means']
        self.user_id_map = model_data['user_id_map']
        self.item_id_map = model_data['item_id_map']
        self.reverse_user_map = model_data['reverse_user_map']
        self.reverse_item_map = model_data['reverse_item_map']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.n_factors = model_data['n_factors']
        
        print(f"📂 Model loaded from {filepath}")

if __name__ == "__main__":
    # Test the model
    interactions_df = pd.read_csv('data/processed/interactions.csv')
    
    cf_model = CollaborativeFiltering(n_factors=50)
    cf_model.fit(interactions_df)
    
    # Test recommendations for user 1
    recommendations = cf_model.recommend(user_id=1, n_recommendations=5)
    print(f"\n🎯 Top 5 recommendations for User 1:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. Activity {rec['activity_id']} (Score: {rec['predicted_rating']})")
    
    # Save the model
    cf_model.save_model('saved_models/collaborative_filtering.pkl')
