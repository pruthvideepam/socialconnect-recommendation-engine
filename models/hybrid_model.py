import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.collaborative_filtering import CollaborativeFiltering
from models.content_based import ContentBasedFiltering

class HybridRecommendationSystem:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        """
        Hybrid recommendation system combining collaborative and content-based filtering
        
        Args:
            cf_weight: Weight for collaborative filtering (default: 0.6)
            cb_weight: Weight for content-based filtering (default: 0.4)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = CollaborativeFiltering()
        self.cb_model = ContentBasedFiltering()
        self.activities_df = None
        self.interactions_df = None
        self.is_trained = False
        
    def fit(self, activities_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both collaborative and content-based models"""
        print("🔄 Training Hybrid Recommendation System...")
        print(f"⚖️ Weights: CF={self.cf_weight}, CB={self.cb_weight}")
        
        self.activities_df = activities_df
        self.interactions_df = interactions_df
        
        # Train both models
        print("🤝 Training Collaborative Filtering...")
        self.cf_model.fit(interactions_df)
        
        print("📋 Training Content-Based Filtering...")
        self.cb_model.fit(activities_df, interactions_df)
        
        self.is_trained = True
        print("✅ Hybrid model training completed!")
        
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 categories: Optional[List[str]] = None, 
                 min_score: float = 2.0) -> List[Dict]:
        """
        Generate hybrid recommendations for a user
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            categories: Filter by specific categories (optional)
            min_score: Minimum recommendation score threshold
            
        Returns:
            List of recommendation dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
            
        # Get recommendations from both models
        cf_recs = self.cf_model.recommend(user_id, n_recommendations * 2)
        cb_recs = self.cb_model.recommend(user_id, self.interactions_df, n_recommendations * 2)
        
        # Create dictionaries for faster lookup
        cf_dict = {rec['activity_id']: rec['predicted_rating'] for rec in cf_recs}
        cb_dict = {rec['activity_id']: rec['predicted_rating'] for rec in cb_recs}
        
        # Get all unique activity IDs
        all_activity_ids = set(cf_dict.keys()) | set(cb_dict.keys())
        
        # Calculate hybrid scores
        hybrid_scores = []
        for activity_id in all_activity_ids:
            cf_score = cf_dict.get(activity_id, 0.0)
            cb_score = cb_dict.get(activity_id, 0.0)
            
            # Weighted combination
            hybrid_score = (self.cf_weight * cf_score) + (self.cb_weight * cb_score)
            
            # Apply minimum score filter
            if hybrid_score >= min_score:
                hybrid_scores.append({
                    'activity_id': activity_id,
                    'hybrid_score': hybrid_score,
                    'cf_score': cf_score,
                    'cb_score': cb_score
                })
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Filter by categories if specified
        if categories:
            filtered_scores = []
            for item in hybrid_scores:
                activity = self.activities_df[self.activities_df['activity_id'] == item['activity_id']]
                if not activity.empty and activity.iloc[0]['category'] in categories:
                    filtered_scores.append(item)
            hybrid_scores = filtered_scores
        
        # Build final recommendations with activity details
        recommendations = []
        for item in hybrid_scores[:n_recommendations]:
            activity = self.activities_df[self.activities_df['activity_id'] == item['activity_id']]
            if not activity.empty:
                activity_info = activity.iloc[0]
                recommendations.append({
                    'activity_id': item['activity_id'],
                    'title': activity_info['title'],
                    'category': activity_info['category'],
                    'location': activity_info['location'],
                    'price_tier': activity_info['price_tier'],
                    'duration_hours': activity_info['duration_hours'],
                    'predicted_rating': round(item['hybrid_score'], 2),
                    'cf_score': round(item['cf_score'], 2),
                    'cb_score': round(item['cb_score'], 2),
                    'recommendation_type': self._get_recommendation_type(item['cf_score'], item['cb_score'])
                })
        
        return recommendations
    
    def _get_recommendation_type(self, cf_score: float, cb_score: float) -> str:
        """Determine the primary recommendation approach"""
        if cf_score > cb_score * 1.2:
            return "collaborative"
        elif cb_score > cf_score * 1.2:
            return "content-based"
        else:
            return "hybrid"
    
    def get_user_insights(self, user_id: int) -> Dict:
        """Get insights about a user's preferences and behavior"""
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        if user_interactions.empty:
            return {"status": "new_user", "insights": "No interaction history"}
        
        # Category preferences
        user_activities = user_interactions.merge(self.activities_df, on='activity_id')
        category_stats = user_activities.groupby('category')['rating'].agg(['count', 'mean']).round(2)
        top_categories = category_stats.nlargest(3, 'mean')
        
        # Activity patterns
        insights = {
            'total_interactions': len(user_interactions),
            'average_rating': round(user_interactions['rating'].mean(), 2),
            'top_categories': top_categories.to_dict(),
            'rating_distribution': user_interactions['rating'].value_counts().to_dict(),
            'recommendation_strategy': 'hybrid' if len(user_interactions) >= 10 else 'content-based'
        }
        
        return insights
    
    def save_model(self, filepath: str):
        """Save the entire hybrid model"""
        model_data = {
            'cf_model': self.cf_model,
            'cb_model': self.cb_model,
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'activities_df': self.activities_df,
            'interactions_df': self.interactions_df,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Hybrid model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained hybrid model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.cf_model = model_data['cf_model']
        self.cb_model = model_data['cb_model']
        self.cf_weight = model_data['cf_weight']
        self.cb_weight = model_data['cb_weight']
        self.activities_df = model_data['activities_df']
        self.interactions_df = model_data['interactions_df']
        self.is_trained = model_data['is_trained']
        
        print(f"📂 Hybrid model loaded from {filepath}")

if __name__ == "__main__":
    # Test hybrid model
    activities_df = pd.read_csv('data/processed/activities_improved.csv')
    interactions_df = pd.read_csv('data/processed/interactions_improved.csv')
    users_df = pd.read_csv('data/processed/users_improved.csv')
    
    # Train hybrid model
    hybrid_model = HybridRecommendationSystem(cf_weight=0.6, cb_weight=0.4)
    hybrid_model.fit(activities_df, interactions_df)
    
    # Test recommendations for different users
    test_users = [1, 50, 100]
    
    for user_id in test_users:
        print(f"\n🎯 Hybrid Recommendations for User {user_id}:")
        
        # User profile
        user_profile = users_df[users_df['user_id'] == user_id].iloc[0]
        print(f"👤 Profile: {user_profile['persona']}, {user_profile['age']} years, {user_profile['location']}")
        
        # Get insights
        insights = hybrid_model.get_user_insights(user_id)
        print(f"📊 Insights: {insights['total_interactions']} interactions, avg rating {insights['average_rating']}")
        
        # Get recommendations
        recommendations = hybrid_model.recommend(user_id, n_recommendations=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['title']} ({rec['category']})")
            print(f"     Score: {rec['predicted_rating']} | CF: {rec['cf_score']} | CB: {rec['cb_score']} | Type: {rec['recommendation_type']}")
        
        # Test category filtering
        print(f"\n🎯 Sports-only recommendations for User {user_id}:")
        sports_recs = hybrid_model.recommend(user_id, n_recommendations=3, categories=['Sports'])
        for i, rec in enumerate(sports_recs, 1):
            print(f"  {i}. {rec['title']} - Score: {rec['predicted_rating']}")
    
    # Save model
    hybrid_model.save_model('saved_models/hybrid_model.pkl')
    print(f"\n✅ Hybrid model testing completed!")
