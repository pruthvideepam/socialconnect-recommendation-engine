import pandas as pd
import numpy as np
import random

class ImprovedSyntheticDataGenerator:
    def __init__(self, n_users=1000, n_activities=500, random_seed=42):
        self.n_users = n_users
        self.n_activities = n_activities
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Define user personas for more realistic interactions
        self.user_personas = {
            'music_lover': ['Music', 'Art'],
            'sports_enthusiast': ['Sports', 'Fitness'],
            'tech_geek': ['Tech', 'Art'],
            'foodie': ['Food', 'Travel'],
            'adventurer': ['Travel', 'Sports', 'Fitness'],
            'social_butterfly': ['Music', 'Food', 'Art']
        }

    def generate_users(self):
        personas = list(self.user_personas.keys())
        return pd.DataFrame({
            'user_id': range(1, self.n_users + 1),
            'age': np.random.randint(18, 50, self.n_users),
            'location': np.random.choice(['Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad'], self.n_users),
            'persona': np.random.choice(personas, self.n_users)
        })

    def generate_activities(self):
        categories = ['Music', 'Sports', 'Tech', 'Art', 'Food', 'Travel', 'Fitness']
        
        # Generate activity features for content-based filtering
        activities = []
        for i in range(1, self.n_activities + 1):
            category = np.random.choice(categories)
            activities.append({
                'activity_id': i,
                'title': f"{category} Event {i}",
                'category': category,
                'price_tier': np.random.choice(['Free', 'Low', 'Medium', 'High']),
                'duration_hours': np.random.choice([1, 2, 3, 4, 6]),
                'location': np.random.choice(['Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad'])
            })
        
        return pd.DataFrame(activities)

    def generate_realistic_interactions(self, users_df, activities_df):
        interactions = []
        
        for _, user in users_df.iterrows():
            user_persona = user['persona']
            preferred_categories = self.user_personas[user_persona]
            
            # Generate 15-30 interactions per user (more realistic)
            n_interactions = np.random.randint(15, 31)
            
            # 70% interactions with preferred categories, 30% random
            preferred_activities = activities_df[activities_df['category'].isin(preferred_categories)]
            random_activities = activities_df[~activities_df['category'].isin(preferred_categories)]
            
            # Select activities
            n_preferred = int(n_interactions * 0.7)
            n_random = n_interactions - n_preferred
            
            selected_preferred = preferred_activities.sample(min(n_preferred, len(preferred_activities)), replace=True)
            selected_random = random_activities.sample(min(n_random, len(random_activities)), replace=True) if len(random_activities) > 0 else pd.DataFrame()
            
            selected_activities = pd.concat([selected_preferred, selected_random])
            
            for _, activity in selected_activities.iterrows():
                # Generate more realistic ratings
                if activity['category'] in preferred_categories:
                    # Higher ratings for preferred categories
                    rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                else:
                    # Mixed ratings for non-preferred categories
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                
                interactions.append({
                    'user_id': user['user_id'],
                    'activity_id': activity['activity_id'],
                    'rating': rating
                })
        
        return pd.DataFrame(interactions)

    def generate_all(self):
        users = self.generate_users()
        activities = self.generate_activities()
        interactions = self.generate_realistic_interactions(users, activities)
        return users, activities, interactions

if __name__ == "__main__":
    generator = ImprovedSyntheticDataGenerator()
    users, activities, interactions = generator.generate_all()
    
    # Save improved data
    users.to_csv("data/processed/users_improved.csv", index=False)
    activities.to_csv("data/processed/activities_improved.csv", index=False)
    interactions.to_csv("data/processed/interactions_improved.csv", index=False)
    
    print("✅ Improved synthetic data generated!")
    print(f"📊 Users: {len(users)}, Activities: {len(activities)}, Interactions: {len(interactions)}")
    print(f"🎯 Sparsity: {1 - len(interactions) / (len(users) * len(activities)):.3f}")
    print("💾 Saved to data/processed/ with '_improved' suffix")
