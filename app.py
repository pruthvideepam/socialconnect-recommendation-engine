import sys
import os
sys.path.append(os.path.abspath('.'))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import uvicorn

# Import your model
from models.hybrid_model import HybridRecommendationSystem

# Initialize FastAPI app
app = FastAPI(
    title="SocialConnect Recommendation Engine",
    description="AI-powered social activity recommendation system using hybrid collaborative and content-based filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
recommendation_model = None
activities_df = None
users_df = None

# Pydantic models for API
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID for recommendations")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations (1-50)")
    categories: Optional[List[str]] = Field(None, description="Filter by specific categories")
    min_score: float = Field(2.0, ge=1.0, le=5.0, description="Minimum recommendation score")

class ActivityResponse(BaseModel):
    activity_id: int
    title: str
    category: str
    location: str
    price_tier: str
    duration_hours: int
    predicted_rating: float
    cf_score: float
    cb_score: float
    recommendation_type: str

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[ActivityResponse]
    total_count: int
    model_version: str = "1.0.0"
    filters_applied: Dict

class UserInsightsResponse(BaseModel):
    user_id: int
    insights: Dict
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_users: int
    total_activities: int
    total_interactions: int

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    global recommendation_model, activities_df, users_df
    
    try:
        print("🚀 Loading SocialConnect Recommendation Engine...")
        
        # Load data
        activities_df = pd.read_csv('data/processed/activities_improved.csv')
        interactions_df = pd.read_csv('data/processed/interactions_improved.csv')
        users_df = pd.read_csv('data/processed/users_improved.csv')
        
        # Load or train model
        model_path = 'saved_models/hybrid_model.pkl'
        recommendation_model = HybridRecommendationSystem()
        
        if os.path.exists(model_path):
            recommendation_model.load_model(model_path)
            print("📂 Loaded existing hybrid model")
        else:
            print("🔄 Training new hybrid model...")
            recommendation_model.fit(activities_df, interactions_df)
            recommendation_model.save_model(model_path)
            print("💾 New model trained and saved")
        
        print("✅ SocialConnect API ready!")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise

@app.get("/", response_model=Dict)
async def root():
    """API root endpoint"""
    return {
        "message": "SocialConnect Recommendation Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=recommendation_model is not None and recommendation_model.is_trained,
        total_users=len(users_df) if users_df is not None else 0,
        total_activities=len(activities_df) if activities_df is not None else 0,
        total_interactions=len(recommendation_model.interactions_df) if recommendation_model else 0
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized activity recommendations for a user"""
    
    if not recommendation_model or not recommendation_model.is_trained:
        raise HTTPException(status_code=503, detail="Recommendation model not available")
    
    # Validate user exists
    if request.user_id not in users_df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
    
    try:
        # Get recommendations
        recommendations = recommendation_model.recommend(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
            categories=request.categories,
            min_score=request.min_score
        )
        
        # Convert to response format
        activity_responses = [
            ActivityResponse(**rec) for rec in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=activity_responses,
            total_count=len(activity_responses),
            filters_applied={
                "categories": request.categories,
                "min_score": request.min_score,
                "n_recommendations": request.n_recommendations
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/users/{user_id}/insights", response_model=UserInsightsResponse)
async def get_user_insights(user_id: int):
    """Get insights about a user's preferences and behavior"""
    
    if not recommendation_model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if user_id not in users_df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    try:
        insights = recommendation_model.get_user_insights(user_id)
        return UserInsightsResponse(
            user_id=user_id,
            insights=insights,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights error: {str(e)}")

@app.get("/activities")
async def get_activities(
    category: Optional[str] = Query(None, description="Filter by category"),
    location: Optional[str] = Query(None, description="Filter by location"),
    limit: int = Query(50, ge=1, le=500, description="Number of activities to return")
):
    """Get list of activities with optional filters"""
    
    if activities_df is None:
        raise HTTPException(status_code=503, detail="Activities data not available")
    
    filtered_df = activities_df.copy()
    
    # Apply filters
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    if location:
        filtered_df = filtered_df[filtered_df['location'] == location]
    
    # Limit results
    filtered_df = filtered_df.head(limit)
    
    return {
        "activities": filtered_df.to_dict('records'),
        "total_count": len(filtered_df),
        "filters_applied": {"category": category, "location": location}
    }

@app.get("/categories")
async def get_categories():
    """Get list of all activity categories"""
    if activities_df is None:
        raise HTTPException(status_code=503, detail="Activities data not available")
    
    categories = sorted(activities_df['category'].unique().tolist())
    return {"categories": categories, "count": len(categories)}

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        # Check if components exist (fixed DataFrame boolean issue)
        if recommendation_model is None:
            raise HTTPException(status_code=503, detail="Recommendation model not loaded")
        
        if activities_df is None or activities_df.empty:
            raise HTTPException(status_code=503, detail="Activities data not loaded")
            
        if users_df is None or users_df.empty:
            raise HTTPException(status_code=503, detail="Users data not loaded")
        
        # Check if model is trained and has interactions
        if not hasattr(recommendation_model, 'interactions_df') or recommendation_model.interactions_df is None:
            raise HTTPException(status_code=503, detail="Model interactions not available")
        
        interactions_df = recommendation_model.interactions_df
        
        if interactions_df.empty:
            raise HTTPException(status_code=503, detail="No interaction data available")
        
        # Calculate statistics safely
        stats = {
            "system_status": "healthy",
            "total_users": int(len(users_df)),
            "total_activities": int(len(activities_df)),
            "total_interactions": int(len(interactions_df)),
            "avg_interactions_per_user": round(float(len(interactions_df)) / float(len(users_df)), 1),
            "sparsity": round(1 - (float(len(interactions_df)) / (float(len(users_df)) * float(len(activities_df)))), 3),
            "avg_rating": round(float(interactions_df['rating'].mean()), 2),
            "rating_distribution": interactions_df['rating'].value_counts().to_dict(),
            "categories": activities_df['category'].value_counts().to_dict(),
            "locations": activities_df['location'].value_counts().to_dict(),
            "model_info": {
                "type": "hybrid",
                "cf_weight": getattr(recommendation_model, 'cf_weight', 0.6),
                "cb_weight": getattr(recommendation_model, 'cb_weight', 0.4),
                "is_trained": getattr(recommendation_model, 'is_trained', False)
            }
        }
        
        return stats
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Return detailed error for debugging
        return {
            "error": f"Statistics calculation failed: {str(e)}",
            "status": "error",
            "suggestion": "Try restarting the server or checking data files"
        }



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
