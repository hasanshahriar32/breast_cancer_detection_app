"""
Database handler for MongoDB operations.
Stores prediction history and retrieval.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from bson import ObjectId


class DatabaseHandler:
    """Handles MongoDB operations for prediction storage."""
    
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI")
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to MongoDB."""
        if not self.uri:
            print("⚠ MONGODB_URI not set - database features disabled")
            return
            
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client.get_database("breast_cancer_analysis")
            self.collection = self.db.get_collection("predictions")
            print("✓ Connected to MongoDB")
        except Exception as e:
            print(f"✗ MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.collection is not None

    def save_prediction(
        self, 
        image_url: str, 
        result: Dict[str, Any], 
        filename: str
    ) -> Optional[str]:
        """
        Save a prediction result to the database.
        
        Args:
            image_url: URL of the uploaded image
            result: Prediction result dictionary
            filename: Original filename
            
        Returns:
            Inserted document ID or None if failed
        """
        if not self.is_connected:
            return None
        
        document = {
            "filename": filename,
            "image_url": image_url,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "benign_probability": result.get("benign_probability"),
            "malignant_probability": result.get("malignant_probability"),
            "attention_metrics": result.get("attention_metrics"),
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            inserted = self.collection.insert_one(document)
            return str(inserted.inserted_id)
        except Exception as e:
            print(f"✗ Error saving prediction: {e}")
            return None

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific prediction by ID.
        
        Args:
            prediction_id: MongoDB document ID
            
        Returns:
            Prediction document or None if not found
        """
        if not self.is_connected:
            return None
        
        try:
            doc = self.collection.find_one({"_id": ObjectId(prediction_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
                # Format timestamp for JSON
                if "timestamp" in doc:
                    doc["timestamp"] = doc["timestamp"].isoformat()
            return doc
        except Exception as e:
            print(f"✗ Error fetching prediction: {e}")
            return None

    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent prediction history.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of prediction documents
        """
        if not self.is_connected:
            return []
        
        try:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                if "timestamp" in doc:
                    doc["timestamp"] = doc["timestamp"].isoformat()
                results.append(doc)
            return results
        except Exception as e:
            print(f"✗ Error fetching history: {e}")
            return []
    
    def delete_prediction(self, prediction_id: str) -> bool:
        """
        Delete a prediction by ID.
        
        Args:
            prediction_id: MongoDB document ID
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            result = self.collection.delete_one({"_id": ObjectId(prediction_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"✗ Error deleting prediction: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.is_connected:
            return {"connected": False, "total": 0}
        
        try:
            total = self.collection.count_documents({})
            benign = self.collection.count_documents({"prediction": "Benign"})
            malignant = self.collection.count_documents({"prediction": "Malignant"})
            return {
                "connected": True,
                "total": total,
                "benign": benign,
                "malignant": malignant
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
