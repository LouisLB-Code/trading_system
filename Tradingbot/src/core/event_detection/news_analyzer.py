# src/core/event_detection/news_analyzer.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
from transformers import pipeline

@dataclass
class NewsEvent:
    title: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float
    sentiment_score: float
    impact_prediction: float
    categories: List[str]

class NewsAnalyzer:
    """Analyzes financial news and its market impact"""
    
    def __init__(self, config):
        self.config = config
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL)
        self.event_history = []
        self.logger = logging.getLogger(__name__)
        
    async def analyze_news(self, news_data: List[Dict]) -> List[NewsEvent]:
        """Analyze news events and their potential impact"""
        try:
            events = []
            for news_item in news_data:
                # Calculate relevance
                relevance = self._calculate_relevance(news_item)
                if relevance < self.config.MIN_RELEVANCE_SCORE:
                    continue
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(news_item)
                
                # Predict impact
                impact = await self._predict_impact(news_item, sentiment)
                
                # Categorize news
                categories = self._categorize_news(news_item)
                
                event = NewsEvent(
                    title=news_item['title'],
                    content=news_item['content'],
                    source=news_item['source'],
                    timestamp=news_item['timestamp'],
                    relevance_score=relevance,
                    sentiment_score=sentiment,
                    impact_prediction=impact,
                    categories=categories
                )
                
                events.append(event)
                self.event_history.append(event)
                
            return events
            
        except Exception as e:
            self.logger.error(f"News analysis error: {str(e)}")
            raise
    
    def _calculate_relevance(self, news_item: Dict) -> float:
        """Calculate news relevance score"""
        # Implementation using keyword matching and entity recognition
        pass

# src/core/event_detection/sentiment_analyzer.py

from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

@dataclass
class MarketSentiment:
    """Market sentiment metrics"""
    overall_sentiment: float  # -1 to 1
    uncertainty: float
    momentum: float
    fear_greed_index: float
    sector_sentiment: Dict[str, float]
    timestamp: datetime

class SentimentAnalyzer:
    """Analyzes market sentiment from multiple sources"""
    
    def __init__(self, config):
        self.config = config
        self.sentiment_model = self._build_sentiment_model()
        self.sentiment_history = []
        
    async def analyze_sentiment(self,
                              news_events: List[NewsEvent],
                              social_data: Dict,
                              market_data: pd.DataFrame) -> MarketSentiment:
        """Analyze overall market sentiment"""
        try:
            # Analyze different sentiment sources
            news_sentiment = self._analyze_news_sentiment(news_events)
            social_sentiment = self._analyze_social_sentiment(social_data)
            technical_sentiment = self._analyze_technical_sentiment(market_data)
            
            # Combine sentiment scores
            overall_sentiment = self._combine_sentiment_scores([
                (news_sentiment, self.config.NEWS_WEIGHT),
                (social_sentiment, self.config.SOCIAL_WEIGHT),
                (technical_sentiment, self.config.TECHNICAL_WEIGHT)
            ])
            
            # Calculate additional metrics
            uncertainty = self._calculate_uncertainty(news_events, market_data)
            momentum = self._calculate_sentiment_momentum()
            fear_greed = self._calculate_fear_greed_index(market_data)
            
            # Calculate sector-specific sentiment
            sector_sentiment = self._calculate_sector_sentiment(news_events)
            
            return MarketSentiment(
                overall_sentiment=overall_sentiment,
                uncertainty=uncertainty,
                momentum=momentum,
                fear_greed_index=fear_greed,
                sector_sentiment=sector_sentiment,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            raise

# src/core/event_detection/anomaly_detector.py

from typing import List, Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
import logging

@dataclass
class MarketAnomaly:
    """Market anomaly event"""
    type: str
    severity: float
    confidence: float
    affected_metrics: List[str]
    timestamp: datetime
    description: str
    suggested_actions: List[str]

class AnomalyDetector:
    """Detects market anomalies and unusual patterns"""
    
    def __init__(self, config):
        self.config = config
        self.models = self._initialize_models()
        self.anomaly_history = []
        self.logger = logging.getLogger(__name__)
        
    async def detect_anomalies(self,
                             market_data: pd.DataFrame,
                             order_book: Dict,
                             sentiment: MarketSentiment) -> List[MarketAnomaly]:
        """Detect market anomalies across different dimensions"""
        try:
            anomalies = []
            
            # Price anomalies
            price_anomalies = self._detect_price_anomalies(market_data)
            anomalies.extend(price_anomalies)
            
            # Volume anomalies
            volume_anomalies = self._detect_volume_anomalies(market_data)
            anomalies.extend(volume_anomalies)
            
            # Order book anomalies
            orderbook_anomalies = self._detect_orderbook_anomalies(order_book)
            anomalies.extend(orderbook_anomalies)
            
            # Sentiment anomalies
            sentiment_anomalies = self._detect_sentiment_anomalies(sentiment)
            anomalies.extend(sentiment_anomalies)
            
            # Filter and rank anomalies
            significant_anomalies = self._filter_anomalies(anomalies)
            ranked_anomalies = self._rank_anomalies(significant_anomalies)
            
            # Store in history
            self.anomaly_history.extend(ranked_anomalies)
            
            return ranked_anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {str(e)}")
            raise
