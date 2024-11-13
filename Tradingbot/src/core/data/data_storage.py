# File: src/core/data/data_storage.py

import sqlite3
import pandas as pd
import json
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

class DataStorage:
    """Manages persistent storage of market data"""
    
    def __init__(self, config):
        self.config = config
        self.db_path = Path(config.DATA_PATH) / "market_data.db"
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize database connection and tables"""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            self.connection = sqlite3.connect(str(self.db_path))
            await self._create_tables()
            
            self.logger.info(f"Data storage initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Storage initialization error: {str(e)}")
            raise

    async def store_market_data(self,
                              symbol: str,
                              timeframe: str,
                              data: pd.DataFrame):
        """Store market data in database"""
        try:
            # Convert DataFrame to records
            records = data.reset_index().to_dict('records')
            
            # Store in database
            cursor = self.connection.cursor()
            
            for record in records:
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    timeframe,
                    record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Data storage error: {str(e)}")
            raise

    async def get_market_data(self,
                            symbol: str,
                            timeframe: str,
                            start_time: datetime,
                            end_time: datetime) -> pd.DataFrame:
        """Retrieve market data from database"""
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? 
                AND timeframe = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(
                query,
                self.connection,
                params=(
                    symbol,
                    timeframe,
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S')
                ),
                parse_dates=['timestamp']
            )
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data retrieval error: {str(e)}")
            raise

    async def _create_tables(self):
        """Create necessary database tables"""
        try:
            cursor = self.connection.cursor()
            
            # Create market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data 
                ON market_data(symbol, timeframe, timestamp)
            """)
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Table creation error: {str(e)}")
            raise

    async def clean_old_data(self, days_to_keep: int = 30):
        """Clean up old market data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor = self.connection.cursor()
            cursor.execute("""
                DELETE FROM market_data
                WHERE timestamp < ?
            """, (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Data cleanup error: {str(e)}")
            raise
