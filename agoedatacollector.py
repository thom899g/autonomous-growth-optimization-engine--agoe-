"""
Data Collection Module
Collects growth metrics from multiple sources with robust error handling and validation.
Primary data source is Firestore, with fallback to local cache for resilience.
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from agoe.core.config import get_config, MetricCategory

logger = logging.getLogger(__name__)

@dataclass
class MetricDataPoint:
    """Type-safe metric data point with validation"""
    metric_id: str
    category: MetricCategory
    value: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Validate value is numeric
        if not isinstance(self.value, (int, float)):
            try:
                self.value = float(self.value)
            except (ValueError, TypeError):
                raise ValueError(f"Metric value must be numeric: {self.value}")
        
        # Validate timestamp
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be datetime object")

class DataSource(Enum):
    """Supported data sources"""
    FIRESTORE = "firestore"
    CACHE = "cache"
    API = "api"
    LOG_FILE = "log_file"

class DataCollector:
    """Main data collection orchestrator with fallback strategies"""
    
    def __init__(self, db_client: Optional[firestore.Client] = None):
        self.config = get_config()
        self.db_client = db_client
        self.cache: Dict[str, List[MetricDataPoint]] = {}
        self.cache_ttl