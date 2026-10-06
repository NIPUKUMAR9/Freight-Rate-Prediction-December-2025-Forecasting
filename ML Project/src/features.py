"""
Feature Engineering & Preprocessing Pipeline for Spotter Freight Rate Model.
Handles missing values, coordinate lookups, spatial, temporal, equipment, and lane features.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd


def compute_haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculates Haversine distance in miles between two lat/lon coordinate arrays."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def compute_bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculates bearing (direction angle in degrees) from pickup to delivery."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(y, x)
    initial_bearing = np.degrees(initial_bearing)
    return (initial_bearing + 360) % 360


class FreightFeaturePipeline:
    def __init__(self):
        self.city_lat_map: Dict[str, float] = {}
        self.city_lon_map: Dict[str, float] = {}
        self.equipment_weight_map: Dict[str, float] = {}
        self.overall_median_weight: float = 32000.0
        self.lane_rpm_stats: Dict[str, Tuple[float, int]] = {}
        self.pickup_rpm_stats: Dict[str, float] = {}
        self.delivery_rpm_stats: Dict[str, float] = {}
        self.global_mean_rpm: float = 2.215
        self.equipment_categories = ["Dry Van", "Reefer", "Flatbed"]
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "FreightFeaturePipeline":
        """Fits coordinate mappings, equipment weight medians, and lane rate stats from training data."""
        df = df_train.copy()
        
        # City lat/lon lookup maps
        for _, row in df.iterrows():
            if row['pickup'] not in self.city_lat_map:
                self.city_lat_map[row['pickup']] = row['pickup_lat']
                self.city_lon_map[row['pickup']] = row['pickup_lon']
            if row['delivery'] not in self.city_lat_map:
                self.city_lat_map[row['delivery']] = row['delivery_lat']
                self.city_lon_map[row['delivery']] = row['delivery_lon']

        # Weight medians by equipment
        eq_weights = df.groupby('equipment')['weight'].median()
        for eq in self.equipment_categories:
            self.equipment_weight_map[eq] = float(eq_weights.get(eq, 32000.0))
        self.overall_median_weight = float(df['weight'].median())

        # RPM target statistics for lane / pickup / delivery encoding
        if 'posted_rate' in df.columns:
            df['rpm'] = df['posted_rate'] / df['distance']
            self.global_mean_rpm = float(df['rpm'].mean())
            
            # Lane stats
            lane_grp = df.groupby(['pickup', 'delivery'])['rpm']
            for (p, d), series in lane_grp:
                self.lane_rpm_stats[f"{p}->{d}"] = (float(series.mean()), int(series.count()))
                
            # Pickup / Delivery stats
            self.pickup_rpm_stats = df.groupby('pickup')['rpm'].mean().to_dict()
            self.delivery_rpm_stats = df.groupby('delivery')['rpm'].mean().to_dict()

        self.is_fitted = True
        return self

    def transform(self, df_input: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Transforms input DataFrame into feature-rich matrix."""
        if not self.is_fitted and not is_training:
            raise RuntimeError("Pipeline must be fitted before transform.")

        df = df_input.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 1. Fill missing Lat/Lon using city coordinate map
        if 'pickup_lat' not in df.columns or df['pickup_lat'].isnull().any():
            df['pickup_lat'] = df['pickup'].map(self.city_lat_map).fillna(38.0)
            df['pickup_lon'] = df['pickup'].map(self.city_lon_map).fillna(-85.0)
        if 'delivery_lat' not in df.columns or df['delivery_lat'].isnull().any():
            df['delivery_lat'] = df['delivery'].map(self.city_lat_map).fillna(38.0)
            df['delivery_lon'] = df['delivery'].map(self.city_lon_map).fillna(-85.0)

        # 2. Impute missing weight
        if 'weight' in df.columns:
            df['weight'] = df.apply(
                lambda r: self.equipment_weight_map.get(r['equipment'], self.overall_median_weight)
                if pd.isna(r['weight']) else r['weight'],
                axis=1
            )
        else:
            df['weight'] = self.overall_median_weight

        # 3. Impute missing market_index & quote_signal if present
        if 'market_index' in df.columns:
            df['market_index'] = df['market_index'].ffill().bfill().fillna(1.0)
        else:
            df['market_index'] = 0.95  # Historical average for Dec

        if 'quote_signal' in df.columns:
            df['quote_signal'] = df['quote_signal'].ffill().bfill().fillna(2.05)
        else:
            df['quote_signal'] = 2.05  # Historical average

        # 4. Spatial & Geographic Features
        df['haversine_dist'] = compute_haversine_distance(
            df['pickup_lat'].values, df['pickup_lon'].values,
            df['delivery_lat'].values, df['delivery_lon'].values
        )
        df['dist_ratio'] = df['distance'] / np.maximum(df['haversine_dist'], 1.0)
        df['delta_lat'] = df['delivery_lat'] - df['pickup_lat']
        df['delta_lon'] = df['delivery_lon'] - df['pickup_lon']
        df['bearing'] = compute_bearing(
            df['pickup_lat'].values, df['pickup_lon'].values,
            df['delivery_lat'].values, df['delivery_lon'].values
        )
        df['midpoint_lat'] = (df['pickup_lat'] + df['delivery_lat']) / 2.0
        df['midpoint_lon'] = (df['pickup_lon'] + df['delivery_lon']) / 2.0

        # 5. Temporal Features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Cyclical Encodings
        df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
        df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

        # 6. Weight & Load Ratios
        df['weight_per_mile'] = df['weight'] / (df['distance'] + 1.0)
        eq_median = df['equipment'].map(self.equipment_weight_map).fillna(self.overall_median_weight)
        df['weight_ratio'] = df['weight'] / (eq_median + 1e-5)

        # 7. Encoded Lane / City Statistics
        lane_str = df['pickup'] + "->" + df['delivery']
        
        def get_lane_enc(lane):
            if lane in self.lane_rpm_stats:
                mean, count = self.lane_rpm_stats[lane]
                # Smoothed Bayesian target encoding: (count * mean + 5 * global_mean) / (count + 5)
                return (count * mean + 5.0 * self.global_mean_rpm) / (count + 5.0)
            return self.global_mean_rpm

        df['lane_encoded_rpm'] = lane_str.map(get_lane_enc)
        df['pickup_encoded_rpm'] = df['pickup'].map(self.pickup_rpm_stats).fillna(self.global_mean_rpm)
        df['delivery_encoded_rpm'] = df['delivery'].map(self.delivery_rpm_stats).fillna(self.global_mean_rpm)

        # 8. Equipment One-Hot Encoding
        for eq in self.equipment_categories:
            df[f"equipment_{eq.replace(' ', '_')}"] = (df['equipment'] == eq).astype(int)

        return df


def get_feature_columns() -> list:
    """Returns exact list of numerical feature column names for model training."""
    return [
        'distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'haversine_dist', 'dist_ratio', 'delta_lat', 'delta_lon', 'bearing', 'midpoint_lat', 'midpoint_lon',
        'market_index', 'quote_signal',
        'dayofweek', 'dayofyear', 'month', 'quarter', 'is_weekend',
        'sin_dayofweek', 'cos_dayofweek', 'sin_dayofyear', 'cos_dayofyear',
        'weight_per_mile', 'weight_ratio',
        'lane_encoded_rpm', 'pickup_encoded_rpm', 'delivery_encoded_rpm',
        'equipment_Dry_Van', 'equipment_Reefer', 'equipment_Flatbed'
    ]
