"""Demo script for feature computation and feature store."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_client_factory import DataClientFactory
from features.feature_store import FeatureStore
from data.universe import TOP_200_LIQUID_US_EQUITIES


def demo_single_ticker():
    client = DataClientFactory.create(cache_dir='data/cache')
    fs = FeatureStore(cache_dir='data/features')

    # Fetch and compute features for AAPL
    raw = client.fetch_ticker('AAPL', period='2y')
    df_feats = fs.compute_and_store('AAPL', raw)
    print('Computed features for AAPL:')
    print(df_feats.columns.tolist())
    print(df_feats.tail(3))


def demo_batch_sample():
    client = DataClientFactory.create(cache_dir='data/cache')
    fs = FeatureStore(cache_dir='data/features')

    sample = TOP_200_LIQUID_US_EQUITIES[:10]
    succeeded = fs.batch_compute(sample, client, period='1y', show_progress=True)
    print(f'Succeeded: {succeeded}')


if __name__ == '__main__':
    demo_single_ticker()
    demo_batch_sample()
