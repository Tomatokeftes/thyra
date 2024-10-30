import pytest
import zarr
from pathlib import Path
from imzml2zarr.utils.temp_store import multi_temp_stores, single_temp_store, temp_store_factory

def test_single_temp_store():
    """Test that single_temp_store creates and cleans up a single TempStore."""
    with single_temp_store() as store:
        assert isinstance(store, zarr.TempStore)
        assert Path(store.path).exists()  # Store should exist during context

    assert not Path(store.path).exists()  # Store should be cleaned up after context

def test_multi_temp_stores():
    """Test that multi_temp_stores creates multiple TempStores and cleans them up."""
    with multi_temp_stores(3) as stores:
        assert len(stores) == 3  # Should create 3 TempStores
        for store in stores:
            assert isinstance(store, zarr.TempStore)
            assert Path(store.path).exists()  # Each store should exist during context

    for store in stores:
        assert not Path(store.path).exists()  # Each store should be cleaned up after context

def test_temp_store_factory_single():
    """Test temp_store_factory with a single store."""
    get_temp_stores = temp_store_factory()
    with get_temp_stores(1) as stores:
        assert len(stores) == 1
        assert isinstance(stores[0], zarr.TempStore)
        assert Path(stores[0].path).exists()

    assert not Path(stores[0].path).exists()

def test_temp_store_factory_multiple():
    """Test temp_store_factory with multiple stores."""
    get_temp_stores = temp_store_factory()
    with get_temp_stores(5) as stores:
        assert len(stores) == 5
        for store in stores:
            assert isinstance(store, zarr.TempStore)
            assert Path(store.path).exists()

    for store in stores:
        assert not Path(store.path).exists()

def test_zero_or_negative_count():
    """Test that multi_temp_stores with a count of zero or negative raises AssertionError."""
    with pytest.raises(AssertionError):
        with multi_temp_stores(0):
            pass
    with pytest.raises(AssertionError):
        with multi_temp_stores(-1):
            pass
