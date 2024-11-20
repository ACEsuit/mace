import torch
import psutil
import logging
import os
import gc
import tracemalloc
from typing import Any, Optional
from . import torch_geometric
from torch.utils.data._utils.worker import _worker_loop
from functools import partial

class MemoryTrackingDataLoader(torch_geometric.dataloader.DataLoader):
    """DataLoader subclass that tracks memory usage in workers"""
    
    def __init__(self, *args, memory_threshold_mb: float = 1000.0, **kwargs):
        self.memory_threshold_mb = memory_threshold_mb
        super().__init__(*args, **kwargs)
        
        # Override the default worker init function
        if kwargs.get('num_workers', 0) > 0:
            self.worker_init_fn = self._memory_tracking_worker_init
    
    def _memory_tracking_worker_init(self, worker_id: int) -> None:
        """Initialize worker with memory tracking"""
        # Enable garbage collection
        gc.enable()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Setup worker logging
        log_file = f"worker_{worker_id}_memory.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - Worker %(process)d - %(message)s'
        )
        
        logging.info(f"Worker {worker_id} started. PID: {os.getpid()}")
        
        # Log initial memory state
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"Initial RSS: {mem_info.rss / 1024 / 1024:.2f} MB")

    def _check_memory_usage(self) -> None:
        """Check current memory usage and log if above threshold"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        
        if mem_mb > self.memory_threshold_mb:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            logging.warning(f"Memory usage above threshold: {mem_mb:.2f} MB")
            logging.warning("Top 10 memory allocations:")
            for stat in top_stats[:10]:
                logging.warning(stat)
                
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()  # If using GPU

    def __iter__(self):
        if self.num_workers == 0:
            return super().__iter__()
        
        # Wrap the original worker loop with memory tracking
        original_worker_loop = torch.utils.data._utils.worker._worker_loop
        
        def memory_tracking_worker_loop(*args, **kwargs):
            try:
                for i, data in enumerate(original_worker_loop(*args, **kwargs)):
                    self._check_memory_usage()
                    yield data
            except Exception as e:
                logging.error(f"Worker error: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        # Replace the worker loop
        torch.utils.data._utils.worker._worker_loop = memory_tracking_worker_loop
        
        try:
            return super().__iter__()
        finally:
            # Restore original worker loop
            torch.utils.data._utils.worker._worker_loop = original_worker_loop

# Usage example:
def create_memory_tracked_loader(dataset, batch_size, num_workers, **kwargs):
    """Create a DataLoader with memory tracking"""
    return MemoryTrackingDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        memory_threshold_mb=1000.0,  # 1GB threshold
        worker_init_fn=None,  # Will be overridden by the class
        **kwargs
    )

# Error handler for worker processes
def worker_error_callback(worker_id: int, exc_info: Any) -> None:
    exc_type, exc_value, exc_traceback = exc_info
    logging.error(f"Worker {worker_id} failed with error: {exc_type.__name__}: {exc_value}")
    logging.error("Traceback:", exc_info=True)
