import os
import sys
from contextlib import contextmanager

from graphics_db_server.core.config import BLENDER_LOG_FILE
from graphics_db_server.logging import logger


def cleanup_blender_memory(num_passes: int = 5):  # NOTE: doesn't seem to work
    """Periodically clear Blender memory for batch operations."""
    try:
        import bpy
        import gc

        # Clear scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Purge all orphaned data
        with suppress_blender_logs():
            for i in range(num_passes):
                bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

        # Force Python garbage collection
        gc.collect()

    except ImportError:
        pass
    

def cleanup_scene():
    """
    Aggressively removes all data from the current Blender scene.
    """
    try:
        import bpy
        import gc
        # Go to object mode
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        # Delete all objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # The above only deletes objects from the scene, but the data blocks
        # (meshes, materials, etc.) remain in memory. The following loop
        # purges them completely.
        for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images, bpy.data.curves, bpy.data.cameras, bpy.data.lights]:
            for block in collection:
                collection.remove(block)

    except Exception as e:
        logger.error(e)


@contextmanager
def suppress_blender_logs(log_file_path: str = BLENDER_LOG_FILE):
    """A context manager that redirects stdout and stderr to a file or devnull.

    This is used to suppress verbose console output from Blender operations
    that cannot be controlled through Python's logging module.

    Args:
        log_file_path: If provided, logs are written to this file.
                      If None, logs are discarded to devnull.
    """
    # Save the original stdout and stderr file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Create duplicates of the original file descriptors
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Open log file or devnull depending on parameter
    if log_file_path:
        target_fd = os.open(
            log_file_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
        )
    else:
        target_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        # Redirect stdout and stderr to the target
        os.dup2(target_fd, original_stdout_fd)
        os.dup2(target_fd, original_stderr_fd)

        # Yield control back to the 'with' block
        yield
    finally:
        # Restore the original stdout and stderr
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Close the file descriptors we opened
        os.close(target_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
