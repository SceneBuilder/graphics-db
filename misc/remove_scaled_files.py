#!/usr/bin/env python3

import os
import glob
import argparse

def remove_scaled_files(directory="."):
    pattern = os.path.join(directory, "*_scaled*")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files ending with '_scaled' found in directory: {directory}")
        return
    
    print(f"Found {len(files)} file(s) ending with '_scaled':")
    for file in files:
        print(f"  {file}")
    
    # confirm = input("\nRemove these files? (y/N): ")
    # if confirm.lower() in ['y', 'yes']:
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except OSError as e:
            print(f"Error removing {file}: {e}")
    # else:
    #     print("Operation cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove files with '_scaled' in their names")
    parser.add_argument("directory", nargs="?", default=".", 
                       help="Directory to search for scaled files (default: current directory)")
    
    args = parser.parse_args()
    remove_scaled_files(args.directory)