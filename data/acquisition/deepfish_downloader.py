#!/usr/bin/env python3
"""
DeepFish Counting Dataset Downloader for Paperspace
Downloads, extracts, converts to JPG, and saves to mounted DigitalOcean Spaces
Includes robust error handling for chunked encoding errors
Based on provided functional deepfish_downloader_paperspace.py
"""

import os
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import tarfile
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import time
import hashlib
import json
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DO bucket path
DO_BUCKET_PATH = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')


class DeepFishDownloader:
    """Download and process DeepFish counting dataset for Paperspace"""
    
    # Dataset URL from the DeepFish website
    DATASET_URL = "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"
    
    def __init__(self, output_dir: str = None, chunk_size: int = 8192, 
                 max_retries: int = 5, timeout: int = 300):
        """
        Initialize the downloader
        
        Args:
            output_dir: Output directory (mounted DO Spaces path)
            chunk_size: Download chunk size in bytes (default: 8KB)
            max_retries: Maximum number of download retry attempts
            timeout: Request timeout in seconds
        """
        self.output_dir = Path(output_dir or f"{DO_BUCKET_PATH}/images/deepfish")
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a robust session with retry logic
        self.session = self._create_robust_session()
        
        # Statistics
        self.stats = {
            'total_downloaded': 0,
            'total_extracted': 0,
            'total_converted': 0,
            'start_time': time.time()
        }
        
    def _create_robust_session(self):
        """
        Create a requests session with retry logic and connection pooling
        
        Returns:
            Configured requests Session object
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=2,  # Exponential backoff: 0, 2, 4, 8, 16 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False
        )
        
        # Mount the adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
            pool_block=False
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers to avoid potential blocking
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        return session
    
    def save_checkpoint(self, state: Dict):
        """Save download state for resumption"""
        checkpoint_path = self.checkpoint_dir / "state.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load previous download state if exists"""
        checkpoint_path = self.checkpoint_dir / "state.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    state = json.load(f)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                return state
            except:
                return None
        return None
    
    def download_with_resume(self, url: str, output_path: Path, expected_size: int = None) -> bool:
        """
        Download a file with resume capability and robust error handling
        
        Args:
            url: URL to download from
            output_path: Path to save the file
            expected_size: Expected file size for validation (optional)
            
        Returns:
            True if download successful, False otherwise
        """
        output_path = Path(output_path)
        temp_path = output_path.with_suffix('.part')
        
        # Check if file already exists and is complete
        if output_path.exists():
            if expected_size and output_path.stat().st_size == expected_size:
                logger.info(f"File already exists and is complete: {output_path}")
                return True
            logger.info(f"Existing file found but may be incomplete, re-downloading...")
            output_path.unlink()
        
        # Resume from partial download if available
        resume_byte_pos = 0
        if temp_path.exists():
            resume_byte_pos = temp_path.stat().st_size
            logger.info(f"Resuming download from byte {resume_byte_pos}")
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                # Set resume header
                headers = {}
                if resume_byte_pos > 0:
                    headers['Range'] = f'bytes={resume_byte_pos}-'
                
                # Make request with streaming
                logger.info(f"Download attempt {attempt + 1}/{self.max_retries}")
                response = self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Get total size
                if 'content-length' in response.headers:
                    total_size = int(response.headers['content-length'])
                elif 'content-range' in response.headers:
                    # Format: bytes start-end/total
                    total_size = int(response.headers['content-range'].split('/')[-1])
                else:
                    total_size = expected_size or 0
                
                # Adjust for resumed download
                if resume_byte_pos > 0:
                    total_size += resume_byte_pos
                
                # Download with progress bar
                mode = 'ab' if resume_byte_pos > 0 else 'wb'
                with open(temp_path, mode) as f:
                    with tqdm(
                        desc="Downloading",
                        initial=resume_byte_pos,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        downloaded_this_session = 0
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:  # Filter out keep-alive chunks
                                size = f.write(chunk)
                                pbar.update(size)
                                downloaded_this_session += size
                                
                                # Flush periodically to avoid data loss
                                if downloaded_this_session % (self.chunk_size * 100) == 0:
                                    f.flush()
                                    os.fsync(f.fileno())
                
                # Verify download completed
                final_size = temp_path.stat().st_size
                if total_size > 0 and final_size < total_size:
                    logger.warning(f"Download incomplete: {final_size}/{total_size} bytes")
                    resume_byte_pos = final_size
                    attempt += 1
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # Move temp file to final location
                temp_path.rename(output_path)
                logger.info(f"Download complete: {output_path}")
                return True
                
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                logger.error(f"Download error: {type(e).__name__}: {e}")
                
                # Update resume position if partial file exists
                if temp_path.exists():
                    resume_byte_pos = temp_path.stat().st_size
                
                attempt += 1
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Download failed.")
                    return False
                    
            except KeyboardInterrupt:
                logger.warning("Download interrupted by user")
                return False
                
            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    def detect_archive_type(self, archive_path: Path) -> Optional[str]:
        """
        Detect the type of archive file
        
        Args:
            archive_path: Path to the archive file
            
        Returns:
            'zip', 'tar', 'tar.gz', 'tar.bz2', 'tar.xz', or None
        """
        archive_path = Path(archive_path)
        
        # Check by extension first
        if archive_path.suffix.lower() == '.zip':
            return 'zip'
        elif archive_path.suffix.lower() in ['.tar', '.tgz']:
            return 'tar'
        elif archive_path.name.lower().endswith('.tar.gz'):
            return 'tar.gz'
        elif archive_path.name.lower().endswith('.tar.bz2'):
            return 'tar.bz2'
        elif archive_path.name.lower().endswith('.tar.xz'):
            return 'tar.xz'
        
        # Check by content (magic numbers)
        try:
            with open(archive_path, 'rb') as f:
                magic = f.read(8)
                
                # ZIP magic number: PK\x03\x04 or PK\x05\x06 or PK\x07\x08
                if magic[:2] == b'PK':
                    return 'zip'
                
                # TAR files (various compressions)
                # Gzip: \x1f\x8b
                if magic[:2] == b'\x1f\x8b':
                    return 'tar.gz'
                
                # Bzip2: BZ
                if magic[:2] == b'BZ':
                    return 'tar.bz2'
                
                # XZ: \xfd7zXZ\x00
                if magic[:6] == b'\xfd7zXZ\x00':
                    return 'tar.xz'
                
                # Uncompressed TAR (check for ustar signature at offset 257)
                f.seek(257)
                ustar = f.read(5)
                if ustar == b'ustar':
                    return 'tar'
        
        except Exception as e:
            logger.error(f"Error detecting archive type: {e}")
        
        return None
    
    def extract_tar(self, tar_path: Path, extract_dir: Path, compression: str = None) -> Optional[Path]:
        """
        Extract a TAR archive (with optional compression)
        
        Args:
            tar_path: Path to the TAR file
            extract_dir: Directory to extract files to
            compression: Compression type ('gz', 'bz2', 'xz', or None)
            
        Returns:
            Path to the extraction directory or None
        """
        try:
            # Determine mode based on compression
            if compression == 'gz':
                mode = 'r:gz'
            elif compression == 'bz2':
                mode = 'r:bz2'
            elif compression == 'xz':
                mode = 'r:xz'
            else:
                mode = 'r'
            
            with tarfile.open(tar_path, mode) as tar_ref:
                # Get total number of members for progress bar
                members = tar_ref.getmembers()
                total_files = len(members)
                
                compression_str = f" ({compression})" if compression else ""
                logger.info(f"Extracting {total_files} files from TAR{compression_str} archive...")
                
                with tqdm(total=total_files, desc=f"Extracting TAR{compression_str}") as pbar:
                    for member in members:
                        try:
                            tar_ref.extract(member, extract_dir, set_attrs=False)
                        except Exception as e:
                            logger.warning(f"Error extracting {member.name}: {e}")
                        pbar.update(1)
            
            self.stats['total_extracted'] = total_files
            return extract_dir
            
        except tarfile.TarError as e:
            logger.error(f"Error: Corrupted TAR file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during TAR extraction: {e}")
            return None
    
    def convert_images_to_jpg(self, source_dir: Path) -> Dict[str, int]:
        """
        Convert all images in directory tree to JPG and organize
        
        Args:
            source_dir: Source directory containing images
            
        Returns:
            Statistics dictionary
        """
        stats = {'converted': 0, 'skipped': 0, 'errors': 0}
        
        # Image extensions to process
        image_extensions = {'.png', '.bmp', '.tiff', '.tif', '.jpeg', '.jpg'}
        
        logger.info(f"Converting images in {source_dir}")
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_dir.rglob(f'*{ext}'))
            image_files.extend(source_dir.rglob(f'*{ext.upper()}'))
        
        with tqdm(total=len(image_files), desc="Converting images") as pbar:
            for img_path in image_files:
                try:
                    # Create relative path structure in output
                    rel_path = img_path.relative_to(source_dir)
                    
                    # Generate new filename with .jpg extension
                    new_name = img_path.stem + '.jpg'
                    output_subdir = self.output_dir / rel_path.parent
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    output_path = output_subdir / new_name
                    
                    # Skip if already converted
                    if output_path.exists():
                        stats['skipped'] += 1
                        pbar.update(1)
                        continue
                    
                    # Open and convert image
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary
                        if img.mode not in ('RGB', 'L'):
                            img = img.convert('RGB')
                        
                        # Save as JPEG with optimization
                        img.save(output_path, 'JPEG', quality=95, optimize=True)
                        stats['converted'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error converting {img_path}: {e}")
                    stats['errors'] += 1
                
                pbar.update(1)
        
        self.stats['total_converted'] = stats['converted']
        return stats
    
    def process_dataset(self) -> bool:
        """
        Complete pipeline to download, extract, and convert DeepFish dataset
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check checkpoint
            checkpoint = self.load_checkpoint()
            if checkpoint and checkpoint.get('completed'):
                logger.info("Dataset already processed according to checkpoint")
                return True
            
            # Step 1: Download dataset
            archive_path = self.output_dir / "DeepFish.tar"
            
            if not archive_path.exists():
                logger.info(f"Downloading DeepFish dataset from {self.DATASET_URL}")
                success = self.download_with_resume(self.DATASET_URL, archive_path)
                
                if not success:
                    logger.error("Failed to download dataset")
                    return False
                
                self.stats['total_downloaded'] = archive_path.stat().st_size
                self.save_checkpoint({'downloaded': True, 'archive_path': str(archive_path)})
            else:
                logger.info(f"Dataset archive already exists: {archive_path}")
            
            # Step 2: Extract dataset
            extract_dir = self.output_dir / "extracted"
            
            if not extract_dir.exists() or not any(extract_dir.iterdir()):
                logger.info(f"Extracting dataset to {extract_dir}")
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                archive_type = self.detect_archive_type(archive_path)
                
                if archive_type == 'tar':
                    result = self.extract_tar(archive_path, extract_dir, None)
                elif archive_type == 'tar.gz':
                    result = self.extract_tar(archive_path, extract_dir, 'gz')
                else:
                    logger.error(f"Unsupported archive type: {archive_type}")
                    return False
                
                if result is None:
                    logger.error("Failed to extract dataset")
                    return False
                
                self.save_checkpoint({'downloaded': True, 'extracted': True, 
                                     'extract_dir': str(extract_dir)})
            else:
                logger.info(f"Dataset already extracted to {extract_dir}")
            
            # Step 3: Convert images to JPG and organize
            conversion_stats = self.convert_images_to_jpg(extract_dir)
            logger.info(f"Conversion complete: {conversion_stats}")
            
            # Step 4: Clean up temporary files (optional)
            if extract_dir.exists():
                logger.info("Cleaning up extracted files...")
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)
            
            # Save final statistics
            self.stats['end_time'] = time.time()
            self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
            self.save_final_report()
            
            # Mark as completed
            self.save_checkpoint({'completed': True, 'stats': self.stats})
            
            logger.info("DeepFish dataset processing complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}", exc_info=True)
            return False
    
    def save_final_report(self):
        """Save final processing report"""
        report = {
            'stats': self.stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_directory': str(self.output_dir)
        }
        
        report_path = self.output_dir / f"processing_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved: {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Download and process DeepFish dataset')
    parser.add_argument('--output-dir', type=str, 
                       default=f"{DO_BUCKET_PATH}/images/deepfish",
                       help='Output directory for processed dataset')
    parser.add_argument('--chunk-size', type=int, default=8192,
                       help='Download chunk size in bytes')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum number of download retries')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Create downloader and process dataset
    downloader = DeepFishDownloader(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        timeout=args.timeout
    )
    
    success = downloader.process_dataset()
    
    if success:
        logger.info("Dataset processing completed successfully!")
        return 0
    else:
        logger.error("Dataset processing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
