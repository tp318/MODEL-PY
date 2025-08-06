import os
import uuid
import requests
import logging
from typing import Optional, Tuple, Union
from urllib.parse import urlparse, unquote

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_extension(url: Union[str, object]) -> str:
    """Extract file extension from URL"""
    try:
        url_str = str(url)  # Convert any URL-like object to string
        parsed = urlparse(url_str)
        path = parsed.path
        # Handle URLs with query parameters
        if '?' in path:
            path = path.split('?')[0]
        # Get the file extension
        _, ext = os.path.splitext(path)
        return ext.lower() or '.bin'
    except Exception as e:
        logger.error(f"Error getting file extension from {url}: {e}")
        return '.bin'

def generate_unique_filename(url: Union[str, object], output_dir: str) -> str:
    """Generate a unique filename for the downloaded file"""
    try:
        ext = get_file_extension(url)
        filename = f"doc_{uuid.uuid4().hex}{ext}"
        return os.path.join(output_dir, filename)
    except Exception as e:
        logger.error(f"Error generating unique filename: {e}")
        return os.path.join(output_dir, f"doc_{uuid.uuid4().hex}.bin")

def download_file(url: Union[str, object], output_dir: str, timeout: int = 30) -> Tuple[str, str]:
    """
    Download a file from a URL and save it to the specified directory.
    
    Args:
        url: The URL of the file to download (can be string or any URL-like object)
        output_dir: The directory where the file should be saved
        timeout: Request timeout in seconds
        
    Returns:
        Tuple[str, str]: A tuple containing (file_path, content_type)
        
    Raises:
        ValueError: If the URL is invalid or the file cannot be downloaded
        requests.exceptions.RequestException: For network-related errors
    """
    file_path = None
    try:
        # Convert URL to string if it's not already
        url_str = str(url)  # This will call __str__ on HttpUrl objects
        logger.info(f"Attempting to download file from: {url_str[:100]}...")  # Log first 100 chars
        
        # Validate URL
        parsed_url = urlparse(url_str)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError(f"Invalid URL format: {url_str}")
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        file_path = generate_unique_filename(url_str, output_dir)
        logger.info(f"Saving to: {file_path}")
        
        # Download the file with streaming
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with requests.get(url_str, stream=True, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            
            # Get content type from response headers
            content_type = response.headers.get('content-type', 'application/octet-stream')
            logger.info(f"Content-Type: {content_type}")
            
            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
        logger.info(f"Successfully downloaded {os.path.getsize(file_path)} bytes to {file_path}")
        return file_path, content_type
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading {url}: {e}")
        raise ValueError(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}", exc_info=True)
        raise ValueError(f"Failed to download file: {str(e)}")
    finally:
        # Clean up partially downloaded file if an error occurred
        if 'response' in locals() and response is not None and not response.ok and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up partially downloaded file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")