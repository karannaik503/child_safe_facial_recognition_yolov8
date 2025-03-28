import os
import shutil
from config import IMAGE_STORAGE_PATH
from encryption import encrypt_image, decrypt_image

def store_encrypted_image(image_url, child_id):
    """
    Store an encrypted image
    
    Args:
        image_url (str): Source image path
        child_id (int): Unique child identifier
    
    Returns:
        str: Path to encrypted image
    """
    # Ensure storage directory exists
    os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)
    
    # Generate encrypted image path
    encrypted_path = os.path.join(IMAGE_STORAGE_PATH, f"{child_id}.enc")
    
    # Encrypt and store image
    encrypt_image(image_url, encrypted_path)
    
    return encrypted_path

def retrieve_encrypted_image(encrypted_path, output_path):
    """
    Retrieve and decrypt an image
    
    Args:
        encrypted_path (str): Encrypted image path
        output_path (str): Decrypted image output path
    """
    decrypt_image(encrypted_path, output_path)