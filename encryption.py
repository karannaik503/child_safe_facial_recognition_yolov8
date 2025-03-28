from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import os

class ImageEncryptor:
    def __init__(self, key_path='encryption_key.bin'):
        """
        Initialize image encryption
        
        Args:
            key_path (str): Path to store/load encryption key
        """
        self.key_path = key_path
        self.key = self._load_or_generate_key()

    def _load_or_generate_key(self):
        """
        Load existing key or generate a new one
        
        Returns:
            bytes: 32-byte encryption key
        """
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                return f.read()
        
        # Generate new key
        key = get_random_bytes(32)
        with open(self.key_path, 'wb') as f:
            f.write(key)
        
        return key

    def encrypt_image(self, input_path, output_path):
        """
        Encrypt an image file
        
        Args:
            input_path (str): Source image path
            output_path (str): Encrypted image path
        """
        cipher = AES.new(self.key, AES.MODE_GCM)
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        ciphertext, tag = cipher.encrypt_and_digest(data)
        
        with open(output_path, 'wb') as f:
            f.write(cipher.nonce)
            f.write(tag)
            f.write(ciphertext)

    def decrypt_image(self, input_path, output_path):
        """
        Decrypt an encrypted image file
        
        Args:
            input_path (str): Encrypted image path
            output_path (str): Decrypted image path
        """
        with open(input_path, 'rb') as f:
            nonce = f.read(16)
            tag = f.read(16)
            ciphertext = f.read()
        
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        data = cipher.decrypt_and_verify(ciphertext, tag)
        
        with open(output_path, 'wb') as f:
            f.write(data)

# Utility functions for direct use
def encrypt_image(input_path, output_path):
    """
    Convenience function to encrypt image
    
    Args:
        input_path (str): Source image path
        output_path (str): Encrypted image path
    """
    encryptor = ImageEncryptor()
    encryptor.encrypt_image(input_path, output_path)

def decrypt_image(input_path, output_path):
    """
    Convenience function to decrypt image
    
    Args:
        input_path (str): Encrypted image path
        output_path (str): Decrypted image path
    """
    encryptor = ImageEncryptor()
    encryptor.decrypt_image(input_path, output_path)