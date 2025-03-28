import mysql.connector
from config import MYSQL_CONFIG
import logging
import os
from config import FAISS_INDEX_PATH, IMAGE_STORAGE_PATH
import faiss
import numpy as np

def create_connection():
    """
    Create a database connection with improved error handling

    Returns:
        mysql.connector.connection: Database connection
    """
    try:
        return mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"]
        )
    except mysql.connector.Error as e:
        logging.error(f"Database Connection Error: {e}")
        return None

def create_database():
    """
    Create the child_safety database if it doesn't exist
    """
    try:
        # Connect without specifying a database
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        conn.commit()
        
        logging.info(f"Database {MYSQL_CONFIG['database']} created or already exists")
        
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        logging.error(f"Database Creation Error: {e}")
        return False

def create_metadata_table():
    """
    Create Children_Metadata table with comprehensive constraints
    """
    conn = create_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Children_Metadata (
                child_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INT CHECK (age > 0 AND age < 18),
                gender ENUM('Male', 'Female', 'Other') NOT NULL,
                guardian_contact VARCHAR(20) NOT NULL,
                embedding_id VARCHAR(255) UNIQUE,
                image_url VARCHAR(255) UNIQUE,
                case_status ENUM('Open', 'Resolved', 'Closed') DEFAULT 'Open',
                distinguishing_features TEXT,
                last_known_location VARCHAR(255),
                registration_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                INDEX idx_embedding_id (embedding_id),
                INDEX idx_case_status (case_status)
            )
        ''')
        conn.commit()
        logging.info("Children_Metadata table created successfully")
        return True
    except mysql.connector.Error as e:
        logging.error(f"Table Creation Error: {e}")
        return False
    finally:
        conn.close()

def insert_child_metadata(
    name, 
    age, 
    gender, 
    guardian_contact, 
    embedding_id, 
    image_url, 
    distinguishing_features=None, 
    last_known_location=None
):
    """
    Insert comprehensive child metadata with enhanced validation
    
    Args:
        name (str): Child's full name
        age (int): Child's age
        gender (str): Child's gender
        guardian_contact (str): Guardian's contact information
        embedding_id (str): Unique embedding identifier
        image_url (str): Path to encrypted image
        distinguishing_features (str, optional): Unique identifying features
        last_known_location (str, optional): Last known location
    
    Returns:
        int or False: ID of inserted record or False if insertion fails
    """
    conn = create_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO Children_Metadata 
        (name, age, gender, guardian_contact, embedding_id, image_url, 
         distinguishing_features, last_known_location) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            name, 
            age, 
            gender, 
            guardian_contact, 
            str(embedding_id), 
            image_url,
            distinguishing_features,
            last_known_location
        ))
        conn.commit()
        
        # Return the ID of the inserted record
        inserted_id = cursor.lastrowid
        logging.info(f"Child metadata inserted successfully. ID: {inserted_id}")
        return inserted_id
    except mysql.connector.Error as e:
        logging.error(f"Metadata Insertion Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_child_by_embedding_id(embedding_id):
    """
    Retrieve child metadata by embedding_id
    
    Args:
        embedding_id (int/str): Embedding ID to search
    
    Returns:
        dict: Child metadata or None
    """
    conn = create_connection()
    if not conn:
        logging.error("Database connection failed")
        return None

    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM Children_Metadata WHERE embedding_id = %s"
        cursor.execute(query, (str(embedding_id),))
        result = cursor.fetchone()
        
        if not result:
            logging.warning(f"No child found with Embedding ID: {embedding_id}")
        
        return result
    except mysql.connector.Error as e:
        logging.error(f"Metadata Retrieval Error: {e}")
        return None
    finally:
        conn.close()

def update_case_status(embedding_id, status='Closed'):
    """
    Update case status and optionally clean up associated data
    
    Args:
        embedding_id (int): Unique embedding ID of the child
        status (str): New case status (default 'Closed')
    
    Returns:
        bool: True if update successful, False otherwise
    """
    conn = create_connection()
    if not conn:
        logging.error("Database connection failed")
        return False

    try:
        cursor = conn.cursor()
        
        # First, retrieve current child details
        child_details = get_child_by_embedding_id(embedding_id)
        
        if not child_details:
            logging.error(f"No child found with embedding ID: {embedding_id}")
            return False
        
        # Update case status
        query = "UPDATE Children_Metadata SET case_status = %s WHERE embedding_id = %s"
        cursor.execute(query, (status, str(embedding_id)))
        conn.commit()
        
        # If status is Closed, clean up associated data
        if status == 'Closed':
            # Remove from FAISS index
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                # Remove the embedding from the index
                index.remove_ids(np.array([embedding_id], dtype=np.int64))
                faiss.write_index(index, FAISS_INDEX_PATH)
                logging.info(f"Removed embedding {embedding_id} from FAISS index")
            except Exception as e:
                logging.error(f"Error removing embedding from FAISS: {e}")
            
            # Remove encrypted image
            encrypted_image_path = child_details['image_url']
            if encrypted_image_path and os.path.exists(encrypted_image_path):
                try:
                    # Secure deletion method
                    def secure_delete(file_path, passes=3):
                        import secrets
                        
                        file_size = os.path.getsize(file_path)
                        
                        # Overwrite file multiple times
                        for _ in range(passes):
                            with open(file_path, 'wb') as f:
                                f.write(secrets.token_bytes(file_size))
                        
                        # Then remove the file
                        os.remove(file_path)
                        logging.info(f"Securely deleted: {file_path}")
                    
                    secure_delete(encrypted_image_path)
                except Exception as e:
                    logging.error(f"Error removing encrypted image: {e}")
        
        return True
    
    except mysql.connector.Error as e:
        logging.error(f"Case Status Update Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def search_open_cases():
    """
    Retrieve all open case details
    
    Returns:
        list: List of open case details
    """
    conn = create_connection()
    if not conn:
        logging.error("Database connection failed")
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM Children_Metadata WHERE case_status = 'Open'"
        cursor.execute(query)
        open_cases = cursor.fetchall()
        
        return open_cases
    
    except mysql.connector.Error as e:
        logging.error(f"Open Cases Retrieval Error: {e}")
        return []
    finally:
        conn.close()

# Initialize database setup function
def initialize_database():
    """
    Comprehensive database initialization
    """
    # Create database
    if not create_database():
        logging.error("Failed to create database")
        return False
    
    # Create table
    if not create_metadata_table():
        logging.error("Failed to create metadata table")
        return False
    
    logging.info("Database initialization complete")
    return True

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('database_operations.log')
    ]
)