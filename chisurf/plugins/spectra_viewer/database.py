import os
import sqlite3
import json
import numpy as np
from pathlib import Path

class SpectraDatabase:
    """A class to manage the storage and retrieval of spectra data using SQLite."""

    def __init__(self, db_path=None):
        """Initialize the database connection.

        Args:
            db_path (str, optional): Path to the SQLite database file. If None, uses the default path.
        """
        if db_path is None:
            # Use the default path in the plugin directory
            plugin_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            db_path = plugin_dir / "spectra.db"

        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_tables(self):
        """Create the necessary tables if they don't exist."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS item_types (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            display_name TEXT
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            name TEXT,
            type_id INTEGER,
            description TEXT,
            FOREIGN KEY (type_id) REFERENCES item_types(id),
            UNIQUE (name, type_id)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS optical_properties (
            id INTEGER PRIMARY KEY,
            item_id INTEGER,
            property_name TEXT,
            property_value TEXT,
            FOREIGN KEY (item_id) REFERENCES items(id),
            UNIQUE (item_id, property_name)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS spectra (
            id INTEGER PRIMARY KEY,
            item_id INTEGER,
            spectrum_type TEXT,  -- 'absorption' or 'emission'
            wavelengths BLOB,    -- numpy array stored as binary
            intensity_values BLOB,         -- numpy array stored as binary
            FOREIGN KEY (item_id) REFERENCES items(id),
            UNIQUE (item_id, spectrum_type)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            item_id INTEGER,
            image_name TEXT,
            image_data BLOB,
            image_format TEXT,
            FOREIGN KEY (item_id) REFERENCES items(id)
        )
        ''')

        self.conn.commit()

    def add_item_type(self, name, display_name):
        """Add a new item type.

        Args:
            name (str): Internal name of the item type
            display_name (str): Display name of the item type

        Returns:
            int: ID of the item type
        """
        self.cursor.execute(
            "INSERT OR IGNORE INTO item_types (name, display_name) VALUES (?, ?)",
            (name, display_name)
        )
        self.conn.commit()

        # Get the ID of the item type
        self.cursor.execute(
            "SELECT id FROM item_types WHERE name = ?",
            (name,)
        )
        return self.cursor.fetchone()[0]

    def add_item(self, name, type_id, description=""):
        """Add a new item.

        Args:
            name (str): Name of the item
            type_id (int): ID of the item type
            description (str, optional): Description of the item

        Returns:
            int: ID of the item
        """
        self.cursor.execute(
            "INSERT OR REPLACE INTO items (name, type_id, description) VALUES (?, ?, ?)",
            (name, type_id, description)
        )
        self.conn.commit()

        # Get the ID of the item
        self.cursor.execute(
            "SELECT id FROM items WHERE name = ? AND type_id = ?",
            (name, type_id)
        )
        return self.cursor.fetchone()[0]

    def add_optical_property(self, item_id, property_name, property_value):
        """Add an optical property to an item.

        Args:
            item_id (int): ID of the item
            property_name (str): Name of the property
            property_value (str): Value of the property
        """
        self.cursor.execute(
            "INSERT OR REPLACE INTO optical_properties (item_id, property_name, property_value) VALUES (?, ?, ?)",
            (item_id, property_name, property_value)
        )
        self.conn.commit()

    def add_spectrum(self, item_id, spectrum_type, wavelengths, values):
        """Add a spectrum to an item.

        Args:
            item_id (int): ID of the item
            spectrum_type (str): Type of spectrum ('absorption' or 'emission')
            wavelengths (numpy.ndarray): Array of wavelengths
            values (numpy.ndarray): Array of values
        """
        # Convert numpy arrays to binary blobs
        wavelengths_blob = wavelengths.tobytes()
        values_blob = values.tobytes()

        self.cursor.execute(
            "INSERT OR REPLACE INTO spectra (item_id, spectrum_type, wavelengths, intensity_values) VALUES (?, ?, ?, ?)",
            (item_id, spectrum_type, wavelengths_blob, values_blob)
        )
        self.conn.commit()

    def add_image(self, item_id, image_path, image_data=None):
        """Add an image to an item.

        Args:
            item_id (int): ID of the item
            image_path (str): Path to the image file or image name if image_data is provided
            image_data (bytes, optional): Raw image data. If provided, image_path is treated as the image name
        """
        if image_data is None:
            # Read the image file
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Get the image name and format
            image_name = os.path.basename(image_path)
        else:
            # Use the provided image data and name
            image_name = image_path

        # Get the image format
        image_format = os.path.splitext(image_name)[1].lstrip('.')

        # Store the image data in the database
        self.cursor.execute(
            "INSERT INTO images (item_id, image_name, image_data, image_format) VALUES (?, ?, ?, ?)",
            (item_id, image_name, image_data, image_format)
        )
        self.conn.commit()

    def get_item_types(self):
        """Get all item types.

        Returns:
            list: List of tuples (id, name, display_name)
        """
        self.cursor.execute("SELECT id, name, display_name FROM item_types")
        return self.cursor.fetchall()

    def get_items_by_type(self, type_id):
        """Get all items of a specific type.

        Args:
            type_id (int): ID of the item type

        Returns:
            list: List of tuples (id, name, description)
        """
        self.cursor.execute(
            "SELECT id, name, description FROM items WHERE type_id = ?",
            (type_id,)
        )
        return self.cursor.fetchall()

    def get_item_by_name_and_type(self, name, type_id):
        """Get an item by name and type.

        Args:
            name (str): Name of the item
            type_id (int): ID of the item type

        Returns:
            tuple: (id, name, description) or None if not found
        """
        self.cursor.execute(
            "SELECT id, name, description FROM items WHERE name = ? AND type_id = ?",
            (name, type_id)
        )
        return self.cursor.fetchone()

    def get_optical_properties(self, item_id):
        """Get all optical properties of an item.

        Args:
            item_id (int): ID of the item

        Returns:
            dict: Dictionary of property_name: property_value
        """
        self.cursor.execute(
            "SELECT property_name, property_value FROM optical_properties WHERE item_id = ?",
            (item_id,)
        )
        return dict(self.cursor.fetchall())

    def get_spectrum(self, item_id, spectrum_type):
        """Get a spectrum of an item.

        Args:
            item_id (int): ID of the item
            spectrum_type (str): Type of spectrum ('absorption' or 'emission')

        Returns:
            tuple: (wavelengths, values) as numpy arrays, or None if not found
        """
        self.cursor.execute(
            "SELECT wavelengths, intensity_values FROM spectra WHERE item_id = ? AND spectrum_type = ?",
            (item_id, spectrum_type)
        )
        result = self.cursor.fetchone()

        if result is None:
            return None

        # Convert binary blobs back to numpy arrays
        wavelengths_blob, values_blob = result
        wavelengths = np.frombuffer(wavelengths_blob, dtype=np.float64)
        values = np.frombuffer(values_blob, dtype=np.float64)

        return wavelengths, values

    def get_images(self, item_id):
        """Get all images of an item.

        Args:
            item_id (int): ID of the item

        Returns:
            list: List of tuples (image_name, image_data, image_format)
        """
        self.cursor.execute(
            "SELECT image_name, image_data, image_format FROM images WHERE item_id = ?",
            (item_id,)
        )
        return self.cursor.fetchall()

    def get_all_spectra(self):
        """Get all spectra in the database.

        Returns:
            list: List of tuples (item_id, item_name, spectrum_type, wavelengths, values)
        """
        self.cursor.execute("""
            SELECT i.id, i.name, s.spectrum_type, s.wavelengths, s.intensity_values
            FROM items i
            JOIN spectra s ON i.id = s.item_id
        """)

        result = []
        for row in self.cursor.fetchall():
            item_id, item_name, spectrum_type, wavelengths_blob, values_blob = row
            wavelengths = np.frombuffer(wavelengths_blob, dtype=np.float64)
            values = np.frombuffer(values_blob, dtype=np.float64)
            result.append((item_id, item_name, spectrum_type, wavelengths, values))

        return result
