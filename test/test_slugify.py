import re
import unicodedata

def slugify(text, separator='_', regex_pattern=r'[^-a-z0-9_]+'):
    """
    Convert a string to a slug.
    
    Parameters
    ----------
    text : str
        The string to convert
    separator : str
        The separator to use (default is '_')
    regex_pattern : str
        The regex pattern used to identify characters to replace
        
    Returns
    -------
    str
        A slugified string
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Convert accented characters to their ASCII equivalents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Replace characters matching the regex pattern with the separator
    text = re.sub(regex_pattern, separator, text)
    
    # Replace multiple consecutive separators with a single one
    text = re.sub(f'{separator}+', separator, text)
    
    # Remove leading/trailing separators
    text = text.strip(separator)
    
    return text

def clean_string(s, regex_pattern=r'[^-a-z0-9_]+'):
    """Get a slugified a string."""
    r = slugify(s, separator='_', regex_pattern=regex_pattern)
    return r

# Test case from test_base.py
s1 = "ldldöö_ddd   dd**"
s2 = "ldldoo_ddd_dd"
result = clean_string(s1)
print(f"Input: {s1}")
print(f"Expected: {s2}")
print(f"Result: {result}")
print(f"Test passed: {result == s2}")