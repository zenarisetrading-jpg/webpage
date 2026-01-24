"""
Hashing Utilities
=================
Wrapper for bcrypt implementation.
PRD Reference: ORG_USERS_ROLES_PRD.md §15

Security Rules:
- Salt rounds: Auto (bcrypt default is 12)
- Encoding: UTF-8 for all password strings
"""

import bcrypt

def hash_password(plain_password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    Returns: Hash string (including salt and algorithm info)
    """
    # bcrypt requires bytes
    pwd_bytes = plain_password.encode('utf-8')
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored hash.
    Safe against timing attacks.
    """
    pwd_bytes = plain_password.encode('utf-8')
    hash_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(pwd_bytes, hash_bytes)
