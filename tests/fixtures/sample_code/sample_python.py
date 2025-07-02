"""
Sample Python code for testing intelligent chunking.

This file contains various Python constructs to test the parsing
and chunking capabilities of the CodeParser service.
"""

import asyncio
import json
import os
from dataclasses import dataclass

# Module-level constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Module-level variable
global_counter = 0


@dataclass
class User:
    """A simple user data class."""

    id: int
    name: str
    email: str | None = None

    def to_dict(self) -> dict[str, any]:
        """Convert user to dictionary."""
        return {"id": self.id, "name": self.name, "email": self.email}


class UserManager:
    """Manages user operations and data storage."""

    def __init__(self, storage_path: str = "/tmp/users.json"):
        """Initialize the user manager.

        Args:
            storage_path: Path to store user data
        """
        self.storage_path = storage_path
        self.users: dict[int, User] = {}
        self._load_users()

    def _load_users(self) -> None:
        """Load users from storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for user_data in data:
                        user = User(**user_data)
                        self.users[user.id] = user
            except Exception as e:
                print(f"Error loading users: {e}")

    def add_user(self, user: User) -> bool:
        """Add a new user to the system.

        Args:
            user: The user to add

        Returns:
            True if user was added successfully
        """
        if user.id in self.users:
            return False

        self.users[user.id] = user
        self._save_users()
        return True

    def get_user(self, user_id: int) -> User | None:
        """Retrieve a user by ID.

        Args:
            user_id: The ID of the user to retrieve

        Returns:
            The user if found, None otherwise
        """
        return self.users.get(user_id)

    def list_users(self) -> list[User]:
        """Get all users in the system.

        Returns:
            List of all users
        """
        return list(self.users.values())

    async def async_operation(self, data: dict[str, any]) -> dict[str, any]:
        """An example async operation.

        Args:
            data: Input data to process

        Returns:
            Processed data
        """
        # Simulate async work
        await asyncio.sleep(0.1)
        return {"processed": True, "data": data}

    def _save_users(self) -> None:
        """Save users to storage file."""
        try:
            with open(self.storage_path, "w") as f:
                user_data = [user.to_dict() for user in self.users.values()]
                json.dump(user_data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")

    @property
    def user_count(self) -> int:
        """Get the number of users."""
        return len(self.users)

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format.

        Args:
            email: Email to validate

        Returns:
            True if email is valid
        """
        return "@" in email and "." in email


def create_user_from_dict(data: dict[str, any]) -> User:
    """Create a User instance from dictionary data.

    Args:
        data: Dictionary containing user data

    Returns:
        A new User instance
    """
    return User(id=data["id"], name=data["name"], email=data.get("email"))


def batch_process_users(users: list[User], operation: str) -> list[dict[str, any]]:
    """Process multiple users with the same operation.

    Args:
        users: List of users to process
        operation: Operation to perform

    Returns:
        List of processing results
    """
    results = []
    for user in users:
        try:
            if operation == "validate":
                result = {"user_id": user.id, "valid": bool(user.email)}
            elif operation == "serialize":
                result = {"user_id": user.id, "data": user.to_dict()}
            else:
                result = {
                    "user_id": user.id,
                    "error": f"Unknown operation: {operation}",
                }

            results.append(result)
        except Exception as e:
            results.append({"user_id": user.id, "error": str(e)})

    return results


# Lambda function example
def calculate_age(birth_year):
    """Calculate age based on birth year."""
    return 2024 - birth_year


# Generator function
def fibonacci_generator(n: int):
    """Generate fibonacci numbers up to n."""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1


# Decorator example
def retry_on_failure(max_attempts: int = 3):
    """Decorator to retry function on failure."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    continue

        return wrapper

    return decorator


@retry_on_failure(max_attempts=5)
def unreliable_operation(data: str) -> str:
    """An operation that might fail."""
    if len(data) < 5:
        raise ValueError("Data too short")
    return data.upper()


# Context manager example
class DatabaseConnection:
    """Example context manager for database connections."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None

    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connection = "mock_connection"
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        self.connection = None


if __name__ == "__main__":
    # Example usage
    manager = UserManager("/tmp/test_users.json")

    # Create some test users
    user1 = User(1, "Alice", "alice@example.com")
    user2 = User(2, "Bob", "bob@example.com")

    manager.add_user(user1)
    manager.add_user(user2)

    print(f"Total users: {manager.user_count}")

    # Test batch processing
    users = manager.list_users()
    results = batch_process_users(users, "validate")
    print(f"Validation results: {results}")
