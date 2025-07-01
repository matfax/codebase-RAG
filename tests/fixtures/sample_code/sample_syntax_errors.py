# Sample Python code with various syntax errors for testing error handling

# Error 1: Missing closing parenthesis
def broken_function(arg1, arg2
    """Function with missing closing parenthesis."""
    return arg1 + arg2

# Error 2: Missing colon in class definition
class IncompleteClass
    """Class missing colon after name."""
    
    def __init__(self, name):
        self.name = name
    
    def get_name(self)
        # Error 3: Missing colon in method definition
        return self.name

# Error 4: Unclosed string literal
invalid_string = "This string is never closed

# Error 5: Invalid indentation
def indentation_error():
    if True:
        print("Correct indentation")
      print("Incorrect indentation")  # Wrong indentation level

# Error 6: Missing import
def use_undefined_module():
    # Error: os module not imported
    return os.getcwd()

# Error 7: Invalid syntax in list comprehension
bad_comprehension = [x for x in range(10 if x % 2 == 0]  # Missing closing parenthesis

# Error 8: Incorrect exception handling
try:
    risky_operation()
except ValueError as e
    # Error: Missing colon
    print(f"Error: {e}")

# Error 9: Invalid decorator syntax
@property
@invalid_decorator(
def decorated_function():
    return "decorated"

# Error 10: Mismatched brackets
data = {
    "key1": "value1",
    "key2": ["item1", "item2",
    "key3": "value3"
}  # Missing closing bracket for list

# Error 11: Invalid assignment
5 = x  # Cannot assign to literal

# Error 12: Unclosed function call
result = some_function(arg1, arg2, arg3

# Error 13: Invalid escape sequence in string
path = "C:\new\folder"  # Invalid escape sequence

# Error 14: Missing closing bracket in dictionary
config = {
    "database": {
        "host": "localhost",
        "port": 5432
    # Missing closing bracket

# Error 15: Invalid lambda syntax
calculate = lambda x, y: x + y if x > 0 else  # Incomplete expression

# Some valid code mixed in to test error recovery
def valid_function():
    """This function should parse correctly."""
    return "I'm valid!"

class ValidClass:
    """This class should also parse correctly."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

# Error 16: Unclosed parentheses in function call
valid_function().upper(

# Error 17: Invalid operator
def invalid_operation():
    return 5 ++ 3  # Invalid operator

# Error 18: Missing comma in function arguments
def function_call_error():
    return max(1 2 3)  # Missing commas

# Some more valid code for testing recovery
def another_valid_function():
    """Another valid function."""
    data = [1, 2, 3, 4, 5]
    return sum(data)

# Error 19: Invalid assignment in comprehension
bad_list = [x = y for x, y in enumerate(range(10))]  # Should use x, y

# Error 20: Unclosed triple quotes
def function_with_docstring():
    """This docstring is never closed
    It continues here
    And here