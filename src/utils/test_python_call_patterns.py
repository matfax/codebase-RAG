"""
Test cases for Python call pattern detection.

This module contains test code examples that demonstrate different types of
function calls that should be detected by the Tree-sitter query patterns.
"""

# Test code samples for pattern validation
TEST_PYTHON_CODE = """
# Direct function calls
print("Hello World")
len(data)
calculate_sum(a, b, c)
process_data()

# Method calls
obj.method()
data.append(item)
result.get_value()
user.save()

# Self method calls
class MyClass:
    def __init__(self):
        self.value = 0

    def process(self):
        self.update_value()
        self.validate()
        return self.get_result()

    def update_value(self):
        self.value += 1

    def validate(self):
        pass

    def get_result(self):
        return self.value

# Chained attribute calls
config.database.connection.execute(query)
user.profile.settings.update(data)
app.logger.handlers[0].setLevel(logging.DEBUG)

# Module function calls
os.path.join(dir1, dir2)
json.dumps(data)
datetime.datetime.now()
math.sqrt(value)

# Subscript method calls
items[0].process()
data['key'].execute()
matrix[i][j].transform()

# Super method calls
class ChildClass(ParentClass):
    def __init__(self):
        super().__init__()
        super().setup()

    def method(self):
        result = super().method()
        return result

# Class method calls
MyClass.class_method()
cls.validate()
User.from_dict(data)

# Dynamic attribute calls
getattr(obj, 'method')()
getattr(user, 'process_data')(args)

# Calls with unpacking
function(*args)
method(**kwargs)
complex_call(*args, **kwargs)

# Nested and complex calls
process(calculate(get_data()))
obj.method(other.get_value())
transform(data.filter(lambda x: x.is_valid()))

# Async function calls
async def async_example():
    # Basic await calls
    result = await fetch_data()
    value = await process_async()

    # Await method calls
    data = await obj.async_method()
    response = await client.get_data()

    # Await self method calls
    await self.async_process()
    await self.save_async()

    # Await chained calls
    result = await user.profile.async_update()
    data = await config.database.async_connect()

    # Asyncio function calls
    results = await asyncio.gather(task1, task2, task3)
    task = asyncio.create_task(coroutine)
    completed, pending = await asyncio.wait(tasks)
    result = await asyncio.wait_for(coro, timeout=5.0)

    # Await asyncio calls
    await asyncio.sleep(1.0)
    result = await asyncio.create_task(coro)

    # Combined async patterns
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    results = await asyncio.gather(*tasks)

    return results

# Asyncio module usage
def main():
    # Asyncio.run pattern
    asyncio.run(async_main())

    # Direct asyncio calls (non-awaited)
    task = asyncio.create_task(async_function())
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(new_loop)
"""


class CallPatternValidator:
    """
    Validator for testing call pattern detection.

    This class can be used to validate that the Tree-sitter patterns
    correctly identify different types of function calls.
    """

    def __init__(self):
        """Initialize the validator."""
        self.expected_patterns = {
            "direct_function_calls": [
                'print("Hello World")',
                "len(data)",
                "calculate_sum(a, b, c)",
                "process_data()",
                "function(*args)",
                "method(**kwargs)",
                "complex_call(*args, **kwargs)",
            ],
            "method_calls": [
                "obj.method()",
                "data.append(item)",
                "result.get_value()",
                "user.save()",
                "items[0].process()",
                "data['key'].execute()",
                "matrix[i][j].transform()",
            ],
            "self_method_calls": ["self.update_value()", "self.validate()", "self.get_result()"],
            "chained_attribute_calls": [
                "config.database.connection.execute(query)",
                "user.profile.settings.update(data)",
                "app.logger.handlers[0].setLevel(logging.DEBUG)",
            ],
            "module_function_calls": ["os.path.join(dir1, dir2)", "json.dumps(data)", "datetime.datetime.now()", "math.sqrt(value)"],
            "super_method_calls": ["super().__init__()", "super().setup()", "super().method()"],
            "class_method_calls": ["MyClass.class_method()", "cls.validate()", "User.from_dict(data)"],
            "dynamic_attribute_calls": ["getattr(obj, 'method')()", "getattr(user, 'process_data')(args)"],
            "nested_calls": [
                "process(calculate(get_data()))",
                "obj.method(other.get_value())",
                "transform(data.filter(lambda x: x.is_valid()))",
            ],
            "await_function_calls": ["await fetch_data()", "await process_async()"],
            "await_method_calls": ["await obj.async_method()", "await client.get_data()", "await asyncio.sleep(1.0)"],
            "await_self_method_calls": ["await self.async_process()", "await self.save_async()"],
            "await_chained_calls": ["await user.profile.async_update()", "await config.database.async_connect()"],
            "asyncio_calls": [
                "asyncio.create_task(coroutine)",
                "asyncio.create_task(async_function())",
                "asyncio.run(async_main())",
                "asyncio.get_event_loop()",
                "asyncio.set_event_loop(new_loop)",
            ],
            "asyncio_await_calls": [
                "await asyncio.gather(task1, task2, task3)",
                "await asyncio.wait(tasks)",
                "await asyncio.wait_for(coro, timeout=5.0)",
                "await asyncio.create_task(coro)",
                "await asyncio.gather(*tasks)",
            ],
        }

    def get_expected_call_count(self) -> dict:
        """
        Get expected call counts for each pattern type.

        Returns:
            Dictionary with expected counts for each call pattern
        """
        counts = {}
        for pattern_type, calls in self.expected_patterns.items():
            counts[pattern_type] = len(calls)

        # Calculate total expected calls
        counts["total_calls"] = sum(counts.values())
        return counts

    def get_test_code(self) -> str:
        """Get the test Python code."""
        return TEST_PYTHON_CODE

    def get_expected_patterns(self) -> dict:
        """Get expected patterns for validation."""
        return self.expected_patterns


# Edge cases and complex scenarios
EDGE_CASE_CODE = """
# Complex nested calls
data.filter(lambda x: x.process()).map(transform).collect()

# Calls in comprehensions
[item.process() for item in data]
{key: value.transform() for key, value in items.items()}

# Calls in conditional expressions
result = obj.method() if condition else other.method()

# Calls as arguments
process(data.get_items(), config.get_settings())

# Calls with generators
consume(item.process() for item in generator)

# Calls in exception handling
try:
    result = risky.operation()
except Exception as e:
    logger.error(e.format_message())

# Calls in decorators (function definitions will be handled separately)
@property
def value(self):
    return self._value.get()

# Calls in with statements
with manager.get_connection() as conn:
    conn.execute(query)

# Multiple calls on same line
x, y = first.get(), second.get()
"""


def get_edge_case_code() -> str:
    """Get edge case test code."""
    return EDGE_CASE_CODE


if __name__ == "__main__":
    validator = CallPatternValidator()

    print("Expected Call Counts:")
    counts = validator.get_expected_call_count()
    for pattern_type, count in counts.items():
        print(f"  {pattern_type}: {count}")

    print(f"\nTest Code Length: {len(validator.get_test_code())} characters")
    print(f"Edge Case Code Length: {len(get_edge_case_code())} characters")
