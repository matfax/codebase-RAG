/**
 * Sample C++ code for testing intelligent chunking.
 * 
 * This file demonstrates C++ specific features including:
 * - Classes, inheritance, and polymorphism
 * - Templates and generic programming
 * - STL containers and algorithms
 * - RAII and smart pointers
 * - Namespaces and operator overloading
 */

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include <functional>
#include <chrono>
#include <mutex>
#include <thread>
#include <future>
#include <optional>
#include <variant>

// Namespace declarations
namespace UserService {
    namespace Models {
        // Forward declarations
        class User;
        class UserPreferences;
        template<typename T> class Repository;
    }
    
    namespace Utils {
        class Logger;
        template<typename K, typename V> class Cache;
    }
}

// Constants and type aliases
namespace UserService {
    
    // Global constants
    constexpr int DEFAULT_CACHE_SIZE = 1000;
    constexpr long long CACHE_TTL_MS = 300000; // 5 minutes
    constexpr size_t MAX_USERNAME_LENGTH = 100;
    constexpr size_t MAX_EMAIL_LENGTH = 255;
    
    // Type aliases
    using UserId = uint64_t;
    using UserPtr = std::shared_ptr<Models::User>;
    using UserMap = std::map<UserId, UserPtr>;
    using ValidationFunction = std::function<bool(const std::string&)>;
    
    // Enums
    enum class UserRole {
        GUEST = 0,
        USER = 1,
        MODERATOR = 2,
        ADMIN = 3
    };
    
    enum class NotificationFrequency {
        NEVER,
        DAILY,
        WEEKLY,
        MONTHLY
    };
    
    // Enum utility functions
    std::string userRoleToString(UserRole role) {
        switch (role) {
            case UserRole::GUEST: return "guest";
            case UserRole::USER: return "user";
            case UserRole::MODERATOR: return "moderator";
            case UserRole::ADMIN: return "admin";
            default: return "unknown";
        }
    }
    
    UserRole stringToUserRole(const std::string& roleStr) {
        if (roleStr == "admin") return UserRole::ADMIN;
        if (roleStr == "moderator") return UserRole::MODERATOR;
        if (roleStr == "user") return UserRole::USER;
        return UserRole::GUEST;
    }
}

// Custom exceptions
namespace UserService {
    
    class UserServiceException : public std::exception {
    private:
        std::string message;
        std::string errorCode;
        
    public:
        UserServiceException(const std::string& msg, const std::string& code = "GENERIC_ERROR") 
            : message(msg), errorCode(code) {}
        
        const char* what() const noexcept override {
            return message.c_str();
        }
        
        const std::string& getErrorCode() const { return errorCode; }
    };
    
    class UserNotFoundException : public UserServiceException {
    public:
        explicit UserNotFoundException(UserId userId) 
            : UserServiceException("User not found with ID: " + std::to_string(userId), "USER_NOT_FOUND") {}
    };
    
    class ValidationException : public UserServiceException {
    public:
        explicit ValidationException(const std::string& message) 
            : UserServiceException(message, "VALIDATION_ERROR") {}
    };
}

// User preferences class
namespace UserService::Models {
    
    class UserPreferences {
    private:
        std::string theme;
        std::string language;
        bool notificationsEnabled;
        NotificationFrequency emailFrequency;
        
    public:
        // Constructors
        UserPreferences() 
            : theme("light"), language("en"), notificationsEnabled(true), 
              emailFrequency(NotificationFrequency::WEEKLY) {}
        
        UserPreferences(const std::string& theme, const std::string& language,
                       bool notifications, NotificationFrequency frequency)
            : theme(theme), language(language), notificationsEnabled(notifications),
              emailFrequency(frequency) {}
        
        // Copy constructor
        UserPreferences(const UserPreferences& other) = default;
        
        // Move constructor
        UserPreferences(UserPreferences&& other) noexcept = default;
        
        // Assignment operators
        UserPreferences& operator=(const UserPreferences& other) = default;
        UserPreferences& operator=(UserPreferences&& other) noexcept = default;
        
        // Destructor
        ~UserPreferences() = default;
        
        // Getters
        const std::string& getTheme() const { return theme; }
        const std::string& getLanguage() const { return language; }
        bool areNotificationsEnabled() const { return notificationsEnabled; }
        NotificationFrequency getEmailFrequency() const { return emailFrequency; }
        
        // Setters
        void setTheme(const std::string& newTheme) { theme = newTheme; }
        void setLanguage(const std::string& newLanguage) { language = newLanguage; }
        void setNotificationsEnabled(bool enabled) { notificationsEnabled = enabled; }
        void setEmailFrequency(NotificationFrequency frequency) { emailFrequency = frequency; }
        
        // Utility methods
        bool operator==(const UserPreferences& other) const {
            return theme == other.theme && 
                   language == other.language &&
                   notificationsEnabled == other.notificationsEnabled &&
                   emailFrequency == other.emailFrequency;
        }
        
        bool operator!=(const UserPreferences& other) const {
            return !(*this == other);
        }
        
        std::string toString() const {
            return "UserPreferences{theme: " + theme + 
                   ", language: " + language +
                   ", notifications: " + (notificationsEnabled ? "true" : "false") + "}";
        }
    };
}

// Main User class
namespace UserService::Models {
    
    class User {
    private:
        UserId id;
        std::string name;
        std::string email;
        UserRole role;
        std::chrono::system_clock::time_point createdAt;
        std::optional<std::chrono::system_clock::time_point> updatedAt;
        UserPreferences preferences;
        mutable std::mutex mutex; // For thread safety
        
    public:
        // Constructors
        User() : id(0), role(UserRole::USER), createdAt(std::chrono::system_clock::now()) {}
        
        User(UserId id, const std::string& name, const std::string& email, UserRole role = UserRole::USER)
            : id(id), name(name), email(email), role(role), 
              createdAt(std::chrono::system_clock::now()) {}
        
        // Copy constructor
        User(const User& other) {
            std::lock_guard<std::mutex> lock(other.mutex);
            id = other.id;
            name = other.name;
            email = other.email;
            role = other.role;
            createdAt = other.createdAt;
            updatedAt = other.updatedAt;
            preferences = other.preferences;
        }
        
        // Move constructor
        User(User&& other) noexcept {
            std::lock_guard<std::mutex> lock(other.mutex);
            id = other.id;
            name = std::move(other.name);
            email = std::move(other.email);
            role = other.role;
            createdAt = other.createdAt;
            updatedAt = other.updatedAt;
            preferences = std::move(other.preferences);
        }
        
        // Assignment operators
        User& operator=(const User& other) {
            if (this != &other) {
                std::lock(mutex, other.mutex);
                std::lock_guard<std::mutex> lock1(mutex, std::adopt_lock);
                std::lock_guard<std::mutex> lock2(other.mutex, std::adopt_lock);
                
                id = other.id;
                name = other.name;
                email = other.email;
                role = other.role;
                createdAt = other.createdAt;
                updatedAt = other.updatedAt;
                preferences = other.preferences;
            }
            return *this;
        }
        
        User& operator=(User&& other) noexcept {
            if (this != &other) {
                std::lock(mutex, other.mutex);
                std::lock_guard<std::mutex> lock1(mutex, std::adopt_lock);
                std::lock_guard<std::mutex> lock2(other.mutex, std::adopt_lock);
                
                id = other.id;
                name = std::move(other.name);
                email = std::move(other.email);
                role = other.role;
                createdAt = other.createdAt;
                updatedAt = other.updatedAt;
                preferences = std::move(other.preferences);
            }
            return *this;
        }
        
        // Destructor
        virtual ~User() = default;
        
        // Getters (thread-safe)
        UserId getId() const {
            std::lock_guard<std::mutex> lock(mutex);
            return id;
        }
        
        std::string getName() const {
            std::lock_guard<std::mutex> lock(mutex);
            return name;
        }
        
        std::string getEmail() const {
            std::lock_guard<std::mutex> lock(mutex);
            return email;
        }
        
        UserRole getRole() const {
            std::lock_guard<std::mutex> lock(mutex);
            return role;
        }
        
        std::chrono::system_clock::time_point getCreatedAt() const {
            std::lock_guard<std::mutex> lock(mutex);
            return createdAt;
        }
        
        std::optional<std::chrono::system_clock::time_point> getUpdatedAt() const {
            std::lock_guard<std::mutex> lock(mutex);
            return updatedAt;
        }
        
        UserPreferences getPreferences() const {
            std::lock_guard<std::mutex> lock(mutex);
            return preferences;
        }
        
        // Setters (thread-safe)
        void setId(UserId newId) {
            std::lock_guard<std::mutex> lock(mutex);
            id = newId;
        }
        
        void setName(const std::string& newName) {
            std::lock_guard<std::mutex> lock(mutex);
            name = newName;
            updatedAt = std::chrono::system_clock::now();
        }
        
        void setEmail(const std::string& newEmail) {
            std::lock_guard<std::mutex> lock(mutex);
            email = newEmail;
            updatedAt = std::chrono::system_clock::now();
        }
        
        void setRole(UserRole newRole) {
            std::lock_guard<std::mutex> lock(mutex);
            role = newRole;
            updatedAt = std::chrono::system_clock::now();
        }
        
        void setPreferences(const UserPreferences& newPreferences) {
            std::lock_guard<std::mutex> lock(mutex);
            preferences = newPreferences;
            updatedAt = std::chrono::system_clock::now();
        }
        
        // Business methods
        std::string getDisplayName() const {
            std::lock_guard<std::mutex> lock(mutex);
            return name + " (" + email + ")";
        }
        
        bool hasAdminPrivileges() const {
            std::lock_guard<std::mutex> lock(mutex);
            return role == UserRole::ADMIN || role == UserRole::MODERATOR;
        }
        
        bool canModerate() const {
            return hasAdminPrivileges();
        }
        
        std::chrono::duration<double> getAccountAge() const {
            std::lock_guard<std::mutex> lock(mutex);
            return std::chrono::system_clock::now() - createdAt;
        }
        
        // Operators
        bool operator==(const User& other) const {
            std::lock(mutex, other.mutex);
            std::lock_guard<std::mutex> lock1(mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.mutex, std::adopt_lock);
            return id == other.id && email == other.email;
        }
        
        bool operator!=(const User& other) const {
            return !(*this == other);
        }
        
        bool operator<(const User& other) const {
            std::lock(mutex, other.mutex);
            std::lock_guard<std::mutex> lock1(mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.mutex, std::adopt_lock);
            return id < other.id;
        }
        
        // Stream operator
        friend std::ostream& operator<<(std::ostream& os, const User& user) {
            std::lock_guard<std::mutex> lock(user.mutex);
            os << "User{id: " << user.id << ", name: \"" << user.name 
               << "\", email: \"" << user.email << "\", role: " 
               << userRoleToString(user.role) << "}";
            return os;
        }
    };
}

// Generic repository template
namespace UserService::Models {
    
    template<typename T, typename IdType = UserId>
    class Repository {
    public:
        virtual ~Repository() = default;
        virtual std::optional<T> findById(IdType id) const = 0;
        virtual std::vector<T> findAll() const = 0;
        virtual T save(const T& entity) = 0;
        virtual void deleteById(IdType id) = 0;
        virtual bool existsById(IdType id) const = 0;
        virtual size_t count() const = 0;
    };
    
    // Specialized user repository interface
    class UserRepository : public Repository<User, UserId> {
    public:
        virtual std::optional<User> findByEmail(const std::string& email) const = 0;
        virtual std::vector<User> findByRole(UserRole role) const = 0;
        virtual std::vector<User> findByNameContaining(const std::string& nameFragment) const = 0;
        virtual std::vector<User> findUsersCreatedAfter(const std::chrono::system_clock::time_point& date) const = 0;
    };
}

// In-memory repository implementation
namespace UserService::Models {
    
    class InMemoryUserRepository : public UserRepository {
    private:
        mutable std::mutex mutex;
        UserMap users;
        UserId nextId;
        
    public:
        InMemoryUserRepository() : nextId(1) {}
        
        std::optional<User> findById(UserId id) const override {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = users.find(id);
            return (it != users.end()) ? std::make_optional(*it->second) : std::nullopt;
        }
        
        std::optional<User> findByEmail(const std::string& email) const override {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = std::find_if(users.begin(), users.end(),
                [&email](const auto& pair) {
                    return pair.second->getEmail() == email;
                });
            return (it != users.end()) ? std::make_optional(*it->second) : std::nullopt;
        }
        
        std::vector<User> findByRole(UserRole role) const override {
            std::lock_guard<std::mutex> lock(mutex);
            std::vector<User> result;
            std::copy_if(users.begin(), users.end(), std::back_inserter(result),
                [role](const auto& pair) {
                    return pair.second->getRole() == role;
                });
            return result;
        }
        
        std::vector<User> findByNameContaining(const std::string& nameFragment) const override {
            std::lock_guard<std::mutex> lock(mutex);
            std::vector<User> result;
            std::copy_if(users.begin(), users.end(), std::back_inserter(result),
                [&nameFragment](const auto& pair) {
                    return pair.second->getName().find(nameFragment) != std::string::npos;
                });
            return result;
        }
        
        std::vector<User> findAll() const override {
            std::lock_guard<std::mutex> lock(mutex);
            std::vector<User> result;
            std::transform(users.begin(), users.end(), std::back_inserter(result),
                [](const auto& pair) { return *pair.second; });
            return result;
        }
        
        User save(const User& user) override {
            std::lock_guard<std::mutex> lock(mutex);
            UserId id = user.getId();
            if (id == 0) {
                id = nextId++;
            }
            
            auto userPtr = std::make_shared<User>(user);
            userPtr->setId(id);
            users[id] = userPtr;
            return *userPtr;
        }
        
        void deleteById(UserId id) override {
            std::lock_guard<std::mutex> lock(mutex);
            users.erase(id);
        }
        
        bool existsById(UserId id) const override {
            std::lock_guard<std::mutex> lock(mutex);
            return users.find(id) != users.end();
        }
        
        size_t count() const override {
            std::lock_guard<std::mutex> lock(mutex);
            return users.size();
        }
        
        std::vector<User> findUsersCreatedAfter(const std::chrono::system_clock::time_point& date) const override {
            std::lock_guard<std::mutex> lock(mutex);
            std::vector<User> result;
            std::copy_if(users.begin(), users.end(), std::back_inserter(result),
                [&date](const auto& pair) {
                    return pair.second->getCreatedAt() > date;
                });
            return result;
        }
    };
}

// Generic cache template
namespace UserService::Utils {
    
    template<typename K, typename V>
    class Cache {
    private:
        struct CacheEntry {
            V value;
            std::chrono::system_clock::time_point expiryTime;
            
            CacheEntry(const V& val, std::chrono::milliseconds ttl)
                : value(val), expiryTime(std::chrono::system_clock::now() + ttl) {}
        };
        
        mutable std::mutex mutex;
        std::map<K, CacheEntry> cache;
        std::chrono::milliseconds ttl;
        size_t maxSize;
        
        void cleanup() {
            auto now = std::chrono::system_clock::now();
            auto it = cache.begin();
            while (it != cache.end()) {
                if (it->second.expiryTime <= now) {
                    it = cache.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
    public:
        explicit Cache(size_t maxSize = DEFAULT_CACHE_SIZE, 
                      std::chrono::milliseconds ttl = std::chrono::milliseconds(CACHE_TTL_MS))
            : ttl(ttl), maxSize(maxSize) {}
        
        std::optional<V> get(const K& key) {
            std::lock_guard<std::mutex> lock(mutex);
            cleanup();
            
            auto it = cache.find(key);
            if (it != cache.end() && it->second.expiryTime > std::chrono::system_clock::now()) {
                return it->second.value;
            }
            return std::nullopt;
        }
        
        void put(const K& key, const V& value) {
            std::lock_guard<std::mutex> lock(mutex);
            cleanup();
            
            if (cache.size() >= maxSize && cache.find(key) == cache.end()) {
                // Remove oldest entry
                cache.erase(cache.begin());
            }
            
            cache.emplace(key, CacheEntry(value, ttl));
        }
        
        void remove(const K& key) {
            std::lock_guard<std::mutex> lock(mutex);
            cache.erase(key);
        }
        
        void clear() {
            std::lock_guard<std::mutex> lock(mutex);
            cache.clear();
        }
        
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex);
            return cache.size();
        }
    };
}

// Service layer
namespace UserService {
    
    class UserService {
    private:
        std::unique_ptr<Models::UserRepository> repository;
        std::unique_ptr<Utils::Cache<UserId, Models::User>> cache;
        std::vector<ValidationFunction> emailValidators;
        std::vector<ValidationFunction> nameValidators;
        
        void initializeValidators() {
            // Email validators
            emailValidators.push_back([](const std::string& email) {
                return email.find('@') != std::string::npos && 
                       email.find('.') != std::string::npos;
            });
            
            emailValidators.push_back([](const std::string& email) {
                return email.length() <= MAX_EMAIL_LENGTH;
            });
            
            // Name validators
            nameValidators.push_back([](const std::string& name) {
                return !name.empty() && name.length() <= MAX_USERNAME_LENGTH;
            });
            
            nameValidators.push_back([](const std::string& name) {
                return std::all_of(name.begin(), name.end(), 
                    [](char c) { return std::isprint(c); });
            });
        }
        
    public:
        UserService(std::unique_ptr<Models::UserRepository> repo)
            : repository(std::move(repo)),
              cache(std::make_unique<Utils::Cache<UserId, Models::User>>()) {
            initializeValidators();
        }
        
        ~UserService() = default;
        
        // Disable copy constructor and assignment
        UserService(const UserService&) = delete;
        UserService& operator=(const UserService&) = delete;
        
        // Enable move constructor and assignment
        UserService(UserService&&) = default;
        UserService& operator=(UserService&&) = default;
        
        Models::User createUser(const std::string& name, const std::string& email, UserRole role = UserRole::USER) {
            validateEmail(email);
            validateName(name);
            
            if (repository->findByEmail(email)) {
                throw ValidationException("Email already exists: " + email);
            }
            
            Models::User user(0, name, email, role);
            Models::User savedUser = repository->save(user);
            cache->put(savedUser.getId(), savedUser);
            
            return savedUser;
        }
        
        Models::User getUserById(UserId id) {
            // Check cache first
            if (auto cachedUser = cache->get(id)) {
                return *cachedUser;
            }
            
            // Fetch from repository
            auto user = repository->findById(id);
            if (!user) {
                throw UserNotFoundException(id);
            }
            
            cache->put(id, *user);
            return *user;
        }
        
        Models::User getUserByEmail(const std::string& email) {
            auto user = repository->findByEmail(email);
            if (!user) {
                throw UserServiceException("User not found with email: " + email, "USER_NOT_FOUND");
            }
            return *user;
        }
        
        Models::User updateUser(UserId id, const std::string& name, const std::string& email) {
            validateName(name);
            validateEmail(email);
            
            auto user = getUserById(id);
            user.setName(name);
            user.setEmail(email);
            
            Models::User updatedUser = repository->save(user);
            cache->put(id, updatedUser);
            
            return updatedUser;
        }
        
        void deleteUser(UserId id) {
            if (!repository->existsById(id)) {
                throw UserNotFoundException(id);
            }
            
            repository->deleteById(id);
            cache->remove(id);
        }
        
        std::vector<Models::User> getAllUsers() {
            return repository->findAll();
        }
        
        std::vector<Models::User> getUsersByRole(UserRole role) {
            return repository->findByRole(role);
        }
        
        std::vector<Models::User> searchUsers(const std::string& nameFragment) {
            return repository->findByNameContaining(nameFragment);
        }
        
        // Async operations
        std::future<Models::User> getUserByIdAsync(UserId id) {
            return std::async(std::launch::async, [this, id]() {
                return getUserById(id);
            });
        }
        
        std::future<std::vector<Models::User>> getAllUsersAsync() {
            return std::async(std::launch::async, [this]() {
                return getAllUsers();
            });
        }
        
    private:
        void validateEmail(const std::string& email) {
            for (const auto& validator : emailValidators) {
                if (!validator(email)) {
                    throw ValidationException("Invalid email: " + email);
                }
            }
        }
        
        void validateName(const std::string& name) {
            for (const auto& validator : nameValidators) {
                if (!validator(name)) {
                    throw ValidationException("Invalid name: " + name);
                }
            }
        }
    };
}

// Utility functions in global namespace
namespace UserService::Utils {
    
    template<typename Container, typename Predicate>
    auto filter(const Container& container, Predicate pred) 
        -> std::vector<typename Container::value_type> {
        std::vector<typename Container::value_type> result;
        std::copy_if(container.begin(), container.end(), 
                    std::back_inserter(result), pred);
        return result;
    }
    
    template<typename Container, typename Comparator>
    auto sort(Container container, Comparator comp) -> Container {
        std::sort(container.begin(), container.end(), comp);
        return container;
    }
    
    template<typename Container, typename KeyExtractor>
    auto groupBy(const Container& container, KeyExtractor keyFunc) 
        -> std::map<decltype(keyFunc(*container.begin())), 
                   std::vector<typename Container::value_type>> {
        std::map<decltype(keyFunc(*container.begin())), 
                std::vector<typename Container::value_type>> result;
        
        for (const auto& item : container) {
            result[keyFunc(item)].push_back(item);
        }
        
        return result;
    }
}

// Main application
int main() {
    try {
        // Initialize repository and service
        auto repository = std::make_unique<UserService::Models::InMemoryUserRepository>();
        UserService::UserService userService(std::move(repository));
        
        std::cout << "User Service Application Started\n" << std::endl;
        
        // Create some users
        auto admin = userService.createUser("Admin User", "admin@example.com", UserService::UserRole::ADMIN);
        auto user1 = userService.createUser("John Doe", "john@example.com");
        auto user2 = userService.createUser("Jane Smith", "jane@example.com");
        
        std::cout << "Created users:" << std::endl;
        std::cout << admin << std::endl;
        std::cout << user1 << std::endl;
        std::cout << user2 << std::endl;
        
        // Test search functionality
        auto allUsers = userService.getAllUsers();
        std::cout << "\nTotal users: " << allUsers.size() << std::endl;
        
        auto adminUsers = userService.getUsersByRole(UserService::UserRole::ADMIN);
        std::cout << "Admin users: " << adminUsers.size() << std::endl;
        
        // Test async operations
        auto futureUser = userService.getUserByIdAsync(user1.getId());
        auto retrievedUser = futureUser.get();
        std::cout << "\nAsync retrieved: " << retrievedUser << std::endl;
        
        // Test utility functions
        auto sortedUsers = UserService::Utils::sort(allUsers, 
            [](const auto& a, const auto& b) { return a.getName() < b.getName(); });
        
        std::cout << "\nSorted users by name:" << std::endl;
        for (const auto& user : sortedUsers) {
            std::cout << "  " << user.getName() << std::endl;
        }
        
        // Group users by role
        auto groupedUsers = UserService::Utils::groupBy(allUsers,
            [](const auto& user) { return user.getRole(); });
        
        std::cout << "\nUsers grouped by role:" << std::endl;
        for (const auto& [role, users] : groupedUsers) {
            std::cout << "  " << UserService::userRoleToString(role) 
                     << ": " << users.size() << " users" << std::endl;
        }
        
    } catch (const UserService::UserServiceException& e) {
        std::cerr << "User service error: " << e.what() 
                  << " (code: " << e.getErrorCode() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nApplication completed successfully." << std::endl;
    return 0;
}