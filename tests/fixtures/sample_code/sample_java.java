/**
 * Sample Java code for testing intelligent chunking.
 * 
 * This file demonstrates Java-specific features including:
 * - Classes, interfaces, and inheritance
 * - Generics and collections
 * - Annotations and reflection
 * - Exception handling
 * - Java 8+ features (streams, lambdas, optionals)
 */

package com.example.userservice;

import java.time.LocalDateTime;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.function.Predicate;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Email;
import javax.annotation.Nullable;
import org.springframework.stereotype.Service;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Component;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnore;

// Constants class
public final class Constants {
    public static final int DEFAULT_PAGE_SIZE = 20;
    public static final int MAX_PAGE_SIZE = 100;
    public static final long CACHE_TTL_MINUTES = 30;
    public static final String DEFAULT_ROLE = "USER";
    
    private Constants() {
        // Utility class - prevent instantiation
    }
}

// Enums
public enum UserRole {
    ADMIN("admin", 100),
    MODERATOR("moderator", 50),
    USER("user", 10),
    GUEST("guest", 1);
    
    private final String displayName;
    private final int priority;
    
    UserRole(String displayName, int priority) {
        this.displayName = displayName;
        this.priority = priority;
    }
    
    public String getDisplayName() {
        return displayName;
    }
    
    public int getPriority() {
        return priority;
    }
    
    public static UserRole fromString(String role) {
        return Arrays.stream(values())
                .filter(r -> r.displayName.equalsIgnoreCase(role))
                .findFirst()
                .orElse(USER);
    }
}

public enum NotificationFrequency {
    NEVER,
    DAILY,
    WEEKLY,
    MONTHLY;
    
    public boolean isEnabled() {
        return this != NEVER;
    }
}

// Custom exceptions
public class UserServiceException extends Exception {
    private final String errorCode;
    
    public UserServiceException(String message) {
        super(message);
        this.errorCode = "GENERIC_ERROR";
    }
    
    public UserServiceException(String message, String errorCode) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public UserServiceException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "GENERIC_ERROR";
    }
    
    public String getErrorCode() {
        return errorCode;
    }
}

public class UserNotFoundException extends UserServiceException {
    public UserNotFoundException(Long userId) {
        super("User not found with ID: " + userId, "USER_NOT_FOUND");
    }
}

public class ValidationException extends UserServiceException {
    public ValidationException(String message) {
        super(message, "VALIDATION_ERROR");
    }
}

// Data Transfer Objects
public class UserPreferences {
    @JsonProperty("theme")
    private String theme = "light";
    
    @JsonProperty("language")
    private String language = "en";
    
    @JsonProperty("notifications_enabled")
    private boolean notificationsEnabled = true;
    
    @JsonProperty("email_frequency")
    private NotificationFrequency emailFrequency = NotificationFrequency.WEEKLY;
    
    // Default constructor for JSON deserialization
    public UserPreferences() {}
    
    public UserPreferences(String theme, String language, boolean notificationsEnabled, 
                          NotificationFrequency emailFrequency) {
        this.theme = theme;
        this.language = language;
        this.notificationsEnabled = notificationsEnabled;
        this.emailFrequency = emailFrequency;
    }
    
    // Getters and setters
    public String getTheme() { return theme; }
    public void setTheme(String theme) { this.theme = theme; }
    
    public String getLanguage() { return language; }
    public void setLanguage(String language) { this.language = language; }
    
    public boolean isNotificationsEnabled() { return notificationsEnabled; }
    public void setNotificationsEnabled(boolean notificationsEnabled) { 
        this.notificationsEnabled = notificationsEnabled; 
    }
    
    public NotificationFrequency getEmailFrequency() { return emailFrequency; }
    public void setEmailFrequency(NotificationFrequency emailFrequency) { 
        this.emailFrequency = emailFrequency; 
    }
    
    @Override
    public String toString() {
        return String.format("UserPreferences{theme='%s', language='%s', notificationsEnabled=%s, emailFrequency=%s}",
                theme, language, notificationsEnabled, emailFrequency);
    }
}

// Main entity class
public class User {
    @JsonProperty("id")
    private Long id;
    
    @NotNull
    @JsonProperty("name")
    private String name;
    
    @NotNull
    @Email
    @JsonProperty("email")
    private String email;
    
    @JsonProperty("role")
    private UserRole role = UserRole.USER;
    
    @JsonProperty("created_at")
    private LocalDateTime createdAt;
    
    @JsonProperty("updated_at")
    @Nullable
    private LocalDateTime updatedAt;
    
    @JsonProperty("preferences")
    private UserPreferences preferences;
    
    @JsonIgnore
    private String passwordHash;
    
    // Constructors
    public User() {
        this.createdAt = LocalDateTime.now();
        this.preferences = new UserPreferences();
    }
    
    public User(String name, String email) {
        this();
        this.name = name;
        this.email = email;
    }
    
    public User(String name, String email, UserRole role) {
        this(name, email);
        this.role = role;
    }
    
    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getName() { return name; }
    public void setName(String name) { 
        this.name = name;
        this.updatedAt = LocalDateTime.now();
    }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { 
        this.email = email;
        this.updatedAt = LocalDateTime.now();
    }
    
    public UserRole getRole() { return role; }
    public void setRole(UserRole role) { 
        this.role = role;
        this.updatedAt = LocalDateTime.now();
    }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public UserPreferences getPreferences() { return preferences; }
    public void setPreferences(UserPreferences preferences) { 
        this.preferences = preferences;
        this.updatedAt = LocalDateTime.now();
    }
    
    public String getPasswordHash() { return passwordHash; }
    public void setPasswordHash(String passwordHash) { this.passwordHash = passwordHash; }
    
    // Business methods
    public String getDisplayName() {
        return String.format("%s (%s)", name, email);
    }
    
    public boolean hasAdminPrivileges() {
        return role == UserRole.ADMIN || role == UserRole.MODERATOR;
    }
    
    public boolean canModerate() {
        return role == UserRole.ADMIN || role == UserRole.MODERATOR;
    }
    
    public Duration getAccountAge() {
        return Duration.between(createdAt, LocalDateTime.now());
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return Objects.equals(id, user.id) && Objects.equals(email, user.email);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id, email);
    }
    
    @Override
    public String toString() {
        return String.format("User{id=%d, name='%s', email='%s', role=%s, createdAt=%s}",
                id, name, email, role, createdAt);
    }
}

// Repository interface
public interface UserRepository {
    Optional<User> findById(Long id);
    Optional<User> findByEmail(String email);
    List<User> findByRole(UserRole role);
    List<User> findByNameContaining(String nameFragment);
    List<User> findAll(int page, int size);
    User save(User user);
    void deleteById(Long id);
    boolean existsByEmail(String email);
    long count();
    List<User> findUsersCreatedAfter(LocalDateTime date);
}

// Repository implementation
@Repository
public class InMemoryUserRepository implements UserRepository {
    private final Map<Long, User> users = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);
    
    @Override
    public Optional<User> findById(Long id) {
        return Optional.ofNullable(users.get(id));
    }
    
    @Override
    public Optional<User> findByEmail(String email) {
        return users.values().stream()
                .filter(user -> user.getEmail().equalsIgnoreCase(email))
                .findFirst();
    }
    
    @Override
    public List<User> findByRole(UserRole role) {
        return users.values().stream()
                .filter(user -> user.getRole() == role)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<User> findByNameContaining(String nameFragment) {
        return users.values().stream()
                .filter(user -> user.getName().toLowerCase()
                        .contains(nameFragment.toLowerCase()))
                .collect(Collectors.toList());
    }
    
    @Override
    public List<User> findAll(int page, int size) {
        return users.values().stream()
                .sorted(Comparator.comparing(User::getCreatedAt))
                .skip((long) page * size)
                .limit(size)
                .collect(Collectors.toList());
    }
    
    @Override
    public User save(User user) {
        if (user.getId() == null) {
            user.setId(idGenerator.getAndIncrement());
        }
        users.put(user.getId(), user);
        return user;
    }
    
    @Override
    public void deleteById(Long id) {
        users.remove(id);
    }
    
    @Override
    public boolean existsByEmail(String email) {
        return users.values().stream()
                .anyMatch(user -> user.getEmail().equalsIgnoreCase(email));
    }
    
    @Override
    public long count() {
        return users.size();
    }
    
    @Override
    public List<User> findUsersCreatedAfter(LocalDateTime date) {
        return users.values().stream()
                .filter(user -> user.getCreatedAt().isAfter(date))
                .collect(Collectors.toList());
    }
}

// Service interface
public interface UserService {
    User createUser(CreateUserRequest request) throws UserServiceException;
    User getUserById(Long id) throws UserNotFoundException;
    User getUserByEmail(String email) throws UserNotFoundException;
    User updateUser(Long id, UpdateUserRequest request) throws UserServiceException;
    void deleteUser(Long id) throws UserNotFoundException;
    List<User> getUsers(int page, int size);
    List<User> getUsersByRole(UserRole role);
    List<User> searchUsers(String query);
    UserStatistics getUserStatistics();
    void changeUserRole(Long id, UserRole newRole) throws UserServiceException;
}

// Request/Response DTOs
public class CreateUserRequest {
    @NotNull
    private String name;
    
    @NotNull
    @Email
    private String email;
    
    private UserRole role = UserRole.USER;
    private UserPreferences preferences;
    
    // Constructors
    public CreateUserRequest() {}
    
    public CreateUserRequest(String name, String email) {
        this.name = name;
        this.email = email;
    }
    
    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    
    public UserRole getRole() { return role; }
    public void setRole(UserRole role) { this.role = role; }
    
    public UserPreferences getPreferences() { return preferences; }
    public void setPreferences(UserPreferences preferences) { this.preferences = preferences; }
}

public class UpdateUserRequest {
    private Optional<String> name = Optional.empty();
    private Optional<String> email = Optional.empty();
    private Optional<UserRole> role = Optional.empty();
    private Optional<UserPreferences> preferences = Optional.empty();
    
    // Getters and setters
    public Optional<String> getName() { return name; }
    public void setName(String name) { this.name = Optional.ofNullable(name); }
    
    public Optional<String> getEmail() { return email; }
    public void setEmail(String email) { this.email = Optional.ofNullable(email); }
    
    public Optional<UserRole> getRole() { return role; }
    public void setRole(UserRole role) { this.role = Optional.ofNullable(role); }
    
    public Optional<UserPreferences> getPreferences() { return preferences; }
    public void setPreferences(UserPreferences preferences) { 
        this.preferences = Optional.ofNullable(preferences); 
    }
}

public class UserStatistics {
    private long totalUsers;
    private Map<UserRole, Long> usersByRole;
    private long newUsersToday;
    private long activeUsers;
    
    public UserStatistics() {
        this.usersByRole = new EnumMap<>(UserRole.class);
    }
    
    // Getters and setters
    public long getTotalUsers() { return totalUsers; }
    public void setTotalUsers(long totalUsers) { this.totalUsers = totalUsers; }
    
    public Map<UserRole, Long> getUsersByRole() { return usersByRole; }
    public void setUsersByRole(Map<UserRole, Long> usersByRole) { this.usersByRole = usersByRole; }
    
    public long getNewUsersToday() { return newUsersToday; }
    public void setNewUsersToday(long newUsersToday) { this.newUsersToday = newUsersToday; }
    
    public long getActiveUsers() { return activeUsers; }
    public void setActiveUsers(long activeUsers) { this.activeUsers = activeUsers; }
}

// Service implementation
@Service
public class UserServiceImpl implements UserService {
    private final UserRepository userRepository;
    private final UserValidator userValidator;
    private final Map<Long, User> cache;
    private final ExecutorService executorService;
    
    public UserServiceImpl(UserRepository userRepository, UserValidator userValidator) {
        this.userRepository = userRepository;
        this.userValidator = userValidator;
        this.cache = new ConcurrentHashMap<>();
        this.executorService = Executors.newFixedThreadPool(10);
    }
    
    @Override
    public User createUser(CreateUserRequest request) throws UserServiceException {
        // Validate request
        userValidator.validateCreateRequest(request);
        
        // Check if email already exists
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new ValidationException("Email already exists: " + request.getEmail());
        }
        
        // Create user
        User user = new User(request.getName(), request.getEmail(), request.getRole());
        if (request.getPreferences() != null) {
            user.setPreferences(request.getPreferences());
        }
        
        User savedUser = userRepository.save(user);
        cache.put(savedUser.getId(), savedUser);
        
        return savedUser;
    }
    
    @Override
    public User getUserById(Long id) throws UserNotFoundException {
        // Check cache first
        User cachedUser = cache.get(id);
        if (cachedUser != null) {
            return cachedUser;
        }
        
        // Fetch from repository
        User user = userRepository.findById(id)
                .orElseThrow(() -> new UserNotFoundException(id));
        
        cache.put(id, user);
        return user;
    }
    
    @Override
    public User getUserByEmail(String email) throws UserNotFoundException {
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new UserServiceException("User not found with email: " + email));
    }
    
    @Override
    public User updateUser(Long id, UpdateUserRequest request) throws UserServiceException {
        User existingUser = getUserById(id);
        
        // Apply updates
        request.getName().ifPresent(existingUser::setName);
        request.getEmail().ifPresent(email -> {
            userValidator.validateEmail(email);
            existingUser.setEmail(email);
        });
        request.getRole().ifPresent(existingUser::setRole);
        request.getPreferences().ifPresent(existingUser::setPreferences);
        
        User updatedUser = userRepository.save(existingUser);
        cache.put(id, updatedUser);
        
        return updatedUser;
    }
    
    @Override
    public void deleteUser(Long id) throws UserNotFoundException {
        if (!userRepository.findById(id).isPresent()) {
            throw new UserNotFoundException(id);
        }
        
        userRepository.deleteById(id);
        cache.remove(id);
    }
    
    @Override
    public List<User> getUsers(int page, int size) {
        size = Math.min(size, Constants.MAX_PAGE_SIZE);
        return userRepository.findAll(page, size);
    }
    
    @Override
    public List<User> getUsersByRole(UserRole role) {
        return userRepository.findByRole(role);
    }
    
    @Override
    public List<User> searchUsers(String query) {
        return userRepository.findByNameContaining(query);
    }
    
    @Override
    public UserStatistics getUserStatistics() {
        UserStatistics stats = new UserStatistics();
        stats.setTotalUsers(userRepository.count());
        
        // Count users by role
        Map<UserRole, Long> roleCount = Arrays.stream(UserRole.values())
                .collect(Collectors.toMap(
                        role -> role,
                        role -> (long) userRepository.findByRole(role).size()
                ));
        stats.setUsersByRole(roleCount);
        
        // Count new users today
        LocalDateTime startOfDay = LocalDateTime.now().toLocalDate().atStartOfDay();
        stats.setNewUsersToday(userRepository.findUsersCreatedAfter(startOfDay).size());
        
        return stats;
    }
    
    @Override
    public void changeUserRole(Long id, UserRole newRole) throws UserServiceException {
        User user = getUserById(id);
        UserRole oldRole = user.getRole();
        
        if (oldRole == newRole) {
            return; // No change needed
        }
        
        user.setRole(newRole);
        userRepository.save(user);
        cache.put(id, user);
        
        // Log role change (in a real app, this might be an audit log)
        System.out.printf("User %d role changed from %s to %s%n", 
                id, oldRole.getDisplayName(), newRole.getDisplayName());
    }
    
    // Async method example
    public CompletableFuture<List<User>> getUsersAsync(int page, int size) {
        return CompletableFuture.supplyAsync(() -> getUsers(page, size), executorService);
    }
    
    // Batch operation example
    public List<User> createUsersInBatch(List<CreateUserRequest> requests) {
        return requests.parallelStream()
                .map(request -> {
                    try {
                        return createUser(request);
                    } catch (UserServiceException e) {
                        System.err.println("Failed to create user: " + e.getMessage());
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
}

// Validator component
@Component
public class UserValidator {
    public void validateCreateRequest(CreateUserRequest request) throws ValidationException {
        if (request.getName() == null || request.getName().trim().isEmpty()) {
            throw new ValidationException("Name is required");
        }
        
        if (request.getEmail() == null || request.getEmail().trim().isEmpty()) {
            throw new ValidationException("Email is required");
        }
        
        validateEmail(request.getEmail());
        validateName(request.getName());
    }
    
    public void validateEmail(String email) throws ValidationException {
        if (!email.contains("@") || !email.contains(".")) {
            throw new ValidationException("Invalid email format");
        }
        
        if (email.length() > 255) {
            throw new ValidationException("Email too long");
        }
    }
    
    public void validateName(String name) throws ValidationException {
        if (name.length() > 100) {
            throw new ValidationException("Name too long");
        }
        
        if (name.trim().length() < 2) {
            throw new ValidationException("Name too short");
        }
    }
}

// Utility class with static methods
public final class UserUtils {
    private UserUtils() {} // Prevent instantiation
    
    public static List<User> filterUsersByPredicate(List<User> users, Predicate<User> predicate) {
        return users.stream()
                .filter(predicate)
                .collect(Collectors.toList());
    }
    
    public static Map<UserRole, List<User>> groupUsersByRole(List<User> users) {
        return users.stream()
                .collect(Collectors.groupingBy(User::getRole));
    }
    
    public static List<User> sortUsersByName(List<User> users) {
        return users.stream()
                .sorted(Comparator.comparing(User::getName))
                .collect(Collectors.toList());
    }
    
    public static List<User> sortUsersByCreationDate(List<User> users, boolean ascending) {
        Comparator<User> comparator = Comparator.comparing(User::getCreatedAt);
        if (!ascending) {
            comparator = comparator.reversed();
        }
        
        return users.stream()
                .sorted(comparator)
                .collect(Collectors.toList());
    }
    
    public static double calculateAverageAccountAge(List<User> users) {
        return users.stream()
                .mapToLong(user -> user.getAccountAge().toDays())
                .average()
                .orElse(0.0);
    }
    
    public static User findOldestUser(List<User> users) {
        return users.stream()
                .min(Comparator.comparing(User::getCreatedAt))
                .orElse(null);
    }
    
    public static User findNewestUser(List<User> users) {
        return users.stream()
                .max(Comparator.comparing(User::getCreatedAt))
                .orElse(null);
    }
}

// Main application class
public class UserServiceApplication {
    public static void main(String[] args) {
        // Initialize dependencies
        UserRepository repository = new InMemoryUserRepository();
        UserValidator validator = new UserValidator();
        UserService userService = new UserServiceImpl(repository, validator);
        
        System.out.println("User Service Application started");
        
        try {
            // Create some sample users
            CreateUserRequest adminRequest = new CreateUserRequest("Admin User", "admin@example.com");
            adminRequest.setRole(UserRole.ADMIN);
            
            CreateUserRequest userRequest = new CreateUserRequest("Regular User", "user@example.com");
            
            User admin = userService.createUser(adminRequest);
            User user = userService.createUser(userRequest);
            
            System.out.println("Created users:");
            System.out.println("Admin: " + admin);
            System.out.println("User: " + user);
            
            // Get statistics
            UserStatistics stats = userService.getUserStatistics();
            System.out.println("Total users: " + stats.getTotalUsers());
            
        } catch (UserServiceException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}