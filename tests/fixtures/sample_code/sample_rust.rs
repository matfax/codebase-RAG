/**
 * Sample Rust code for testing intelligent chunking.
 *
 * This file demonstrates Rust-specific features including:
 * - Structs, enums, and implementations
 * - Traits and generics
 * - Error handling with Result types
 * - Ownership, borrowing, and lifetimes
 * - Pattern matching and iterators
 */

use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::error::Error;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, Instant};

// Constants
const DEFAULT_CAPACITY: usize = 100;
const MAX_RETRY_ATTEMPTS: u32 = 3;
const CACHE_TTL_SECONDS: u64 = 300;

// Type aliases
type UserId = u64;
type UserCache = Arc<Mutex<HashMap<UserId, User>>>;
type AppResult<T> = Result<T, AppError>;

// Enums
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserRole {
    Admin,
    Moderator,
    User,
    Guest,
}

impl Display for UserRole {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            UserRole::Admin => write!(f, "admin"),
            UserRole::Moderator => write!(f, "moderator"),
            UserRole::User => write!(f, "user"),
            UserRole::Guest => write!(f, "guest"),
        }
    }
}

#[derive(Debug)]
pub enum AppError {
    UserNotFound(UserId),
    ValidationError(String),
    DatabaseError(String),
    NetworkError(String),
    AuthorizationError,
    InternalError(Box<dyn Error + Send + Sync>),
}

impl Display for AppError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            AppError::UserNotFound(id) => write!(f, "User with ID {} not found", id),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            AppError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            AppError::AuthorizationError => write!(f, "Authorization error"),
            AppError::InternalError(err) => write!(f, "Internal error: {}", err),
        }
    }
}

impl Error for AppError {}

// Structs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: UserId,
    pub name: String,
    pub email: String,
    pub role: UserRole,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub preferences: UserPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub theme: String,
    pub language: String,
    pub notifications_enabled: bool,
    pub email_frequency: EmailFrequency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailFrequency {
    Never,
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    pub email: String,
    pub role: UserRole,
    pub preferences: Option<UserPreferences>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
    pub role: Option<UserRole>,
    pub preferences: Option<UserPreferences>,
}

// Traits
pub trait UserRepository: Send + Sync {
    async fn get_user(&self, id: UserId) -> AppResult<User>;
    async fn create_user(&self, request: CreateUserRequest) -> AppResult<User>;
    async fn update_user(&self, id: UserId, request: UpdateUserRequest) -> AppResult<User>;
    async fn delete_user(&self, id: UserId) -> AppResult<()>;
    async fn list_users(&self, limit: usize, offset: usize) -> AppResult<Vec<User>>;
    async fn find_users_by_role(&self, role: UserRole) -> AppResult<Vec<User>>;
}

pub trait UserValidator {
    fn validate_email(&self, email: &str) -> bool;
    fn validate_name(&self, name: &str) -> bool;
    fn validate_create_request(&self, request: &CreateUserRequest) -> AppResult<()>;
    fn validate_update_request(&self, request: &UpdateUserRequest) -> AppResult<()>;
}

pub trait Cache<K, V> {
    fn get(&self, key: &K) -> Option<V>;
    fn set(&self, key: K, value: V);
    fn remove(&self, key: &K) -> Option<V>;
    fn clear(&self);
    fn size(&self) -> usize;
}

// Service implementations
pub struct UserService<R: UserRepository> {
    repository: R,
    cache: UserCache,
    validator: Box<dyn UserValidator>,
}

impl<R: UserRepository> UserService<R> {
    pub fn new(repository: R, validator: Box<dyn UserValidator>) -> Self {
        Self {
            repository,
            cache: Arc::new(Mutex::new(HashMap::new())),
            validator,
        }
    }

    pub async fn get_user(&self, id: UserId) -> AppResult<User> {
        // Check cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(user) = cache.get(&id) {
                return Ok(user.clone());
            }
        }

        // Fetch from repository
        let user = self.repository.get_user(id).await?;

        // Update cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(id, user.clone());
        }

        Ok(user)
    }

    pub async fn create_user(&self, request: CreateUserRequest) -> AppResult<User> {
        // Validate request
        self.validator.validate_create_request(&request)?;

        // Create user
        let user = self.repository.create_user(request).await?;

        // Cache the new user
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(user.id, user.clone());
        }

        Ok(user)
    }

    pub async fn update_user(&self, id: UserId, request: UpdateUserRequest) -> AppResult<User> {
        // Validate request
        self.validator.validate_update_request(&request)?;

        // Update user
        let user = self.repository.update_user(id, request).await?;

        // Update cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(id, user.clone());
        }

        Ok(user)
    }

    pub async fn delete_user(&self, id: UserId) -> AppResult<()> {
        // Delete from repository
        self.repository.delete_user(id).await?;

        // Remove from cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.remove(&id);
        }

        Ok(())
    }

    pub async fn list_users_by_role(&self, role: UserRole) -> AppResult<Vec<User>> {
        self.repository.find_users_by_role(role).await
    }

    pub async fn search_users(&self, query: &str) -> AppResult<Vec<User>> {
        let all_users = self.repository.list_users(1000, 0).await?;

        let matching_users = all_users
            .into_iter()
            .filter(|user| {
                user.name.to_lowercase().contains(&query.to_lowercase()) ||
                user.email.to_lowercase().contains(&query.to_lowercase())
            })
            .collect();

        Ok(matching_users)
    }

    pub async fn get_user_statistics(&self) -> AppResult<UserStatistics> {
        let all_users = self.repository.list_users(10000, 0).await?;

        let stats = all_users.iter().fold(UserStatistics::default(), |mut acc, user| {
            acc.total_users += 1;
            match user.role {
                UserRole::Admin => acc.admin_count += 1,
                UserRole::Moderator => acc.moderator_count += 1,
                UserRole::User => acc.user_count += 1,
                UserRole::Guest => acc.guest_count += 1,
            }
            acc
        });

        Ok(stats)
    }
}

// Default validator implementation
pub struct DefaultUserValidator;

impl UserValidator for DefaultUserValidator {
    fn validate_email(&self, email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }

    fn validate_name(&self, name: &str) -> bool {
        !name.trim().is_empty() && name.len() <= 100
    }

    fn validate_create_request(&self, request: &CreateUserRequest) -> AppResult<()> {
        if !self.validate_name(&request.name) {
            return Err(AppError::ValidationError("Invalid name".to_string()));
        }

        if !self.validate_email(&request.email) {
            return Err(AppError::ValidationError("Invalid email".to_string()));
        }

        Ok(())
    }

    fn validate_update_request(&self, request: &UpdateUserRequest) -> AppResult<()> {
        if let Some(ref name) = request.name {
            if !self.validate_name(name) {
                return Err(AppError::ValidationError("Invalid name".to_string()));
            }
        }

        if let Some(ref email) = request.email {
            if !self.validate_email(email) {
                return Err(AppError::ValidationError("Invalid email".to_string()));
            }
        }

        Ok(())
    }
}

// Statistics struct
#[derive(Debug, Default, Serialize)]
pub struct UserStatistics {
    pub total_users: usize,
    pub admin_count: usize,
    pub moderator_count: usize,
    pub user_count: usize,
    pub guest_count: usize,
}

// Generic cache implementation
pub struct MemoryCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    data: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    expires_at: Instant,
}

impl<K, V> MemoryCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(ttl: Duration) -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            ttl,
        }
    }

    pub fn with_capacity(capacity: usize, ttl: Duration) -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::with_capacity(capacity))),
            ttl,
        }
    }

    fn cleanup_expired(&self) {
        if let Ok(mut data) = self.data.lock() {
            let now = Instant::now();
            data.retain(|_, entry| entry.expires_at > now);
        }
    }
}

impl<K, V> Cache<K, V> for MemoryCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    fn get(&self, key: &K) -> Option<V> {
        if let Ok(data) = self.data.lock() {
            if let Some(entry) = data.get(key) {
                if entry.expires_at > Instant::now() {
                    return Some(entry.value.clone());
                }
            }
        }
        None
    }

    fn set(&self, key: K, value: V) {
        if let Ok(mut data) = self.data.lock() {
            let entry = CacheEntry {
                value,
                expires_at: Instant::now() + self.ttl,
            };
            data.insert(key, entry);
        }
    }

    fn remove(&self, key: &K) -> Option<V> {
        if let Ok(mut data) = self.data.lock() {
            data.remove(key).map(|entry| entry.value)
        } else {
            None
        }
    }

    fn clear(&self) {
        if let Ok(mut data) = self.data.lock() {
            data.clear();
        }
    }

    fn size(&self) -> usize {
        if let Ok(data) = self.data.lock() {
            data.len()
        } else {
            0
        }
    }
}

// Utility functions
pub fn create_default_preferences() -> UserPreferences {
    UserPreferences {
        theme: "light".to_string(),
        language: "en".to_string(),
        notifications_enabled: true,
        email_frequency: EmailFrequency::Weekly,
    }
}

pub async fn retry_async_operation<T, F, Fut>(
    operation: F,
    max_attempts: u32,
    initial_delay: Duration,
) -> AppResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = AppResult<T>>,
{
    let mut attempts = 0;
    let mut delay = initial_delay;

    loop {
        attempts += 1;

        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if attempts >= max_attempts {
                    return Err(error);
                }

                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
        }
    }
}

pub fn filter_users_by_predicate<P>(users: Vec<User>, predicate: P) -> Vec<User>
where
    P: Fn(&User) -> bool,
{
    users.into_iter().filter(predicate).collect()
}

pub fn group_users_by_role(users: Vec<User>) -> HashMap<UserRole, Vec<User>> {
    users.into_iter().fold(HashMap::new(), |mut acc, user| {
        acc.entry(user.role.clone()).or_insert_with(Vec::new).push(user);
        acc
    })
}

// Tests module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_role_display() {
        assert_eq!(UserRole::Admin.to_string(), "admin");
        assert_eq!(UserRole::User.to_string(), "user");
    }

    #[test]
    fn test_default_validator() {
        let validator = DefaultUserValidator;

        assert!(validator.validate_email("test@example.com"));
        assert!(!validator.validate_email("invalid-email"));

        assert!(validator.validate_name("John Doe"));
        assert!(!validator.validate_name(""));
    }

    #[tokio::test]
    async fn test_memory_cache() {
        let cache = MemoryCache::new(Duration::from_secs(1));

        cache.set("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(cache.get(&"key1".to_string()), None);
    }
}

// Main function for demonstration
#[tokio::main]
async fn main() -> AppResult<()> {
    // Example usage of the user service
    println!("User Service Demo");

    // This would typically be a real database implementation
    // let repository = DatabaseUserRepository::new("connection_string").await?;
    // let validator = Box::new(DefaultUserValidator);
    // let service = UserService::new(repository, validator);

    println!("Service initialized successfully");

    Ok(())
}
