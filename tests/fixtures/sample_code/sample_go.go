// Sample Go code for testing intelligent chunking
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

// Constants
const (
	DefaultPort     = 8080
	DefaultTimeout  = 30 * time.Second
	MaxRetryCount   = 3
	CacheExpiration = 10 * time.Minute
)

// Global variables
var (
	userCache = make(map[int]*User)
	cacheMu   sync.RWMutex
	logger    = log.New(os.Stdout, "[USER-API] ", log.LstdFlags)
)

// User represents a user in the system
type User struct {
	ID        int       `json:"id"`
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	Role      UserRole  `json:"role"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at,omitempty"`
}

// UserRole represents different user roles
type UserRole string

const (
	RoleAdmin     UserRole = "admin"
	RoleUser      UserRole = "user"
	RoleModerator UserRole = "moderator"
	RoleGuest     UserRole = "guest"
)

// UserRepository defines the interface for user data operations
type UserRepository interface {
	GetUser(ctx context.Context, id int) (*User, error)
	CreateUser(ctx context.Context, user *User) (*User, error)
	UpdateUser(ctx context.Context, id int, updates map[string]interface{}) (*User, error)
	DeleteUser(ctx context.Context, id int) error
	ListUsers(ctx context.Context, limit, offset int) ([]*User, error)
}

// UserService provides business logic for user operations
type UserService struct {
	repo   UserRepository
	cache  UserCache
	logger *log.Logger
}

// NewUserService creates a new user service instance
func NewUserService(repo UserRepository, cache UserCache, logger *log.Logger) *UserService {
	return &UserService{
		repo:   repo,
		cache:  cache,
		logger: logger,
	}
}

// GetUser retrieves a user by ID with caching
func (s *UserService) GetUser(ctx context.Context, id int) (*User, error) {
	// Check cache first
	if user, found := s.cache.Get(id); found {
		s.logger.Printf("Cache hit for user ID: %d", id)
		return user, nil
	}

	// Fetch from repository
	user, err := s.repo.GetUser(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("failed to get user %d: %w", id, err)
	}

	// Cache the result
	s.cache.Set(id, user)
	s.logger.Printf("Fetched user %d from repository", id)

	return user, nil
}

// CreateUser creates a new user
func (s *UserService) CreateUser(ctx context.Context, userData *CreateUserRequest) (*User, error) {
	if err := s.validateUserData(userData); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	user := &User{
		Name:      userData.Name,
		Email:     userData.Email,
		Role:      userData.Role,
		CreatedAt: time.Now(),
	}

	createdUser, err := s.repo.CreateUser(ctx, user)
	if err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	// Cache the new user
	s.cache.Set(createdUser.ID, createdUser)
	s.logger.Printf("Created user with ID: %d", createdUser.ID)

	return createdUser, nil
}

// UpdateUser updates an existing user
func (s *UserService) UpdateUser(ctx context.Context, id int, updates map[string]interface{}) (*User, error) {
	// Validate updates
	if err := s.validateUpdates(updates); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	updatedUser, err := s.repo.UpdateUser(ctx, id, updates)
	if err != nil {
		return nil, fmt.Errorf("failed to update user %d: %w", id, err)
	}

	// Update cache
	s.cache.Set(id, updatedUser)
	s.logger.Printf("Updated user with ID: %d", id)

	return updatedUser, nil
}

// DeleteUser removes a user from the system
func (s *UserService) DeleteUser(ctx context.Context, id int) error {
	if err := s.repo.DeleteUser(ctx, id); err != nil {
		return fmt.Errorf("failed to delete user %d: %w", id, err)
	}

	// Remove from cache
	s.cache.Delete(id)
	s.logger.Printf("Deleted user with ID: %d", id)

	return nil
}

// validateUserData validates user creation data
func (s *UserService) validateUserData(userData *CreateUserRequest) error {
	if userData.Name == "" {
		return fmt.Errorf("name is required")
	}

	if userData.Email == "" {
		return fmt.Errorf("email is required")
	}

	if !isValidEmail(userData.Email) {
		return fmt.Errorf("invalid email format")
	}

	return nil
}

// validateUpdates validates update data
func (s *UserService) validateUpdates(updates map[string]interface{}) error {
	if email, exists := updates["email"]; exists {
		emailStr, ok := email.(string)
		if !ok {
			return fmt.Errorf("email must be a string")
		}
		if !isValidEmail(emailStr) {
			return fmt.Errorf("invalid email format")
		}
	}

	return nil
}

// UserCache interface for caching user data
type UserCache interface {
	Get(id int) (*User, bool)
	Set(id int, user *User)
	Delete(id int)
	Clear()
}

// MemoryCache implements UserCache using in-memory storage
type MemoryCache struct {
	data   map[int]*User
	expiry map[int]time.Time
	mutex  sync.RWMutex
	ttl    time.Duration
}

// NewMemoryCache creates a new memory cache instance
func NewMemoryCache(ttl time.Duration) *MemoryCache {
	cache := &MemoryCache{
		data:   make(map[int]*User),
		expiry: make(map[int]time.Time),
		ttl:    ttl,
	}

	// Start cleanup goroutine
	go cache.cleanup()

	return cache
}

// Get retrieves a user from cache
func (c *MemoryCache) Get(id int) (*User, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if expiry, exists := c.expiry[id]; exists && time.Now().After(expiry) {
		// Entry expired
		delete(c.data, id)
		delete(c.expiry, id)
		return nil, false
	}

	user, exists := c.data[id]
	return user, exists
}

// Set stores a user in cache
func (c *MemoryCache) Set(id int, user *User) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data[id] = user
	c.expiry[id] = time.Now().Add(c.ttl)
}

// Delete removes a user from cache
func (c *MemoryCache) Delete(id int) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	delete(c.data, id)
	delete(c.expiry, id)
}

// Clear removes all entries from cache
func (c *MemoryCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data = make(map[int]*User)
	c.expiry = make(map[int]time.Time)
}

// cleanup removes expired entries
func (c *MemoryCache) cleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.mutex.Lock()
			now := time.Now()
			for id, expiry := range c.expiry {
				if now.After(expiry) {
					delete(c.data, id)
					delete(c.expiry, id)
				}
			}
			c.mutex.Unlock()
		}
	}
}

// HTTP Handler for user operations
type UserHandler struct {
	service *UserService
}

// NewUserHandler creates a new user handler
func NewUserHandler(service *UserService) *UserHandler {
	return &UserHandler{service: service}
}

// GetUserHandler handles GET /users/{id}
func (h *UserHandler) GetUserHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	idStr := vars["id"]

	id, err := strconv.Atoi(idStr)
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}

	user, err := h.service.GetUser(r.Context(), id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

// CreateUserHandler handles POST /users
func (h *UserHandler) CreateUserHandler(w http.ResponseWriter, r *http.Request) {
	var userData CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&userData); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	user, err := h.service.CreateUser(r.Context(), &userData)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

// Request/Response types
type CreateUserRequest struct {
	Name  string   `json:"name"`
	Email string   `json:"email"`
	Role  UserRole `json:"role"`
}

type UpdateUserRequest struct {
	Name  *string   `json:"name,omitempty"`
	Email *string   `json:"email,omitempty"`
	Role  *UserRole `json:"role,omitempty"`
}

type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// Utility functions
func isValidEmail(email string) bool {
	// Simplified email validation
	return len(email) > 0 && 
		   strings.Contains(email, "@") && 
		   strings.Contains(email, ".")
}

func respondWithJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	response := APIResponse{
		Success: statusCode >= 200 && statusCode < 300,
		Data:    data,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}

func respondWithError(w http.ResponseWriter, statusCode int, message string) {
	response := APIResponse{
		Success: false,
		Error:   message,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}

// Main function and server setup
func main() {
	// Initialize dependencies
	cache := NewMemoryCache(CacheExpiration)
	repo := NewInMemoryUserRepository() // Assuming this exists
	service := NewUserService(repo, cache, logger)
	handler := NewUserHandler(service)

	// Setup routes
	router := mux.NewRouter()
	router.HandleFunc("/users/{id:[0-9]+}", handler.GetUserHandler).Methods("GET")
	router.HandleFunc("/users", handler.CreateUserHandler).Methods("POST")

	// Start server
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", DefaultPort),
		Handler:      router,
		ReadTimeout:  DefaultTimeout,
		WriteTimeout: DefaultTimeout,
	}

	logger.Printf("Starting server on port %d", DefaultPort)
	if err := srv.ListenAndServe(); err != nil {
		logger.Fatalf("Server failed to start: %v", err)
	}
}