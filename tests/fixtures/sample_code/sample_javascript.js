/**
 * Sample JavaScript code for testing intelligent chunking.
 * 
 * This file contains various JavaScript constructs including:
 * - ES6 classes and methods
 * - Arrow functions and regular functions
 * - Async/await patterns
 * - Import/export statements
 * - Constants and variables
 */

import { EventEmitter } from 'events';
import axios from 'axios';

// Constants
const API_BASE_URL = 'https://api.example.com';
const MAX_RETRY_ATTEMPTS = 3;
const DEFAULT_TIMEOUT = 5000;

// Module-level variables
let requestCounter = 0;
const cache = new Map();

/**
 * User class representing a system user
 */
class User {
    constructor(id, name, email) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.createdAt = new Date();
    }
    
    /**
     * Get user's display name
     * @returns {string} The display name
     */
    getDisplayName() {
        return `${this.name} (${this.email})`;
    }
    
    /**
     * Update user information
     * @param {Object} updates - Object containing fields to update
     */
    update(updates) {
        Object.assign(this, updates);
        this.updatedAt = new Date();
    }
    
    /**
     * Convert user to JSON representation
     * @returns {Object} JSON object
     */
    toJSON() {
        return {
            id: this.id,
            name: this.name,
            email: this.email,
            createdAt: this.createdAt.toISOString(),
            updatedAt: this.updatedAt?.toISOString()
        };
    }
    
    /**
     * Static method to create user from API data
     * @param {Object} apiData - Data from API
     * @returns {User} New User instance
     */
    static fromApiData(apiData) {
        const user = new User(apiData.id, apiData.name, apiData.email);
        if (apiData.created_at) {
            user.createdAt = new Date(apiData.created_at);
        }
        return user;
    }
}

/**
 * API client for user operations
 */
class UserApiClient extends EventEmitter {
    constructor(baseUrl = API_BASE_URL, timeout = DEFAULT_TIMEOUT) {
        super();
        this.baseUrl = baseUrl;
        this.timeout = timeout;
        this.axios = axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        this._setupInterceptors();
    }
    
    /**
     * Setup request/response interceptors
     * @private
     */
    _setupInterceptors() {
        this.axios.interceptors.request.use(
            config => {
                requestCounter++;
                this.emit('request', { count: requestCounter, config });
                return config;
            },
            error => Promise.reject(error)
        );
        
        this.axios.interceptors.response.use(
            response => {
                this.emit('response', { data: response.data, status: response.status });
                return response;
            },
            error => {
                this.emit('error', error);
                return Promise.reject(error);
            }
        );
    }
    
    /**
     * Fetch a user by ID
     * @param {number} userId - The user ID
     * @returns {Promise<User>} The user object
     */
    async getUser(userId) {
        const cacheKey = `user_${userId}`;
        
        if (cache.has(cacheKey)) {
            return cache.get(cacheKey);
        }
        
        try {
            const response = await this.axios.get(`/users/${userId}`);
            const user = User.fromApiData(response.data);
            cache.set(cacheKey, user);
            return user;
        } catch (error) {
            throw new Error(`Failed to fetch user ${userId}: ${error.message}`);
        }
    }
    
    /**
     * Create a new user
     * @param {Object} userData - User data
     * @returns {Promise<User>} Created user
     */
    async createUser(userData) {
        try {
            const response = await this.axios.post('/users', userData);
            const user = User.fromApiData(response.data);
            this.emit('userCreated', user);
            return user;
        } catch (error) {
            throw new Error(`Failed to create user: ${error.message}`);
        }
    }
    
    /**
     * Update an existing user
     * @param {number} userId - User ID
     * @param {Object} updates - Updates to apply
     * @returns {Promise<User>} Updated user
     */
    async updateUser(userId, updates) {
        try {
            const response = await this.axios.put(`/users/${userId}`, updates);
            const user = User.fromApiData(response.data);
            
            // Update cache
            const cacheKey = `user_${userId}`;
            cache.set(cacheKey, user);
            
            this.emit('userUpdated', user);
            return user;
        } catch (error) {
            throw new Error(`Failed to update user ${userId}: ${error.message}`);
        }
    }
    
    /**
     * Delete a user
     * @param {number} userId - User ID
     * @returns {Promise<boolean>} Success status
     */
    async deleteUser(userId) {
        try {
            await this.axios.delete(`/users/${userId}`);
            
            // Remove from cache
            const cacheKey = `user_${userId}`;
            cache.delete(cacheKey);
            
            this.emit('userDeleted', { userId });
            return true;
        } catch (error) {
            throw new Error(`Failed to delete user ${userId}: ${error.message}`);
        }
    }
    
    /**
     * List users with pagination
     * @param {Object} options - Query options
     * @returns {Promise<Object>} Users list with pagination info
     */
    async listUsers(options = {}) {
        const { page = 1, limit = 10, search = '' } = options;
        
        try {
            const response = await this.axios.get('/users', {
                params: { page, limit, search }
            });
            
            const users = response.data.users.map(userData => User.fromApiData(userData));
            
            return {
                users,
                pagination: response.data.pagination,
                total: response.data.total
            };
        } catch (error) {
            throw new Error(`Failed to list users: ${error.message}`);
        }
    }
}

/**
 * Utility functions
 */

// Arrow function for validation
const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

// Arrow function with async
const fetchUserWithRetry = async (apiClient, userId, maxAttempts = MAX_RETRY_ATTEMPTS) => {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await apiClient.getUser(userId);
        } catch (error) {
            if (attempt === maxAttempts) {
                throw error;
            }
            
            const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
};

/**
 * Regular function for batch operations
 * @param {UserApiClient} apiClient - API client instance
 * @param {Array} userIds - Array of user IDs
 * @returns {Promise<Array>} Array of user objects
 */
async function batchFetchUsers(apiClient, userIds) {
    const promises = userIds.map(id => fetchUserWithRetry(apiClient, id));
    const results = await Promise.allSettled(promises);
    
    return results.map((result, index) => ({
        userId: userIds[index],
        success: result.status === 'fulfilled',
        data: result.status === 'fulfilled' ? result.value : null,
        error: result.status === 'rejected' ? result.reason.message : null
    }));
}

/**
 * Higher-order function example
 * @param {Function} operation - Operation to perform
 * @returns {Function} Debounced version of the operation
 */
function debounce(operation, delay) {
    let timeoutId;
    
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => operation.apply(this, args), delay);
    };
}

/**
 * Event handler setup
 * @param {UserApiClient} apiClient - API client to set up handlers for
 */
function setupEventHandlers(apiClient) {
    apiClient.on('userCreated', (user) => {
        console.log(`User created: ${user.getDisplayName()}`);
    });
    
    apiClient.on('userUpdated', (user) => {
        console.log(`User updated: ${user.getDisplayName()}`);
    });
    
    apiClient.on('userDeleted', ({ userId }) => {
        console.log(`User ${userId} deleted`);
    });
    
    apiClient.on('error', (error) => {
        console.error('API Error:', error.message);
    });
}

// Object with methods
const userUtils = {
    formatUser: (user) => `${user.name} <${user.email}>`,
    
    sortUsers: (users, field = 'name') => {
        return [...users].sort((a, b) => {
            const aVal = a[field];
            const bVal = b[field];
            
            if (typeof aVal === 'string') {
                return aVal.localeCompare(bVal);
            }
            
            return aVal - bVal;
        });
    },
    
    filterUsers: (users, predicate) => users.filter(predicate),
    
    groupUsersByDomain: (users) => {
        return users.reduce((groups, user) => {
            const domain = user.email.split('@')[1];
            if (!groups[domain]) {
                groups[domain] = [];
            }
            groups[domain].push(user);
            return groups;
        }, {});
    }
};

// Export statements
export default UserApiClient;
export { User, validateEmail, fetchUserWithRetry, batchFetchUsers, userUtils, setupEventHandlers };

// Example usage (this would typically be in a separate file)
if (typeof window !== 'undefined') {
    // Browser environment example
    const apiClient = new UserApiClient();
    setupEventHandlers(apiClient);
    
    // Example async operation
    (async () => {
        try {
            const users = await apiClient.listUsers({ limit: 5 });
            console.log('Users:', users);
        } catch (error) {
            console.error('Error fetching users:', error);
        }
    })();
}