/**
 * Sample TypeScript code for testing intelligent chunking.
 * 
 * This file demonstrates TypeScript-specific features including:
 * - Interfaces and type definitions
 * - Generics and type constraints
 * - Enums and union types
 * - Classes with access modifiers
 * - Decorators and metadata
 */

import { Observable } from 'rxjs';
import { map, filter, catchError } from 'rxjs/operators';

// Type definitions and interfaces
interface User {
    readonly id: number;
    name: string;
    email: string;
    role: UserRole;
    createdAt: Date;
    updatedAt?: Date;
    preferences?: UserPreferences;
}

interface UserPreferences {
    theme: 'light' | 'dark';
    language: string;
    notifications: {
        email: boolean;
        push: boolean;
        sms: boolean;
    };
}

interface ApiResponse<T> {
    data: T;
    success: boolean;
    message?: string;
    errors?: string[];
}

interface PaginatedResponse<T> extends ApiResponse<T[]> {
    pagination: {
        page: number;
        limit: number;
        total: number;
        totalPages: number;
    };
}

// Type aliases
type UserId = number;
type UserEmail = string;
type ApiKey = string;

type CreateUserRequest = Omit<User, 'id' | 'createdAt' | 'updatedAt'>;
type UpdateUserRequest = Partial<Pick<User, 'name' | 'email' | 'preferences'>>;

// Union types
type UserRole = 'admin' | 'user' | 'moderator' | 'guest';
type ApiStatus = 'idle' | 'loading' | 'success' | 'error';

// Enums
enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4
}

enum HttpStatus {
    OK = 200,
    CREATED = 201,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    INTERNAL_SERVER_ERROR = 500
}

// Generic interfaces
interface Repository<T, K = number> {
    findById(id: K): Promise<T | null>;
    findAll(options?: QueryOptions): Promise<T[]>;
    create(entity: Omit<T, 'id'>): Promise<T>;
    update(id: K, updates: Partial<T>): Promise<T>;
    delete(id: K): Promise<boolean>;
}

interface QueryOptions {
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
    filters?: Record<string, any>;
}

// Generic constraint example
interface Identifiable {
    id: number;
}

interface Timestamped {
    createdAt: Date;
    updatedAt?: Date;
}

// Utility types
type RequiredUser = Required<User>;
type PartialUser = Partial<User>;
type UserKeys = keyof User;
type UserValues = User[keyof User];

// Conditional types
type NonNullable<T> = T extends null | undefined ? never : T;
type ApiResult<T> = T extends string ? string : T extends number ? number : object;

// Class with generics and decorators
@injectable()
class UserService<T extends Identifiable = User> implements Repository<T> {
    private readonly cache = new Map<number, T>();
    private readonly logger: Logger;

    constructor(
        private readonly apiClient: ApiClient,
        @inject('Logger') logger: Logger
    ) {
        this.logger = logger;
    }

    /**
     * Find a user by ID
     * @param id - The user ID
     * @returns Promise resolving to user or null
     */
    async findById(id: number): Promise<T | null> {
        try {
            this.logger.info(`Fetching user with ID: ${id}`);
            
            // Check cache first
            if (this.cache.has(id)) {
                return this.cache.get(id)!;
            }

            const response = await this.apiClient.get<ApiResponse<T>>(`/users/${id}`);
            
            if (response.success && response.data) {
                this.cache.set(id, response.data);
                return response.data;
            }

            return null;
        } catch (error) {
            this.logger.error(`Error fetching user ${id}:`, error);
            throw new UserServiceError(`Failed to fetch user ${id}`, error);
        }
    }

    /**
     * Find all users with optional filtering
     * @param options - Query options
     * @returns Promise resolving to array of users
     */
    async findAll(options: QueryOptions = {}): Promise<T[]> {
        try {
            const response = await this.apiClient.get<PaginatedResponse<T>>('/users', {
                params: options
            });

            if (response.success) {
                // Update cache with results
                response.data.forEach(user => {
                    this.cache.set((user as any).id, user);
                });

                return response.data;
            }

            throw new Error(response.message || 'Failed to fetch users');
        } catch (error) {
            this.logger.error('Error fetching users:', error);
            throw new UserServiceError('Failed to fetch users', error);
        }
    }

    /**
     * Create a new user
     * @param userData - User data without ID
     * @returns Promise resolving to created user
     */
    async create(userData: Omit<T, 'id'>): Promise<T> {
        try {
            this.validateUserData(userData);

            const response = await this.apiClient.post<ApiResponse<T>>('/users', userData);

            if (response.success && response.data) {
                this.cache.set((response.data as any).id, response.data);
                this.logger.info(`User created with ID: ${(response.data as any).id}`);
                return response.data;
            }

            throw new Error(response.message || 'Failed to create user');
        } catch (error) {
            this.logger.error('Error creating user:', error);
            throw new UserServiceError('Failed to create user', error);
        }
    }

    /**
     * Update an existing user
     * @param id - User ID
     * @param updates - Partial user data
     * @returns Promise resolving to updated user
     */
    async update(id: number, updates: Partial<T>): Promise<T> {
        try {
            const response = await this.apiClient.put<ApiResponse<T>>(`/users/${id}`, updates);

            if (response.success && response.data) {
                this.cache.set(id, response.data);
                this.logger.info(`User ${id} updated successfully`);
                return response.data;
            }

            throw new Error(response.message || 'Failed to update user');
        } catch (error) {
            this.logger.error(`Error updating user ${id}:`, error);
            throw new UserServiceError(`Failed to update user ${id}`, error);
        }
    }

    /**
     * Delete a user
     * @param id - User ID
     * @returns Promise resolving to success status
     */
    async delete(id: number): Promise<boolean> {
        try {
            const response = await this.apiClient.delete<ApiResponse<void>>(`/users/${id}`);

            if (response.success) {
                this.cache.delete(id);
                this.logger.info(`User ${id} deleted successfully`);
                return true;
            }

            throw new Error(response.message || 'Failed to delete user');
        } catch (error) {
            this.logger.error(`Error deleting user ${id}:`, error);
            throw new UserServiceError(`Failed to delete user ${id}`, error);
        }
    }

    /**
     * Search users by criteria
     * @param criteria - Search criteria
     * @returns Promise resolving to matching users
     */
    async search(criteria: UserSearchCriteria): Promise<T[]> {
        const queryOptions: QueryOptions = {
            filters: criteria
        };

        return this.findAll(queryOptions);
    }

    /**
     * Get user statistics
     * @returns Promise resolving to user statistics
     */
    async getStatistics(): Promise<UserStatistics> {
        try {
            const response = await this.apiClient.get<ApiResponse<UserStatistics>>('/users/statistics');
            
            if (response.success && response.data) {
                return response.data;
            }

            throw new Error(response.message || 'Failed to fetch statistics');
        } catch (error) {
            this.logger.error('Error fetching user statistics:', error);
            throw new UserServiceError('Failed to fetch user statistics', error);
        }
    }

    /**
     * Validate user data
     * @private
     * @param userData - User data to validate
     */
    private validateUserData(userData: Partial<T>): void {
        const user = userData as any;
        
        if (!user.name || user.name.trim().length === 0) {
            throw new ValidationError('User name is required');
        }

        if (!user.email || !this.isValidEmail(user.email)) {
            throw new ValidationError('Valid email is required');
        }
    }

    /**
     * Validate email format
     * @private
     * @param email - Email to validate
     * @returns True if email is valid
     */
    private isValidEmail(email: string): boolean {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * Clear cache
     */
    clearCache(): void {
        this.cache.clear();
        this.logger.debug('User cache cleared');
    }

    /**
     * Get cache statistics
     * @returns Cache statistics
     */
    getCacheStats(): CacheStats {
        return {
            size: this.cache.size,
            keys: Array.from(this.cache.keys())
        };
    }
}

// Abstract class example
abstract class BaseEntity implements Identifiable, Timestamped {
    abstract readonly id: number;
    public readonly createdAt: Date = new Date();
    public updatedAt?: Date;

    /**
     * Update the entity's timestamp
     */
    touch(): void {
        this.updatedAt = new Date();
    }

    /**
     * Convert entity to JSON
     */
    abstract toJSON(): Record<string, any>;
}

// Interface implementations
interface UserSearchCriteria {
    name?: string;
    email?: string;
    role?: UserRole;
    createdAfter?: Date;
    createdBefore?: Date;
}

interface UserStatistics {
    total: number;
    byRole: Record<UserRole, number>;
    recentlyCreated: number;
    activeUsers: number;
}

interface CacheStats {
    size: number;
    keys: number[];
}

// Error classes
class UserServiceError extends Error {
    constructor(
        message: string,
        public readonly cause?: Error,
        public readonly code?: string
    ) {
        super(message);
        this.name = 'UserServiceError';
    }
}

class ValidationError extends Error {
    constructor(message: string, public readonly field?: string) {
        super(message);
        this.name = 'ValidationError';
    }
}

// Utility classes and interfaces
interface Logger {
    debug(message: string, ...args: any[]): void;
    info(message: string, ...args: any[]): void;
    warn(message: string, ...args: any[]): void;
    error(message: string, ...args: any[]): void;
}

interface ApiClient {
    get<T>(url: string, config?: any): Promise<T>;
    post<T>(url: string, data?: any, config?: any): Promise<T>;
    put<T>(url: string, data?: any, config?: any): Promise<T>;
    delete<T>(url: string, config?: any): Promise<T>;
}

// Decorator functions (simplified versions)
function injectable<T extends new (...args: any[]) => {}>(constructor: T) {
    return constructor;
}

function inject(token: string) {
    return function (target: any, propertyKey: string | symbol | undefined, parameterIndex: number) {
        // Decorator implementation would go here
    };
}

// Generic utility functions
function createMapper<TSource, TTarget>(
    mapFn: (source: TSource) => TTarget
): (items: TSource[]) => TTarget[] {
    return (items: TSource[]) => items.map(mapFn);
}

function createFilter<T>(
    predicate: (item: T) => boolean
): (items: T[]) => T[] {
    return (items: T[]) => items.filter(predicate);
}

function createSorter<T>(
    compareFn: (a: T, b: T) => number
): (items: T[]) => T[] {
    return (items: T[]) => [...items].sort(compareFn);
}

// Observable patterns
class UserObservableService {
    private userSubject = new Subject<User>();

    /**
     * Get user stream
     * @returns Observable of users
     */
    getUserStream(): Observable<User> {
        return this.userSubject.asObservable();
    }

    /**
     * Get filtered user stream
     * @param role - Role to filter by
     * @returns Observable of filtered users
     */
    getUsersByRole(role: UserRole): Observable<User> {
        return this.getUserStream().pipe(
            filter(user => user.role === role),
            map(user => ({ ...user, filtered: true } as User)),
            catchError(error => {
                console.error('Error in user stream:', error);
                return EMPTY;
            })
        );
    }

    /**
     * Emit a new user
     * @param user - User to emit
     */
    emitUser(user: User): void {
        this.userSubject.next(user);
    }
}

// Module augmentation example
declare global {
    interface Window {
        userService?: UserService;
    }
}

// Namespace example
namespace UserUtilities {
    export function formatUserName(user: User): string {
        return `${user.name} (${user.role})`;
    }

    export function isAdmin(user: User): boolean {
        return user.role === 'admin';
    }

    export function getUserAge(user: User): number {
        const now = new Date();
        const created = new Date(user.createdAt);
        return Math.floor((now.getTime() - created.getTime()) / (1000 * 60 * 60 * 24));
    }
}

// Export statements
export {
    User,
    UserRole,
    UserService,
    UserObservableService,
    UserServiceError,
    ValidationError,
    LogLevel,
    HttpStatus
};

export type {
    ApiResponse,
    PaginatedResponse,
    Repository,
    QueryOptions,
    UserPreferences,
    UserSearchCriteria,
    UserStatistics,
    CreateUserRequest,
    UpdateUserRequest
};

export default UserService;