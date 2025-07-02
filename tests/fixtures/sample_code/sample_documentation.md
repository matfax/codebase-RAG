# User Service API Documentation

A comprehensive user management service with authentication, authorization, and user profile management capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Examples](#examples)
8. [Configuration](#configuration)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

## Overview

The User Service API provides a robust foundation for user management in modern applications. It includes:

- **User Registration and Authentication**: Support for email/password and OAuth2 providers
- **Role-Based Access Control (RBAC)**: Flexible permission system with predefined roles
- **Profile Management**: Comprehensive user profile with preferences and settings
- **Security Features**: Two-factor authentication, account lockout, audit logging
- **Scalability**: Redis caching, database read replicas, rate limiting

### Key Features

- ✅ RESTful API with JSON responses
- ✅ JWT-based authentication with refresh tokens
- ✅ OAuth2 integration (Google, GitHub, Microsoft)
- ✅ Role-based permissions system
- ✅ Email verification and password reset
- ✅ Two-factor authentication (TOTP, SMS, Email)
- ✅ Comprehensive audit logging
- ✅ Rate limiting and DDoS protection
- ✅ Health checks and monitoring
- ✅ OpenAPI/Swagger documentation

## Quick Start

### Prerequisites

- Node.js 18+ or Python 3.9+ or Go 1.19+
- PostgreSQL 13+
- Redis 6+
- SMTP server for email notifications

### Installation

```bash
# Clone the repository
git clone https://github.com/example/userservice.git
cd userservice

# Install dependencies
npm install
# or
pip install -r requirements.txt
# or
go mod download

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
npm run migrate
# or
python manage.py migrate
# or
./migrate

# Start the server
npm start
# or
python app.py
# or
go run main.go
```

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=userservice
DB_USERNAME=user
DB_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# JWT
JWT_SECRET=your-super-secret-key

# Email
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=noreply@example.com
SMTP_PASSWORD=your-email-password

# OAuth2 (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. All protected endpoints require a valid JWT token in the `Authorization` header.

### Authentication Flow

1. **Register** or **Login** to obtain access and refresh tokens
2. **Include** the access token in API requests
3. **Refresh** the access token when it expires
4. **Logout** to invalidate tokens

### Token Format

```http
Authorization: Bearer <access_token>
```

### Token Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Register a new user |
| `/auth/login` | POST | Authenticate user |
| `/auth/refresh` | POST | Refresh access token |
| `/auth/logout` | POST | Logout user |
| `/auth/forgot-password` | POST | Request password reset |
| `/auth/reset-password` | POST | Reset password |

## API Endpoints

### Base URL

```
https://api.example.com/v2
```

### User Management

#### Get User Profile

```http
GET /users/me
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "role": "user",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:45:00Z",
  "preferences": {
    "theme": "light",
    "language": "en",
    "notifications_enabled": true,
    "email_frequency": "weekly"
  }
}
```

#### Update User Profile

```http
PUT /users/me
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "John Smith",
  "preferences": {
    "theme": "dark",
    "language": "es"
  }
}
```

#### Get All Users (Admin Only)

```http
GET /users?page=1&limit=20&search=john&role=user
Authorization: Bearer <admin_token>
```

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `limit` (integer): Items per page (default: 20, max: 100)
- `search` (string): Search by name or email
- `role` (string): Filter by user role
- `sort` (string): Sort field (name, email, created_at)
- `order` (string): Sort order (asc, desc)

#### Create User (Admin Only)

```http
POST /users
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "role": "user",
  "send_welcome_email": true
}
```

#### Update User (Admin Only)

```http
PUT /users/{id}
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "name": "Jane Smith",
  "role": "moderator",
  "status": "active"
}
```

#### Delete User (Admin Only)

```http
DELETE /users/{id}
Authorization: Bearer <admin_token>
```

### Role and Permission Management

#### Get User Roles

```http
GET /roles
Authorization: Bearer <token>
```

#### Assign Role to User (Admin Only)

```http
POST /users/{id}/roles
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "role": "moderator"
}
```

#### Check User Permissions

```http
GET /users/me/permissions
Authorization: Bearer <token>
```

### Two-Factor Authentication

#### Enable 2FA

```http
POST /auth/2fa/enable
Authorization: Bearer <token>
Content-Type: application/json

{
  "method": "totp"
}
```

#### Verify 2FA Setup

```http
POST /auth/2fa/verify-setup
Authorization: Bearer <token>
Content-Type: application/json

{
  "code": "123456"
}
```

#### Disable 2FA

```http
POST /auth/2fa/disable
Authorization: Bearer <token>
Content-Type: application/json

{
  "password": "current_password",
  "code": "123456"
}
```

### Account Management

#### Change Password

```http
POST /auth/change-password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "old_password",
  "new_password": "new_strong_password"
}
```

#### Verify Email

```http
POST /auth/verify-email
Content-Type: application/json

{
  "token": "email_verification_token"
}
```

#### Resend Verification Email

```http
POST /auth/resend-verification
Authorization: Bearer <token>
```

### Admin Endpoints

#### Get System Statistics

```http
GET /admin/stats
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "total_users": 1500,
  "active_users": 1200,
  "new_registrations_today": 25,
  "users_by_role": {
    "admin": 5,
    "moderator": 20,
    "user": 1470,
    "guest": 5
  },
  "login_activity": {
    "last_24h": 450,
    "last_7d": 2100
  }
}
```

#### Get Audit Logs

```http
GET /admin/audit?page=1&limit=50&event=user_login&user_id=123
Authorization: Bearer <admin_token>
```

#### Export User Data

```http
GET /admin/export/users?format=csv&date_from=2024-01-01&date_to=2024-01-31
Authorization: Bearer <admin_token>
```

## Data Models

### User

```json
{
  "id": "integer",
  "name": "string",
  "email": "string",
  "role": "enum [admin, moderator, user, guest]",
  "status": "enum [active, inactive, suspended]",
  "email_verified": "boolean",
  "two_factor_enabled": "boolean",
  "last_login": "datetime",
  "created_at": "datetime",
  "updated_at": "datetime",
  "preferences": "UserPreferences"
}
```

### UserPreferences

```json
{
  "theme": "enum [light, dark, auto]",
  "language": "string",
  "timezone": "string",
  "notifications_enabled": "boolean",
  "email_frequency": "enum [never, daily, weekly, monthly]",
  "marketing_emails": "boolean",
  "two_factor_method": "enum [totp, sms, email]"
}
```

### AuthToken

```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": "integer",
  "scope": "string"
}
```

### Role

```json
{
  "name": "string",
  "display_name": "string",
  "description": "string",
  "permissions": ["string"],
  "is_system_role": "boolean",
  "created_at": "datetime"
}
```

### AuditLog

```json
{
  "id": "integer",
  "user_id": "integer",
  "event": "string",
  "resource": "string",
  "resource_id": "string",
  "details": "object",
  "ip_address": "string",
  "user_agent": "string",
  "timestamp": "datetime"
}
```

## Error Handling

The API uses conventional HTTP status codes and returns detailed error information in JSON format.

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - Request successful |
| 201 | Created - Resource created successfully |
| 400 | Bad Request - Invalid request data |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request data is invalid",
    "details": {
      "email": ["Email is required"],
      "password": ["Password must be at least 8 characters"]
    },
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

- `INVALID_CREDENTIALS` - Login credentials are incorrect
- `EMAIL_ALREADY_EXISTS` - Email address is already registered
- `ACCOUNT_LOCKED` - Account is temporarily locked
- `TOKEN_EXPIRED` - JWT token has expired
- `INSUFFICIENT_PERMISSIONS` - User lacks required permissions
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `VALIDATION_ERROR` - Request validation failed
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist

## Examples

### Complete Registration Flow

```javascript
// 1. Register new user
const registerResponse = await fetch('/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'John Doe',
    email: 'john@example.com',
    password: 'StrongPassword123!'
  })
});

const { user, tokens } = await registerResponse.json();

// 2. Verify email (user clicks link in email)
await fetch('/auth/verify-email', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    token: 'email_verification_token_from_email'
  })
});

// 3. Login after verification
const loginResponse = await fetch('/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'john@example.com',
    password: 'StrongPassword123!'
  })
});

const { access_token, refresh_token } = await loginResponse.json();

// 4. Access protected resources
const profileResponse = await fetch('/users/me', {
  headers: { 'Authorization': `Bearer ${access_token}` }
});

const profile = await profileResponse.json();
```

### Password Reset Flow

```javascript
// 1. Request password reset
await fetch('/auth/forgot-password', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'john@example.com'
  })
});

// 2. Reset password (user clicks link in email)
await fetch('/auth/reset-password', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    token: 'password_reset_token_from_email',
    new_password: 'NewStrongPassword123!'
  })
});
```

### Two-Factor Authentication Setup

```javascript
// 1. Enable 2FA
const enable2faResponse = await fetch('/auth/2fa/enable', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    method: 'totp'
  })
});

const { qr_code, secret, backup_codes } = await enable2faResponse.json();

// 2. User scans QR code with authenticator app

// 3. Verify setup with code from app
await fetch('/auth/2fa/verify-setup', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    code: '123456'
  })
});

// 4. Login with 2FA
const loginWith2faResponse = await fetch('/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'john@example.com',
    password: 'password',
    two_factor_code: '654321'
  })
});
```

### Admin User Management

```python
import requests

# Admin authentication
admin_token = "admin_jwt_token"
headers = {
    "Authorization": f"Bearer {admin_token}",
    "Content-Type": "application/json"
}

# Get all users with pagination
response = requests.get(
    "/users",
    params={"page": 1, "limit": 50, "role": "user"},
    headers=headers
)
users = response.json()

# Create new user
new_user_data = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "role": "moderator",
    "send_welcome_email": True
}
response = requests.post("/users", json=new_user_data, headers=headers)
created_user = response.json()

# Update user role
update_data = {"role": "admin"}
response = requests.put(
    f"/users/{created_user['id']}",
    json=update_data,
    headers=headers
)

# Get system statistics
stats_response = requests.get("/admin/stats", headers=headers)
statistics = stats_response.json()
print(f"Total users: {statistics['total_users']}")
```

## Configuration

### Environment-based Configuration

The service supports multiple environments with specific configuration overrides:

#### Development

```yaml
server:
  port: 3000
  debug: true
logging:
  level: debug
features:
  mock_external_services: true
```

#### Staging

```yaml
server:
  port: 8080
monitoring:
  enabled: true
features:
  create_sample_users: true
```

#### Production

```yaml
server:
  workers: 4
logging:
  level: warn
security:
  audit:
    enabled: true
```

### Feature Flags

Enable or disable features using configuration:

```yaml
features:
  user_registration:
    enabled: true
    require_email_verification: true

  two_factor_auth:
    enabled: true
    methods: ["totp", "sms"]

  oauth2:
    enabled: true
    providers: ["google", "github"]
```

### Rate Limiting

Configure rate limits for different endpoints:

```yaml
rate_limiting:
  login: 10_per_minute
  registration: 5_per_hour
  password_reset: 3_per_hour
  api_general: 100_per_minute
```

## Deployment

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 8080
CMD ["npm", "start"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  userservice:
    build: .
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=production
      - DB_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: userservice
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: userservice
spec:
  replicas: 3
  selector:
    matchLabels:
      app: userservice
  template:
    metadata:
      labels:
        app: userservice
    spec:
      containers:
      - name: userservice
        image: userservice:latest
        ports:
        - containerPort: 8080
        env:
        - name: DB_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: userservice
spec:
  selector:
    app: userservice
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `npm test`
6. Run linting: `npm run lint`
7. Commit your changes: `git commit -am 'Add feature'`
8. Push to the branch: `git push origin feature-name`
9. Submit a pull request

### Code Style

We use automated formatting and linting:

```bash
# Format code
npm run format

# Lint code
npm run lint

# Type checking (TypeScript)
npm run type-check
```

### Testing

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run integration tests
npm run test:integration

# Run performance tests
npm run test:performance
```

### Documentation

- Update API documentation for any endpoint changes
- Add JSDoc comments for new functions
- Update this README for significant features
- Include examples for new functionality

### Security

Please report security vulnerabilities to security@example.com rather than creating public issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://docs.example.com/userservice](https://docs.example.com/userservice)
- **API Reference**: [https://api.example.com/docs](https://api.example.com/docs)
- **Issue Tracker**: [https://github.com/example/userservice/issues](https://github.com/example/userservice/issues)
- **Community**: [https://community.example.com](https://community.example.com)
- **Email**: support@example.com
