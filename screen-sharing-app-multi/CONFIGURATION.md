# AirSign Configuration Guide

## ASL API Configuration

The AirSign application now includes a configurable ASL (American Sign Language) detection API system that allows you to easily change endpoints without modifying code.

### How to Update the ASL API Endpoint

#### Method 1: Using the Configuration Modal (Recommended)

1. **Open the meeting room** (`meeting.html`)
2. **Click the gear icon** (⚙️) next to the ASL detection button
3. **Update the configuration fields:**
   - **Base URL**: The domain of your API (e.g., `https://your-api-domain.com`)
   - **Endpoint**: The specific endpoint path (e.g., `/detect-asl`)
   - **Timeout**: Request timeout in milliseconds (default: 10000)
   - **Retry Attempts**: Number of retry attempts (default: 3)
4. **Click "Save Configuration"**

#### Method 2: Direct Code Modification

If you prefer to modify the code directly, edit `public/js/config.js`:

```javascript
const CONFIG = {
    ASL_API: {
        BASE_URL: 'https://your-new-api-domain.com',  // Change this
        ENDPOINT: '/your-new-endpoint',               // Change this
        TIMEOUT: 10000,
        RETRY_ATTEMPTS: 3
    },
    // ... other config
};
```

#### Method 3: Runtime Configuration Update

You can also update the configuration programmatically:

```javascript
// Update specific values
updateConfig('ASL_API.BASE_URL', 'https://new-domain.com');
updateConfig('ASL_API.ENDPOINT', '/new-endpoint');

// Or update the ASL detector directly
if (window.aslDetector) {
    window.aslDetector.updateApiConfig({
        BASE_URL: 'https://new-domain.com',
        ENDPOINT: '/new-endpoint',
        TIMEOUT: 15000,
        RETRY_ATTEMPTS: 5
    });
}
```

### Configuration Persistence

- **Local Storage**: Your configuration is automatically saved to the browser's local storage
- **Session Persistence**: Configuration persists across browser sessions
- **Default Fallback**: If no saved configuration exists, the system uses default values

### Default Configuration

```javascript
ASL_API: {
    BASE_URL: 'https://airsign-api.onrender.com',
    ENDPOINT: '/detect-asl',
    TIMEOUT: 10000,        // 10 seconds
    RETRY_ATTEMPTS: 3
}
```

### File Structure

- `public/js/config.js` - Main configuration file
- `public/js/aslDetector.js` - Updated ASL detector with configurable endpoints
- `public/meeting.html` - Configuration modal interface

### Benefits of the New System

1. **Easy Updates**: Change endpoints without touching code
2. **Environment Flexibility**: Different URLs for dev/staging/production
3. **User-Friendly**: Visual interface for configuration changes
4. **Persistent**: Settings saved across sessions
5. **Fallback Safety**: Default values ensure the app always works

### Troubleshooting

- **Configuration not saving**: Check browser console for errors
- **API calls failing**: Verify the endpoint URL is correct and accessible
- **Modal not opening**: Ensure `config.js` is loaded before `aslDetector.js`

### Example Configuration Changes

#### Change to a different API service:
```
Base URL: https://api.example.com
Endpoint: /v1/asl-detect
```

#### Change to a local development server:
```
Base URL: http://localhost:3000
Endpoint: /api/asl
```

#### Change to a different production server:
```
Base URL: https://my-production-api.com
Endpoint: /detect-sign-language
```
