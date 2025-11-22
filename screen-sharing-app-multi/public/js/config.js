// Configuration file for AirSign application
const CONFIG = {
    // ASL Detection API Configuration
    ASL_API: {
        // Automatically determine URL based on environment
        BASE_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
            ? 'http://localhost:8000'
            : 'https://airsign-api.onrender.com', // Production URL
        ENDPOINT: '/detect-asl',
        TIMEOUT: 15000, // 15 seconds
        FIRST_REQUEST_TIMEOUT: 30000, // 30 seconds for cold start
        RETRY_ATTEMPTS: 3
    },

    // Video Configuration
    VIDEO: {
        CAPTURE_INTERVAL: 3000, // Send frames every 3 seconds for ASL detection
        QUALITY: 0.7, // Optimized quality for balance between size and clarity
        MAX_DIMENSIONS: {
            WIDTH: 640, // Optimal resolution for ASL detection
            HEIGHT: 480
        },
        // Camera optimization settings
        FRAME_RATE: 30, // Smooth camera display
        ENABLE_SMOOTH_RENDERING: true
    },

    // UI Configuration
    UI: {
        ANIMATION_DURATION: 300,
        NOTIFICATION_TIMEOUT: 5000
    }
};

// Function to get full API URL
function getASLApiUrl() {
    return CONFIG.ASL_API.BASE_URL + CONFIG.ASL_API.ENDPOINT;
}

// Function to update configuration at runtime
function updateConfig(key, value) {
    const keys = key.split('.');
    let current = CONFIG;

    for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
    }

    current[keys[keys.length - 1]] = value;

    // Log the update for debugging
    console.log(`Configuration updated: ${key} = ${value}`);
    console.log('Current CONFIG:', CONFIG);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, getASLApiUrl, updateConfig };
}
