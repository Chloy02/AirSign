class ASLDetector {
    constructor(config = null) {
        console.log('🔧 ASLDetector: Initializing ASL detector...');
        
        // Use provided config or default from CONFIG object
        this.config = config || (typeof CONFIG !== 'undefined' ? CONFIG.ASL_API : {
            BASE_URL: 'https://airsign-api.onrender.com',
            ENDPOINT: '/detect-asl',
            TIMEOUT: 15000,
            FIRST_REQUEST_TIMEOUT: 30000,
            RETRY_ATTEMPTS: 3
        });
        
        this.apiUrl = this.config.BASE_URL + this.config.ENDPOINT;
        this.baseUrl = this.config.BASE_URL;
        this.isDetecting = false;
        this.detectionInterval = null;
        this.onDetectionResult = null; // Callback for detection results
        this.onErrorCallback = null; // Callback for errors
        this.frameCount = 0;
        this.lastCaptureTime = 0;
        this.processingFrame = false;
        this.isApiWarmedUp = false; // Track if API has been warmed up
        
        console.log('✅ ASLDetector: Initialized with config:', {
            apiUrl: this.apiUrl,
            timeout: this.config.TIMEOUT,
            firstRequestTimeout: this.config.FIRST_REQUEST_TIMEOUT,
            retryAttempts: this.config.RETRY_ATTEMPTS
        });
    }

    // Wake up the API (important for Render free tier)
    async wakeUpAPI() {
        if (this.isApiWarmedUp) {
            console.log('🌐 ASLDetector: API already warmed up');
            return true;
        }

        console.log('🌐 ASLDetector: Waking up API...');
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout for wake up

            const response = await fetch(this.baseUrl + '/', {
                method: 'GET',
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                console.log('✅ ASLDetector: API is now awake');
                this.isApiWarmedUp = true;
                return true;
            } else {
                console.warn('⚠️ ASLDetector: API wake up returned status:', response.status);
                return false;
            }
        } catch (error) {
            console.error('❌ ASLDetector: Failed to wake up API:', error);
            return false;
        }
    }

    // Start real-time ASL detection
    startDetection(videoElement, intervalMs = null) {
        if (this.isDetecting) {
            console.log('⚠️ ASLDetector: Detection already running, ignoring start request');
            return;
        }
        
        // Use provided interval or default from config (3 seconds)
        const interval = intervalMs || (typeof CONFIG !== 'undefined' ? CONFIG.VIDEO.CAPTURE_INTERVAL : 3000);
        
        console.log('🚀 ASLDetector: Starting detection with interval:', interval + 'ms');
        console.log('📹 ASLDetector: Video element details:', {
            width: videoElement.videoWidth,
            height: videoElement.videoHeight,
            readyState: videoElement.readyState,
            currentTime: videoElement.currentTime
        });
        
        this.isDetecting = true;
        this.frameCount = 0;
        this.lastCaptureTime = Date.now();
        
        this.detectionInterval = setInterval(() => {
            this.captureAndDetect(videoElement);
        }, interval);
        
        console.log('✅ ASLDetector: Detection started successfully');
    }

    // Stop detection
    stopDetection() {
        console.log('🛑 ASLDetector: Stopping detection...');
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
            console.log('✅ ASLDetector: Detection interval cleared');
        }
        
        this.isDetecting = false;
        this.processingFrame = false;
        
        console.log('📊 ASLDetector: Final stats - Total frames processed:', this.frameCount);
        console.log('✅ ASLDetector: Detection stopped successfully');
    }

    // Capture frame from video and send for detection
    async captureAndDetect(videoElement) {
        // Skip if already processing a frame to prevent backlog
        if (this.processingFrame) {
            console.log('⏭️ ASLDetector: Skipping frame capture - already processing');
            return;
        }
        
        const currentTime = Date.now();
        const timeSinceLastCapture = currentTime - this.lastCaptureTime;
        
        console.log(`📸 ASLDetector: Frame ${++this.frameCount} - Starting capture (${timeSinceLastCapture}ms since last)`);
        
        try {
            this.processingFrame = true;
            
            // Check if video is ready and has dimensions
            if (!videoElement.videoWidth || !videoElement.videoHeight) {
                console.warn('⚠️ ASLDetector: Video not ready yet, dimensions:', {
                    width: videoElement.videoWidth,
                    height: videoElement.videoHeight,
                    readyState: videoElement.readyState
                });
                return;
            }
            
            console.log('📹 ASLDetector: Video ready, capturing frame from video:', {
                sourceWidth: videoElement.videoWidth,
                sourceHeight: videoElement.videoHeight,
                currentTime: videoElement.currentTime.toFixed(2)
            });
            
            // Create canvas to capture video frame (with optional downscale)
            const sourceWidth = videoElement.videoWidth;
            const sourceHeight = videoElement.videoHeight;
            const maxW = (typeof CONFIG !== 'undefined' && CONFIG.VIDEO?.MAX_DIMENSIONS?.WIDTH) ? CONFIG.VIDEO.MAX_DIMENSIONS.WIDTH : 640;
            const maxH = (typeof CONFIG !== 'undefined' && CONFIG.VIDEO?.MAX_DIMENSIONS?.HEIGHT) ? CONFIG.VIDEO.MAX_DIMENSIONS.HEIGHT : 480;
            
            let targetW = sourceWidth;
            let targetH = sourceHeight;
            const scale = Math.min(maxW / sourceWidth, maxH / sourceHeight, 1);
            if (scale < 1) {
                targetW = Math.round(sourceWidth * scale);
                targetH = Math.round(sourceHeight * scale);
                console.log('🔄 ASLDetector: Scaling frame:', {
                    from: `${sourceWidth}x${sourceHeight}`,
                    to: `${targetW}x${targetH}`,
                    scale: scale.toFixed(3)
                });
            }
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = targetW;
            canvas.height = targetH;
            
            // Optimize canvas rendering
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            const drawStart = performance.now();
            ctx.drawImage(videoElement, 0, 0, targetW, targetH);
            const drawTime = performance.now() - drawStart;
            
            console.log('🎨 ASLDetector: Canvas draw completed in', drawTime.toFixed(2) + 'ms');
            
            // Convert canvas to blob with configurable quality
            const quality = typeof CONFIG !== 'undefined' ? CONFIG.VIDEO.QUALITY : 0.7;
            const blobStart = performance.now();
            
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/jpeg', quality);
            });
            
            const blobTime = performance.now() - blobStart;
            console.log('📦 ASLDetector: Image blob created:', {
                size: (blob.size / 1024).toFixed(2) + 'KB',
                quality: quality,
                time: blobTime.toFixed(2) + 'ms'
            });
            
            // Send to AI API
            this.lastCaptureTime = currentTime;
            await this.detectASL(blob);
            
        } catch (error) {
            console.error('❌ ASLDetector: Error capturing video frame:', error);
            if (this.onErrorCallback) this.onErrorCallback(error);
        } finally {
            this.processingFrame = false;
            console.log('✅ ASLDetector: Frame processing completed');
        }
    }

    // Send image to AI API for ASL detection
    async detectASL(imageBlob) {
        const attempts = Math.max(1, this.config.RETRY_ATTEMPTS || 1);
        // Use longer timeout for first request if API hasn't been warmed up
        const timeoutMs = this.isApiWarmedUp ? 
            (this.config.TIMEOUT || 15000) : 
            (this.config.FIRST_REQUEST_TIMEOUT || 30000);
        let lastError = null;
        
        console.log('🌐 ASLDetector: Starting API request to:', this.apiUrl);
        console.log('⚙️ ASLDetector: Request settings:', {
            attempts: attempts,
            timeout: timeoutMs + 'ms',
            imageSize: (imageBlob.size / 1024).toFixed(2) + 'KB',
            apiWarmedUp: this.isApiWarmedUp
        });
        
        for (let attempt = 1; attempt <= attempts; attempt++) {
            console.log(`🔄 ASLDetector: Attempt ${attempt}/${attempts}`);
            const attemptStart = performance.now();
            
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => {
                    console.log('⏰ ASLDetector: Request timeout reached, aborting...');
                    controller.abort();
                }, timeoutMs);
                
                const formData = new FormData();
                // Attach using both common field names to maximize compatibility
                formData.append('image', imageBlob, 'frame.jpg');
                formData.append('file', imageBlob, 'frame.jpg');
                
                console.log('📤 ASLDetector: Sending request...');
                const requestStart = performance.now();
                
                const response = await fetch(this.apiUrl, {
                    method: 'POST',
                    body: formData,
                    headers: { 'Accept': 'application/json' },
                    signal: controller.signal
                });
                
                const requestTime = performance.now() - requestStart;
                clearTimeout(timeoutId);
                
                console.log('📥 ASLDetector: Response received:', {
                    status: response.status,
                    statusText: response.statusText,
                    time: requestTime.toFixed(2) + 'ms'
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                let result;
                try {
                    const parseStart = performance.now();
                    result = await response.json();
                    const parseTime = performance.now() - parseStart;
                    
                    console.log('📊 ASLDetector: JSON parsed in', parseTime.toFixed(2) + 'ms');
                    console.log('🎯 ASLDetector: Detection result:', result);
                } catch (e) {
                    throw new Error('Invalid JSON response from API');
                }
                
                const totalTime = performance.now() - attemptStart;
                console.log('✅ ASLDetector: API request successful in', totalTime.toFixed(2) + 'ms');
                
                // Mark API as warmed up after first successful request
                if (!this.isApiWarmedUp) {
                    this.isApiWarmedUp = true;
                    console.log('🌐 ASLDetector: API marked as warmed up');
                }
                
                if (this.onDetectionResult) {
                    this.onDetectionResult(result);
                }
                return; // success
                
            } catch (error) {
                lastError = error;
                const attemptTime = performance.now() - attemptStart;
                
                console.warn(`❌ ASLDetector: Attempt ${attempt} failed after ${attemptTime.toFixed(2)}ms:`, {
                    error: error.message || error,
                    type: error.constructor.name
                });
                
                if (attempt < attempts) {
                    console.log('⏳ ASLDetector: Waiting 400ms before retry...');
                    await new Promise(r => setTimeout(r, 400));
                }
            }
        }
        
        console.error('💥 ASLDetector: All attempts failed. Final error:', lastError);
        if (this.onErrorCallback) this.onErrorCallback(lastError);
    }

    // Set callback for detection results
    onResult(callback) {
        this.onDetectionResult = callback;
    }

    // Set callback for errors
    onError(callback) {
        this.onErrorCallback = callback;
    }

    // Get detection status
    getStatus() {
        return {
            isDetecting: this.isDetecting,
            apiUrl: this.apiUrl,
            config: this.config
        };
    }

    // Update API configuration at runtime
    updateApiConfig(newConfig) {
        if (newConfig.BASE_URL) {
            this.config.BASE_URL = newConfig.BASE_URL;
        }
        if (newConfig.ENDPOINT) {
            this.config.ENDPOINT = newConfig.ENDPOINT;
        }
        if (newConfig.TIMEOUT) {
            this.config.TIMEOUT = newConfig.TIMEOUT;
        }
        if (newConfig.RETRY_ATTEMPTS) {
            this.config.RETRY_ATTEMPTS = newConfig.RETRY_ATTEMPTS;
        }
        
        // Update the full API URL
        this.apiUrl = this.config.BASE_URL + this.config.ENDPOINT;
        
        console.log('API configuration updated:', this.config);
        console.log('New API URL:', this.apiUrl);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ASLDetector;
}
