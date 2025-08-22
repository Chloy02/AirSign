Write-Host "Starting MongoDB..." -ForegroundColor Green
Write-Host ""
Write-Host "If MongoDB is not installed, you can:" -ForegroundColor Yellow
Write-Host "1. Download from: https://www.mongodb.com/try/download/community" -ForegroundColor Cyan
Write-Host "2. Install MongoDB as a service" -ForegroundColor Cyan
Write-Host "3. Or use MongoDB Atlas (cloud): https://www.mongodb.com/atlas" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting MongoDB daemon..." -ForegroundColor Green

# Create data directory if it doesn't exist
if (!(Test-Path "./data/db")) {
    New-Item -ItemType Directory -Path "./data/db" -Force
    Write-Host "Created data directory: ./data/db" -ForegroundColor Yellow
}

# Start MongoDB
try {
    mongod --dbpath ./data/db
} catch {
    Write-Host "Error starting MongoDB: $_" -ForegroundColor Red
    Write-Host "Make sure MongoDB is installed and in your PATH" -ForegroundColor Red
}

Read-Host "Press Enter to continue"
