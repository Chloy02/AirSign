@echo off
echo Starting MongoDB...
echo.
echo If MongoDB is not installed, you can:
echo 1. Download from: https://www.mongodb.com/try/download/community
echo 2. Install MongoDB as a service
echo 3. Or use MongoDB Atlas (cloud): https://www.mongodb.com/atlas
echo.
echo Starting MongoDB daemon...
mongod --dbpath ./data/db
pause
