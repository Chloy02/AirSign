# Screen Sharing App with Chat

A real-time video conferencing application with screen sharing and chat functionality similar to Google Meet.

## Features

### Video Conferencing
- Real-time video and audio communication
- Screen sharing capability
- Multiple participants support
- WebRTC peer-to-peer connections

### Chat System
- **Real-time messaging**: Send and receive messages instantly
- **Message persistence**: Chat history is saved and loaded when joining a room
- **Participant list**: See who's currently in the meeting
- **Notifications**: Browser notifications for new messages when chat is closed
- **Unread message counter**: Badge showing number of unread messages
- **Auto-resize textarea**: Input field automatically resizes as you type
- **Keyboard shortcuts**: 
  - `Ctrl/Cmd + Enter`: Toggle chat
  - `Escape`: Close chat
  - `Enter`: Send message
  - `Shift + Enter`: New line

### ASL Detection System
- **Real-time ASL detection**: American Sign Language recognition using AI
- **Configurable API endpoints**: Easy to change API URLs without code modification
- **Visual configuration interface**: User-friendly modal for updating settings
- **Configuration persistence**: Settings saved across browser sessions
- **Fallback safety**: Default values ensure the app always works
- **Multiple configuration methods**: 
  - Visual interface (recommended)
  - Direct code modification
  - Runtime programmatic updates

### UI Features
- **Modern design**: Clean, Google Meet-inspired interface
- **Responsive layout**: Works on desktop and mobile devices
- **Dark theme**: Easy on the eyes for long meetings
- **Smooth animations**: Chat sidebar slides in/out smoothly

## How to Use Chat

1. **Open Chat**: Click the chat icon in the top-right corner or press `Ctrl/Cmd + Enter`
2. **Send Message**: Type in the input field and press Enter or click the send button
3. **View Participants**: See the list of participants in the meeting
4. **Chat History**: Previous messages are automatically loaded when you join
5. **Notifications**: Allow browser notifications to get notified of new messages

## How to Use ASL Detection

1. **Start ASL Detection**: Click the ASL button (👐) in the meeting controls
2. **Configure API Endpoint**: Click the gear icon (⚙️) next to the ASL button to open configuration
3. **Update Settings**: 
   - **Base URL**: Your API domain (e.g., `https://your-api.com`)
   - **Endpoint**: API path (e.g., `/detect-asl`)
   - **Timeout**: Request timeout in milliseconds
   - **Retry Attempts**: Number of retry attempts
4. **Save Configuration**: Click "Save Configuration" to apply changes
5. **Configuration Persistence**: Settings are automatically saved and restored

### Testing Configuration
- Open `config-test.html` to test the configuration system
- Verify API endpoints are working correctly
- Test configuration updates and persistence

## Technical Details

### Backend
- **Node.js** with Express server
- **Socket.IO** for real-time communication
- **MongoDB** for storing chat messages and user data
- **JWT** for authentication

### Frontend
- **Vanilla JavaScript** with WebRTC
- **Responsive CSS** with modern design
- **Font Awesome** icons
- **Browser notifications** API

### Database Schema
```javascript
// Chat Message Schema
{
  roomId: String,
  sender: String,
  message: String,
  timestamp: Date
}
```

## File Structure

```
screen-sharing-app-multi/
├── public/
│   ├── js/
│   │   ├── config.js          # Configuration management
│   │   ├── aslDetector.js     # ASL detection system
│   │   └── client.js          # Main client logic
│   ├── meeting.html           # Main meeting interface
│   ├── config-test.html       # Configuration testing page
│   └── ...
├── CONFIGURATION.md           # Detailed configuration guide
└── ...
```

## Installation

1. Install dependencies:
```bash
npm install
```

2. Set up MongoDB (Optional but recommended for full features):
```bash
# Option A: Install MongoDB locally
# Download from: https://www.mongodb.com/try/download/community
# Or use the provided scripts:
# Windows: Double-click start-mongodb.bat
# PowerShell: Run start-mongodb.ps1

# Option B: Use MongoDB Atlas (cloud)
# Sign up at: https://www.mongodb.com/atlas
# Update connection string in server.js

# Option C: Run without MongoDB (Demo Mode)
# The app will work with limited features
# No user accounts or chat history persistence
```

3. Generate SSL certificates (for HTTPS):
```bash
# Place your SSL certificates in the ssl/ folder
# key.pem and cert.pem
```

4. Start the server:
```bash
npm start
```

5. Access the application:
```
https://localhost:3000
```

## Chat API Endpoints

- `GET /api/chat/:roomId` - Get chat history for a room
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login

## Socket Events

- `chatMessage` - Send/receive chat messages
- `join` - Join a meeting room
- `userJoined` - New user joined
- `userLeft` - User left the meeting

## Browser Support

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Security Features

- JWT-based authentication
- HTTPS encryption
- Input sanitization for chat messages
- Secure WebRTC connections

## Troubleshooting

### MongoDB Connection Issues

If you see this error:
```
MongoDB connection error: connect ECONNREFUSED ::1:27017, connect ECONNREFUSED 127.0.0.1:27017
```

**Solutions:**

1. **Start MongoDB locally:**
   - Windows: Double-click `start-mongodb.bat`
   - PowerShell: Run `start-mongodb.ps1`
   - Command line: `mongod --dbpath ./data/db`

2. **Install MongoDB as a service:**
   ```bash
   # Windows (as Administrator)
   mongod --install --dbpath "C:\data\db"
   net start MongoDB
   ```

3. **Use MongoDB Atlas (cloud):**
   - Sign up at https://www.mongodb.com/atlas
   - Get connection string and update `server.js`

4. **Run in Demo Mode:**
   - The app will work without MongoDB
   - Limited features (no user accounts, no chat history)
   - Use `/api/demo/login` endpoint

### Health Check

Check server status at: `https://localhost:3000/api/health`

### Demo Mode

When MongoDB is unavailable, the app automatically switches to demo mode:
- No user registration/login required
- Chat works but messages aren't saved
- Video conferencing works normally
- ASL detection works normally

## Future Enhancements

- File sharing in chat
- Emoji reactions
- Message editing/deletion
- Chat search functionality
- Message threading
- Read receipts 